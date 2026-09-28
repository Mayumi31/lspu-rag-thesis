import os
import json
import re
import hashlib
import argparse
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef, Literal

import faiss
from sentence_transformers import SentenceTransformer

from anthropic import Anthropic

load_dotenv()

# Avoids a known tokenizer parallelism warning/hang on some machines
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# RAGAS is only needed for the optional evaluation step, not the interactive
# demo. Some ragas/langchain_community version combos conflict on import, so
# we import it lazily (inside run_ragas_on_mode) instead of at module load
# time. This flag lets main() decide whether to even attempt evaluation.
RAGAS_AVAILABLE = True
RAGAS_IMPORT_ERROR = None
try:
    from datasets import Dataset  # noqa: F401
    from ragas import evaluate  # noqa: F401
    from ragas.metrics import (  # noqa: F401
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
except Exception as e:  # pragma: no cover
    RAGAS_AVAILABLE = False
    RAGAS_IMPORT_ERROR = e

# =========================
# Config
# =========================
MODEL_NAME = "claude-haiku-4-5-20251001"  # swap for a different Claude model if you like
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_DIR = "data"
PILLAR3_TTL = os.path.join(DATA_DIR, "pillar3_handbook_ontology.ttl")
PILLAR3_TXT = os.path.join(DATA_DIR, "pillar3_handbook_source.txt")  # optional baseline corpus
EVAL_SET = os.path.join("eval", "eval_set.jsonl")

# On-disk embedding cache so we never re-embed unless the corpus changed
CACHE_DIR = "cache"
EMB_CACHE_NPY = os.path.join(CACHE_DIR, "pillar3_embeddings.npy")
EMB_CACHE_META = os.path.join(CACHE_DIR, "pillar3_embeddings_meta.json")

# Smaller batches = lower peak RAM. Drop to 8 if your laptop still struggles.
EMBED_BATCH_SIZE = 16

ALL_MODES = ["bare_llm", "basic_rag", "hybrid_rag", "ontology_graph_only", "ontology_contextual_rag"]


# =========================
# Utilities
# =========================
# Matches the start of a new Chapter or Article heading (e.g. "Chapter 6",
# "Article 6. Procedure for Major Disciplinary Actions"). Used to keep
# fixed-window chunking from straddling two unrelated legal sections.
_STRUCTURE_BOUNDARY_RE = re.compile(r"\n(?=(?:Chapter\s+\d+\b|Article\s+\d+\.))")


def _split_into_structural_sections(text: str) -> List[str]:
    """
    Splits the handbook on Chapter/Article headers so a single chunk can
    never straddle two unrelated sections. This is what let e.g. a
    "suspension of classes" force-majeure clause (end of Article 5) and
    the start of "Article 6. Procedure for Major Disciplinary Actions"
    land in the same retrieved chunk — they're back-to-back in the raw
    text with no blank-line gap between them, so the old sentence-boundary
    heuristic had no way to know they're different topics.
    Falls back to the whole text as one section if no headers are found
    (e.g. a differently-formatted source file), so this never produces
    zero chunks.
    """
    parts = [p.strip() for p in _STRUCTURE_BOUNDARY_RE.split(text) if p.strip()]
    return parts if parts else [text]


def _chunk_one_section(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        # try not to cut mid-sentence if possible
        if end < n:
            last_period = text[start:end].rfind(". ")
            if last_period > chunk_size * 0.6:
                end = start + last_period + 2
        chunks.append(text[start:end].strip())
        if end >= n:
            break  # reached the end of the section — stop
        start = max(start + 1, end - overlap)  # always move forward
    return [c for c in chunks if c]


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks = []
    for section in _split_into_structural_sections(text):
        # overlap only ever bleeds within the same Article/Chapter now,
        # never across one boundary into the next
        chunks.extend(_chunk_one_section(section, chunk_size, overlap))
    return [c for c in chunks if c]


def safe_str(x) -> str:
    if x is None:
        return ""
    return str(x)


# =========================
# Claude client
# =========================
class ClaudeClient:
    def __init__(self):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY in environment/.env")
        self.client = Anthropic(api_key=key)

    def generate(self, system: str, user: str, max_tokens: int = 650) -> str:
        resp = self.client.messages.create(
            model=MODEL_NAME,
            max_tokens=max_tokens,
            temperature=0.2,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # resp.content is a list of content blocks
        return "".join([b.text for b in resp.content if hasattr(b, "text")]).strip()


# =========================
# Vector Store
# =========================
@dataclass
class RetrievedChunk:
    chunk_id: str
    source: str
    text: str
    score: float


class VectorIndex:
    def __init__(self, embedder: SentenceTransformer):
        self.embedder = embedder
        self.index = None
        self.id_map: List[str] = []
        self.meta: Dict[str, Dict[str, Any]] = {}

    def build_from_embeddings(self, items: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Build the FAISS index from precomputed embeddings (either loaded
        from the on-disk cache, or freshly encoded via build()).
        """
        embeddings = np.ascontiguousarray(np.asarray(embeddings), dtype="float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.id_map = [it["id"] for it in items]
        for it in items:
            self.meta[it["id"]] = it

    def build(self, items: List[Dict[str, Any]]) -> np.ndarray:
        """
        items: [{id, text, source, ...}]

        Encodes all items with a visible progress bar and smaller batches
        (lower peak RAM), then builds the index. Returns the embeddings so
        the caller can save them to disk for next time.
        """
        texts = [it["text"] for it in items]
        embeddings = self.embedder.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        embeddings = np.asarray(embeddings, dtype="float32")
        self.build_from_embeddings(items, embeddings)
        return embeddings

    def search(self, query: str, k: int = 5, min_score: float = 0.0) -> List[RetrievedChunk]:
        if self.index is None:
            return []
        q = self.embedder.encode([query], normalize_embeddings=True)
        q = np.array(q, dtype="float32")
        scores, idxs = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            if float(score) < min_score:
                # index.search returns results sorted by score descending,
                # so once we're below threshold everything after is too.
                break
            chunk_id = self.id_map[idx]
            m = self.meta[chunk_id]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    source=m.get("source", "unknown"),
                    text=m.get("text", ""),
                    score=float(score),
                )
            )
        return results

    @staticmethod
    def dedupe_chunks(chunks: List[RetrievedChunk], overlap_threshold: float = 0.7) -> List[RetrievedChunk]:
        """
        Drop near-duplicate chunks (common with overlapping chunking, where
        adjacent chunks repeat the same sentence(s)). Cheap word-set Jaccard
        similarity is enough here since duplicates are near-verbatim, not
        paraphrases. Keeps the higher-scored chunk of any near-duplicate pair.
        """
        kept: List[RetrievedChunk] = []
        kept_word_sets: List[set] = []
        for c in sorted(chunks, key=lambda x: x.score, reverse=True):
            words = set(re.findall(r"[a-z0-9]+", c.text.lower()))
            is_dupe = False
            for wset in kept_word_sets:
                if not words or not wset:
                    continue
                jaccard = len(words & wset) / len(words | wset)
                if jaccard >= overlap_threshold:
                    is_dupe = True
                    break
            if not is_dupe:
                kept.append(c)
                kept_word_sets.append(words)
        return kept

    @staticmethod
    def rescore_lexical(query: str, chunks: List[RetrievedChunk], lexical_weight: float = 0.25) -> List[RetrievedChunk]:
        """
        Cheap lexical-overlap rescoring (same idea already used in
        answer_hybrid_rag) reused for the vector half of Ontology
        Contextual RAG. A pure bi-encoder (all-MiniLM-L6-v2) over a
        legalistic, boilerplate-heavy document tends to score generic
        phrasing ("suspension", "offense", "sanction") as similar even
        across unrelated sections (e.g. "suspension of classes" due to
        force majeure vs. "suspension" as a disciplinary sanction).
        Blending in raw token overlap punishes those false positives
        without needing a bigger/different embedding model.
        Returns the same chunks, re-sorted, with .score replaced by the
        blended score (kept for transparency in the trace).
        """
        q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        rescored = []
        for c in chunks:
            t_tokens = set(re.findall(r"[a-z0-9]+", c.text.lower()))
            lexical = len(q_tokens & t_tokens) / (len(q_tokens) + 1e-6)
            blended = (1 - lexical_weight) * c.score + lexical_weight * lexical
            rescored.append(RetrievedChunk(chunk_id=c.chunk_id, source=c.source, text=c.text, score=blended))
        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored


# =========================
# Ontology (Graph) layer
# =========================
@dataclass
class GraphHit:
    node: str
    label: str
    definition: str
    source_article: str
    relations: List[Tuple[str, str]]  # (predicate, neighbor label) for labeled links
    score: float = 0.0  # relevance score to the query, for threshold filtering/precision
    # All literal-valued properties of the node (durations, fees, requirements,
    # CMO numbers...) — not just lspu:definition. Nodes from the newer pillars
    # keep their content here, so reading only `definition` made them look empty.
    facts: List[Tuple[str, str]] = field(default_factory=list)
    # 1-hop neighbors resolved to readable text: (predicate, neighbor label, neighbor text)
    neighbors: List[Tuple[str, str, str]] = field(default_factory=list)
    # 1-hop *incoming* links: nodes that point AT this node via a non-structural
    # object property (e.g. the parent Service of a Step, the SanctionTier of a
    # Sanction). Surfaces roll-up/aggregate facts (total time, total fee) that
    # live on the parent even when the child node is the semantic best match.
    parents: List[Tuple[str, str, List[Tuple[str, str]]]] = field(default_factory=list)
    # The exact text shown to the LLM for this hit. The same string is passed
    # to RAGAS as this hit's context, so scoring sees what the model saw.
    frame: str = ""


class OntologyStore:
    def __init__(self, ttl_path: str, embedder: Optional[SentenceTransformer] = None):
        self.ttl_path = ttl_path
        self.g = Graph()
        self.LSPU = Namespace("http://lspu.edu.ph/ontology/handbook#")
        # The merged TTL actually carries THREE pillar namespaces, not one:
        #   lspu:      handbook / conduct policy (Pillar 3)
        #   lspuprog:  program information (Pillar 1)
        #   lspuadm:   admission & Citizen's Charter procedures (Pillar 2)
        # Definition/source-citation predicates differ per pillar (see
        # _DEFINITION_PREDS/_SOURCE_PREDS below) — treating lspu: as the only
        # namespace silently blanked the DEFINITION/SOURCE ARTICLE fields for
        # every Pillar 1/2 node.
        self.LSPUPROG = Namespace("http://lspu.edu.ph/ontology/program#")
        self.LSPUADM = Namespace("http://lspu.edu.ph/ontology/admission#")
        # Ordered by priority: first one present on the node wins. rdfs:comment
        # is last and is only ever read on non-class/non-property subjects
        # (see _definition), since on classes/properties it's schema
        # documentation, not a fact about an instance.
        self._DEFINITION_PREDS = [self.LSPU.definition, self.LSPUPROG.programOverview, RDFS.comment]
        self._SOURCE_PREDS = [
            self.LSPU.sourceArticle,
            self.LSPUADM.sourceSection,
            self.LSPUPROG.sourceDocument,
            self.LSPUADM.sourceDocument,
        ]
        # Optional embedder enables semantic (not just lexical) node ranking,
        # which is what actually fixes low context_precision: lexical
        # substring/token matching returns loosely-related nodes with no
        # sense of "how relevant", so a fixed limit just gets filled with
        # noise. Semantic similarity lets us rank AND threshold properly.
        self.embedder = embedder
        self._node_ids: List[Any] = []
        self._node_embeddings: Optional[np.ndarray] = None

    def load(self):
        if not os.path.exists(self.ttl_path):
            raise FileNotFoundError(f"TTL not found: {self.ttl_path}")
        self.g.parse(self.ttl_path, format="turtle")
        if self.embedder is not None:
            self._build_node_embeddings()

    # Predicates that describe graph structure rather than content. They never
    # count as facts/links and never make a node "retrievable" on their own.
    _STRUCTURAL_PREDS = {
        RDF.type, RDFS.subClassOf, RDFS.subPropertyOf, RDFS.domain, RDFS.range,
        RDFS.label, RDFS.isDefinedBy,
    }

    def _is_structural(self, p) -> bool:
        s = str(p)
        return (
            p in self._STRUCTURAL_PREDS
            or s.startswith("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
            or s.startswith("http://www.w3.org/2002/07/owl#")
        )

    @staticmethod
    def _natural_key(text: str):
        # "Step 2" sorts before "Step 10"
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]

    def _pred_name(self, p) -> str:
        """lspu:processingTime -> 'processing time' (readable, and better for embeddings)."""
        name = re.split(r"[#/:]", str(p))[-1]
        return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name).replace("_", " ").lower().strip()

    def _is_schema_node(self, s) -> bool:
        """
        True for owl:Class / rdfs:Class subjects — i.e. vocabulary/schema
        definitions like `lspuprog:CMO a owl:Class` or `lspuprog:AdmissionRequirement
        a owl:Class`, which typically carry an rdfs:comment describing what the
        TERM means in general, not a fact about the university. These are not
        evidence for any student question and must never be indexed as if they
        were content — doing so is what produced hits like "CHED Memorandum
        Order (CMO): <textbook definition>" or "Admission Requirement: <textbook
        definition>" crowding out the *actual* CMO/requirement instances (or,
        worse, standing in for them when no real instance matches).
        """
        return (s, RDF.type, OWL.Class) in self.g or (s, RDF.type, RDFS.Class) in self.g

    def _definition(self, s) -> str:
        """First non-empty value across this pillar's definition-like predicates.
        rdfs:comment is only honored here because _has_content already excludes
        schema nodes (owl:Class/rdfs:Class), so by the time we get here a
        comment is annotating an *instance* (e.g. "Follows the same 5-step
        process as ...") rather than documenting a vocabulary term."""
        for p in self._DEFINITION_PREDS:
            v = self._literal(s, p)
            if v:
                return v
        return ""

    def _source(self, s) -> str:
        for p in self._SOURCE_PREDS:
            v = self._literal(s, p)
            if v:
                return v
        return ""

    def _own_facts(self, s) -> List[Tuple[str, str]]:
        """Every literal property of a node except whichever definition/source
        predicate was used for the DEFINITION/SOURCE ARTICLE lines (so it isn't
        shown twice)."""
        skip = set(self._DEFINITION_PREDS) | set(self._SOURCE_PREDS)
        facts = []
        for p, o in self.g.predicate_objects(subject=s):
            if not isinstance(o, Literal) or self._is_structural(p) or p in skip:
                continue
            val = safe_str(o).strip()
            if val:
                facts.append((self._pred_name(p), val))
        facts.sort(key=lambda kv: (self._natural_key(kv[0]), self._natural_key(kv[1])))
        return facts

    def _is_class_node(self, o) -> bool:
        return (o, RDF.type, OWL.Class) in self.g or (o, RDF.type, RDFS.Class) in self.g

    def _links(self, s) -> List[Tuple[str, Any]]:
        """Non-structural links to nodes that have a readable label, in natural order."""
        links = []
        for p, o in self.g.predicate_objects(subject=s):
            if not isinstance(o, URIRef) or self._is_structural(p) or self._is_class_node(o):
                continue
            if self._label(o) or self._definition(o):
                links.append((self._pred_name(p), o))
        links.sort(key=lambda po: (po[0], self._natural_key(self._label(po[1]) or str(po[1]))))
        return links

    def _parents(self, s, max_parents: int = 2) -> List[Tuple[str, str, List[Tuple[str, str]]]]:
        """
        Nodes that point AT s via a non-structural object property — e.g. the
        parent Service of a ServiceStep (lspu:hasStep), or the SanctionTier
        that imposes a given Sanction (imposesSanction). Ontologies routinely
        put roll-up facts (total processing time, total fee) on the PARENT
        while the CHILD is what best matches a specific query — e.g. a query
        about "Good Moral Certificate processing time" semantically matches
        "Step 3: Processing of Good Moral Certificate" (20 minutes, one step)
        at least as well as it matches the parent Service (which actually
        holds the correct total: 1 hour and 15 minutes). Surfacing the
        parent's own facts alongside the child prevents the model from
        answering with just one step's partial figure.
        """
        seen_subjects = set()
        out = []

        # Step-chain rollup FIRST (and prioritized): a ServiceStep only sits
        # inside a Service via lspu:hasStep on the *first* step, chained
        # onward with lspu:nextStep — so Step 3 has no direct reverse link to
        # its Service at all, only to Step 2 (via nextStep). Walk backward to
        # the chain's root step, then find whichever node lspu:hasStep's that
        # root; that is the actual owning Service, and it is what carries the
        # roll-up totals (total processing time, total fee).
        has_next_step_link = bool(list(self.g.subjects(self.LSPU.nextStep, s))) or bool(
            list(self.g.objects(s, self.LSPU.nextStep))
        )
        if has_next_step_link:
            root = s
            for _ in range(20):  # generous cap; real chains are a handful of steps
                preds = list(self.g.subjects(self.LSPU.nextStep, root))
                if not preds:
                    break
                root = preds[0]
            for owner in self.g.subjects(self.LSPU.hasStep, root):
                if owner in seen_subjects:
                    continue
                o_facts = self._own_facts(owner)
                if not o_facts:
                    continue
                seen_subjects.add(owner)
                out.append(("part of", self._label(owner) or self._shorten(owner), o_facts))

        # Generic 1-hop reverse links (e.g. SanctionTier --imposesSanction--> Sanction).
        # lspu:nextStep is excluded here: it points from the PREVIOUS step, a
        # sibling in the same chain, not a parent — following it would surface
        # an unrelated step's own partial figures instead of the true rollup.
        for subj, pred in self.g.subject_predicates(object=s):
            if pred == self.LSPU.nextStep:
                continue
            if not isinstance(subj, URIRef) or self._is_structural(pred):
                continue
            if subj in seen_subjects or self._is_schema_node(subj) or self._is_property_decl(subj):
                continue
            p_facts = self._own_facts(subj)
            if not p_facts:
                continue
            seen_subjects.add(subj)
            out.append((self._pred_name(pred), self._label(subj) or self._shorten(subj), p_facts))
            if len(out) >= max_parents:
                break
        return out[:max_parents]

    _PROPERTY_TYPES = (OWL.ObjectProperty, OWL.DatatypeProperty, OWL.AnnotationProperty, RDF.Property)

    def _is_property_decl(self, s) -> bool:
        return any((s, RDF.type, t) in self.g for t in self._PROPERTY_TYPES)

    def _has_content(self, s) -> bool:
        """A node can only serve as evidence if it says something beyond its own name.
        Property declarations (schema like `occurrenceOrdinal`) never count, and
        neither do owl:Class/rdfs:Class vocabulary nodes (schema documentation,
        not facts — see _is_schema_node)."""
        if self._is_property_decl(s) or self._is_schema_node(s):
            return False
        return bool(
            self._definition(s)
            or self._own_facts(s)
            or self._links(s)
        )

    def _node_text(self, s) -> str:
        """Text embedded for retrieval: label + definition + facts + labeled links."""
        parts = [self._label(s), self._definition(s)]
        parts += [f"{p}: {v}" for p, v in self._own_facts(s)]
        parts += [f"{p}: {self._label(o) or self._definition(o)}" for p, o in self._links(s)]
        return ". ".join(x.strip() for x in parts if x and x.strip())

    def _neighbors(self, s, max_neighbors: int = 8, max_chars: int = 280) -> List[Tuple[str, str, str]]:
        """1-hop expansion: steps, requirements, sanctions... resolved to readable text."""
        out = []
        for pred, o in self._links(s):
            bits = [self._definition(o)] + [f"{p}: {v}" for p, v in self._own_facts(o)]
            text = "; ".join(b.strip() for b in bits if b and b.strip())[:max_chars]
            out.append((pred, self._label(o), text))
            if len(out) >= max_neighbors:
                break
        return out

    def _render_frame(self, label, definition, source_article, facts, neighbors, parents=None) -> str:
        lines = []
        if label:
            lines.append(f"LABEL: {label}")
        if definition:
            lines.append(f"DEFINITION: {definition}")
        if source_article:
            lines.append(f"SOURCE ARTICLE: {source_article}")
        if facts:
            lines.append("FACTS:")
            lines += [f"  - {p}: {v}" for p, v in facts]
        if neighbors:
            lines.append("RELATED NODES:")
            for p, lbl, txt in neighbors:
                head = f"  - {p}" + (f" -> {lbl}" if lbl else "")
                lines.append(head + (f": {txt}" if txt else ""))
        if parents:
            lines.append("PART OF / REFERENCED BY (use these totals/roll-ups when the question asks for an overall or total figure):")
            for pred, lbl, p_facts in parents:
                lines.append(f"  - {pred} of: {lbl}")
                lines += [f"      - {p}: {v}" for p, v in p_facts]
        return "\n".join(lines)

    def _build_node_embeddings(self):
        """
        Precompute one embedding per *retrievable* node. Text = label +
        definition + all literal facts + labeled links, so a node whose
        content lives in properties other than lspu:definition (process steps,
        programs, services) still embeds — and can be found — by what it says.
        Nodes with no content beyond their own name, and owl:Class/rdfs:Class
        vocabulary nodes, are left out of the index: they can't answer
        anything (or answer with schema documentation instead of a fact),
        and used to surface as noise that pushed context_precision down.
        """
        self._node_ids = [
            s for s in sorted(set(self.g.subjects()), key=lambda s: str(s)) if self._has_content(s)
        ]
        texts = [self._node_text(s) for s in self._node_ids]
        if not texts:
            self._node_embeddings = None
            return
        self._node_embeddings = np.asarray(
            self.embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True),
            dtype="float32",
        )

    def _literal(self, s, p) -> str:
        v = self.g.value(s, p)
        return safe_str(v) if isinstance(v, Literal) else safe_str(v)

    def _label(self, s) -> str:
        return self._literal(s, RDFS.label)

    def search_nodes_by_text(
        self,
        query: str,
        limit: int = 3,
        min_score: float = 0.35,
        score_margin: float = 0.08,
        expand_neighbors: bool = True,
    ) -> List[GraphHit]:
        """
        Node search with a relevance score attached, so the caller can
        threshold instead of always returning `limit` nodes regardless of
        how relevant they are.

        Prefers semantic similarity (cosine sim over sentence-transformer
        embeddings of label+definition) when an embedder was supplied to
        the store; this ranks nodes by *how* relevant they are, not just
        whether a substring/token happens to match. Falls back to the old
        lexical approach (with a synthetic score) if no embedder is set.

        score_margin: beyond the absolute min_score floor, a 2nd/3rd hit
        is only kept if it's within `score_margin` of the TOP hit's score.
        The ontology has ~280 near-identical Sanction/SanctionTier nodes
        (one per offense), so for almost any query using generic
        institutional language, *some* sanction node will clear a flat
        absolute floor even when it's completely unrelated (e.g. a
        "Falsification of Documents" sanction node showing up for a
        question about the 30-day document-issuance right, just because
        both mention "official documents"). An absolute floor alone can't
        tell "clearly related" apart from "barely related" — a relative
        gap from the best match can.
        """
        q = query.strip()
        if not q:
            return []

        if self.embedder is not None and self._node_embeddings is not None and len(self._node_ids) > 0:
            scored = self._search_nodes_semantic(q, limit=limit, min_score=min_score, score_margin=score_margin)
        else:
            scored = self._search_nodes_lexical(q, limit=limit)

        out = []
        for s, score in scored:
            label = self._label(s)
            definition = self._definition(s)
            source_article = self._source(s)
            facts = self._own_facts(s)
            neighbors = self._neighbors(s) if expand_neighbors else []
            parents = self._parents(s)
            out.append(
                GraphHit(
                    node=self._shorten(s),
                    label=label,
                    definition=definition,
                    source_article=source_article,
                    relations=[(p, lbl) for p, lbl, _ in neighbors],
                    score=score,
                    facts=facts,
                    neighbors=neighbors,
                    parents=parents,
                    frame=self._render_frame(label, definition, source_article, facts, neighbors, parents),
                )
            )
        return out

    def _search_nodes_semantic(
        self, query: str, limit: int, min_score: float, score_margin: float = 0.08
    ) -> List[Tuple[Any, float]]:
        q_emb = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = self._node_embeddings @ q_emb  # cosine sim (both sides normalized)
        order = np.argsort(-sims)
        results = []
        top_score = None
        for idx in order:
            score = float(sims[idx])
            if score < min_score:
                break  # sims is sorted descending, so we can stop early
            if top_score is None:
                top_score = score
            elif score < top_score - score_margin:
                break  # too far behind the best match to be "also relevant"
            results.append((self._node_ids[idx], score))
            if len(results) >= limit:
                break
        return results

    def _search_nodes_lexical(self, query: str, limit: int) -> List[Tuple[Any, float]]:
        q = query.lower().strip()
        exact_hits = []
        for s in set(self.g.subjects()):
            lbl = self._label(s).lower()
            definition = self._definition(s).lower()
            if q in lbl or q in definition:
                exact_hits.append((s, 1.0))

        if exact_hits:
            return exact_hits[:limit]

        # fallback: token-overlap scoring (ranked, not just a >=2 cutoff in
        # first-seen order) so at least the lexical path is precision-aware.
        tokens = [t for t in re.findall(r"[a-z0-9]+", q) if len(t) >= 4]
        scored = []
        for s in set(self.g.subjects()):
            lbl = self._label(s).lower()
            definition = self._definition(s).lower()
            hits = sum(1 for t in tokens if t in lbl or t in definition)
            if hits >= 2:
                scored.append((s, hits / max(len(tokens), 1)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def _shorten(self, uri) -> str:
        try:
            return self.g.namespace_manager.normalizeUri(uri)
        except Exception:
            return safe_str(uri)

    @staticmethod
    def dedupe_hits(hits: List["GraphHit"], overlap_threshold: float = 0.6) -> List["GraphHit"]:
        """
        Drop near-duplicate ontology hits. Common case: a SanctionTier node
        ("Sanction tier for 1st offense of X: imposes Suspension (1 month)")
        and the Sanction node it points to via imposesSanction ("Suspension
        sanction, duration one month, imposed for 1st offense of X") both
        get retrieved independently and say almost the same thing — that
        pads the context with filler instead of new information, which
        drags down context_precision. Same word-set Jaccard approach as
        VectorIndex.dedupe_chunks; keeps the higher-scored hit of any pair.
        """
        kept: List["GraphHit"] = []
        kept_word_sets: List[set] = []
        for h in sorted(hits, key=lambda x: x.score, reverse=True):
            text = f"{h.label} {h.definition} " + " ".join(f"{p} {v}" for p, v in h.facts)
            words = set(re.findall(r"[a-z0-9]+", text.lower()))
            is_dupe = False
            for wset in kept_word_sets:
                if not words or not wset:
                    continue
                jaccard = len(words & wset) / len(words | wset)
                if jaccard >= overlap_threshold:
                    is_dupe = True
                    break
            if not is_dupe:
                kept.append(h)
                kept_word_sets.append(words)
        return kept


# =========================
# Retrieval modes (Baselines + Ontology Contextual RAG)
# =========================
@dataclass
class AnswerTrace:
    mode: str
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    graph_hits: List[Dict[str, Any]]
    prompt_context_preview: str
    # Populated only for modes whose displayed answer includes non-factual
    # scaffolding (citations, node IDs, "how derived" commentary) that
    # shouldn't be judged as factual claims by RAGAS faithfulness. When set,
    # this — not the full displayed answer — is what gets scored.
    answer_for_eval: Optional[str] = None


class RAGSystem:
    def __init__(self, vector_index: Optional[VectorIndex], ontology: Optional[OntologyStore], llm: ClaudeClient):
        self.vector_index = vector_index
        self.ontology = ontology
        self.llm = llm

    def answer_bare_llm(self, query: str) -> Tuple[str, AnswerTrace]:
        system = (
            "You are an informational assistant for LSPU-LB students. "
            "Answer directly and confidently using your general knowledge of Philippine higher-education "
            "practices and typical university policy. Do not hedge, and do not say things like 'I'm not "
            "sure', 'I don't have that information', or 'please consult the handbook' — commit to a direct "
            "answer as if you were confident in it, the way a student would expect a straightforward reply."
        )
        user = f"Question: {query}\n\nAnswer concisely."
        ans = self.llm.generate(system=system, user=user)
        trace = AnswerTrace(
            mode="bare_llm",
            query=query,
            retrieved_chunks=[],
            graph_hits=[],
            prompt_context_preview="(no retrieval; bare LLM)",
        )
        return ans, trace

    def answer_basic_rag(self, query: str, k: int = 5) -> Tuple[str, AnswerTrace]:
        chunks = self.vector_index.search(query, k=k) if self.vector_index else []
        context = "\n\n".join([f"[{c.source} | score={c.score:.3f}]\n{c.text}" for c in chunks])
        system = (
            "You are an informational assistant for LSPU-LB. "
            "Answer ONLY using the provided context. "
            "If the context is insufficient, say so and ask for the correct document. "
            "Cite by quoting short phrases from the context."
        )
        user = f"Question: {query}\n\nCONTEXT:\n{context}\n\nAnswer:"
        ans = self.llm.generate(system=system, user=user)
        trace = AnswerTrace(
            mode="basic_rag",
            query=query,
            retrieved_chunks=[asdict(c) for c in chunks],
            graph_hits=[],
            prompt_context_preview=context[:1200],
        )
        return ans, trace

    def answer_hybrid_rag(self, query: str, k: int = 5) -> Tuple[str, AnswerTrace]:
        """
        Simple hybrid: vector top-k, rescored with a cheap lexical overlap
        boost. Just a baseline "hybrid" to compare against.
        """
        chunks = self.vector_index.search(query, k=max(12, k * 3)) if self.vector_index else []
        q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        rescored = []
        for c in chunks:
            t_tokens = set(re.findall(r"[a-z0-9]+", c.text.lower()))
            lexical = len(q_tokens.intersection(t_tokens)) / (len(q_tokens) + 1e-6)
            score = 0.75 * c.score + 0.25 * lexical
            rescored.append((score, c))
        rescored.sort(key=lambda x: x[0], reverse=True)
        top = [c for _, c in rescored[:k]]

        context = "\n\n".join([f"[{c.source} | score={c.score:.3f}]\n{c.text}" for c in top])
        system = (
            "You are an informational assistant for LSPU-LB. "
            "Answer ONLY using the provided context. "
            "If the context is insufficient, say so. "
            "Cite by quoting short phrases from the context."
        )
        user = f"Question: {query}\n\nCONTEXT:\n{context}\n\nAnswer:"
        ans = self.llm.generate(system=system, user=user)
        trace = AnswerTrace(
            mode="hybrid_rag",
            query=query,
            retrieved_chunks=[asdict(c) for c in top],
            graph_hits=[],
            prompt_context_preview=context[:1200],
        )
        return ans, trace

    # Markers the model is instructed to emit verbatim, so we can reliably
    # split "the answer" from "the citation/derivation trace" afterward.
    _FINAL_ANSWER_MARKER = "===FINAL_ANSWER==="
    _DERIVATION_MARKER = "===DERIVATION==="

    def answer_ontology_contextual_rag(
        self,
        query: str,
        k_vec: int = 3,
        k_graph: int = 2,
        vec_min_score: float = 0.25,
        graph_min_score: float = 0.45,
        vec_candidate_pool: int = 9,
        include_text: bool = True,
        mode: str = "ontology_contextual_rag",
    ) -> Tuple[str, AnswerTrace]:
        """
        Ontology Contextual RAG:
          1) Retrieve candidate graph nodes relevant to the query, ranked
             by semantic similarity and filtered by graph_min_score, then
             deduped (a SanctionTier node and the Sanction node it points
             to via imposesSanction often restate the same fact — keeping
             both pads context without adding information).
          2) Turn hits into a compact "contextual knowledge frame"
             (definitions, source article, relations, and — new — any
             PARENT node's own facts, so a Step/Tier-level hit still
             carries its Service/Sanction's roll-up totals).
          3) Combine with vector chunks (raw handbook text): pull a wider
             candidate pool at a looser embedding threshold, then rescore
             with lexical overlap (same trick as answer_hybrid_rag) before
             cutting down to k_vec. Deduped afterward so overlapping
             chunking doesn't repeat the same sentence twice.
          4) Force the model to answer strictly from those contexts, ONLY
             what the specific question asks (not adjacent steps/offenses
             that happen to be in the same context item), AND explain the
             derivation chain (thesis explainability requirement) — but
             behind an explicit marker, so the citation/explanation text
             (node IDs, "paraphrased as", etc.) never gets fed to RAGAS as
             if it were a factual claim about the handbook. Only the Final
             Answer section is scored.

        graph_min_score was raised from 0.40 to 0.45 alongside the
        _is_schema_node exclusion in OntologyStore: schema/vocabulary
        nodes (owl:Class subjects like `CHED Memorandum Order (CMO)` or
        `Admission Requirement`, which only ever carried a textbook-style
        rdfs:comment) are no longer indexed at all, which removes most of
        the score-0.35-0.42 false positives that used to fill the 2nd
        graph slot. 0.45 is a starting point — re-run `--diagnose` after
        any ontology edit to see the new score distribution before
        retuning further.

        include_text: set False for the graph-only ablation (see
        answer_ontology_graph_only below) — skips the vector-retrieval
        track entirely rather than just returning zero chunks, so the
        prompt doesn't even mention a text-context slot that's always
        empty. Graph-side settings (k_graph, graph_min_score, score_margin)
        are left untouched so this is a clean one-variable-at-a-time
        ablation against the full OC-RAG pipeline, not a differently-tuned
        graph search.
        """
        graph_hits_raw = (
            self.ontology.search_nodes_by_text(query, limit=k_graph, min_score=graph_min_score)
            if self.ontology
            else []
        )
        graph_hits = OntologyStore.dedupe_hits(graph_hits_raw)

        if include_text:
            vec_candidates = (
                self.vector_index.search(query, k=vec_candidate_pool, min_score=vec_min_score)
                if self.vector_index
                else []
            )
            vec_candidates = VectorIndex.rescore_lexical(query, vec_candidates)
            vec_chunks = VectorIndex.dedupe_chunks(vec_candidates)[:k_vec]
        else:
            vec_chunks = []

        graph_context = "\n\n".join(h.frame for h in graph_hits)

        vec_context = "\n\n".join([f"[{c.source} | score={c.score:.3f}]\n{c.text}" for c in vec_chunks])

        if include_text:
            system = (
                "You are an informational assistant for LSPU-LB.\n"
                "You must answer ONLY using the provided contexts (Ontology Context and Retrieved Text Context).\n"
                "Every factual claim in your answer must be directly supported by a specific sentence in one of "
                "the provided contexts — do not combine, extrapolate, or infer beyond what is explicitly stated.\n"
                "If a context item is only loosely related and does not directly support the answer, ignore it "
                "rather than blending it in.\n"
                "If the question asks for an overall, total, or combined figure (e.g. total processing time, "
                "total fee) and a context item has a 'PART OF / REFERENCED BY' section, prefer the total/roll-up "
                "figure given there over a single step's partial figure.\n"
                "Answer ONLY what THIS specific question asks. If a context item (e.g. a full multi-step "
                "procedure or a list of related offenses) contains adjacent information beyond what was asked, "
                "use only the part that directly answers the question — do not narrate surrounding steps, "
                "categories, or offenses the user didn't ask about, even if they're right there in the context.\n"
                "If one item bundles several rules (e.g. one for undergraduates and one for graduate students), "
                "give only the rule for the group the question is about — do not add rules for other groups.\n"
                "If insufficient, say what is missing (e.g., Pillar 1/2 documents not yet added).\n"
                "You must output your response in EXACTLY this structure, with the markers reproduced verbatim "
                "on their own line:\n"
                f"{self._FINAL_ANSWER_MARKER}\n"
                "<student-friendly final answer only — plain factual prose, no citations, no node names>\n"
                f"{self._DERIVATION_MARKER}\n"
                "<step-by-step derivation:\n"
                "   - which ontology nodes you used (by LABEL), marked as paraphrased since ontology "
                "definitions are not verbatim handbook text\n"
                "   - which retrieved text snippets you used, as direct quotes only (do not paraphrase these "
                "and present them as quotes)>\n"
                "Do not invent policies, sanctions, fees, or citations (e.g., BOR resolution numbers) that do "
                "not literally appear in the provided contexts."
            )
            user = (
                f"Question: {query}\n\n"
                f"ONTOLOGY CONTEXT (Pillar 3 / Handbook Ontology):\n{graph_context}\n\n"
                f"RETRIEVED TEXT CONTEXT (optional raw handbook chunks):\n{vec_context}\n\n"
                "Produce the required two sections, with the markers exactly as specified."
            )
        else:
            # Graph-only ablation: no text-context slot in the prompt at all,
            # so the model can't lean on "the text context is empty, ignore
            # it" — there simply isn't one to mention.
            system = (
                "You are an informational assistant for LSPU-LB.\n"
                "You must answer ONLY using the provided Ontology Context below — no other knowledge source "
                "is available in this mode.\n"
                "Every factual claim in your answer must be directly supported by a specific node's DEFINITION "
                "or FACTS in the Ontology Context — do not combine, extrapolate, or infer beyond what is "
                "explicitly stated in those definitions/facts.\n"
                "If a context item is only loosely related and does not directly support the answer, ignore it "
                "rather than blending it in.\n"
                "If the question asks for an overall, total, or combined figure (e.g. total processing time, "
                "total fee) and a node has a 'PART OF / REFERENCED BY' section, prefer the total/roll-up figure "
                "given there over a single step's partial figure.\n"
                "Answer ONLY what THIS specific question asks. If a node's definition contains adjacent "
                "information beyond what was asked, use only the part that directly answers the question.\n"
                "If one node bundles several rules (e.g. undergraduate and graduate), give only the rule for "
                "the group the question is about.\n"
                "If the Ontology Context is insufficient to answer, say so plainly rather than guessing.\n"
                "You must output your response in EXACTLY this structure, with the markers reproduced verbatim "
                "on their own line:\n"
                f"{self._FINAL_ANSWER_MARKER}\n"
                "<student-friendly final answer only — plain factual prose, no citations, no node names>\n"
                f"{self._DERIVATION_MARKER}\n"
                "<step-by-step derivation: which ontology nodes you used (by LABEL), marked as paraphrased "
                "since ontology definitions are not verbatim handbook text>\n"
                "Do not invent policies, sanctions, fees, or citations (e.g., BOR resolution numbers) that do "
                "not literally appear in the provided Ontology Context."
            )
            user = (
                f"Question: {query}\n\n"
                f"ONTOLOGY CONTEXT (Pillar 3 / Handbook Ontology):\n{graph_context}\n\n"
                "Produce the required two sections, with the markers exactly as specified."
            )
        raw = self.llm.generate(system=system, user=user, max_tokens=900)
        final_answer, full_display = self._split_final_answer(raw)

        trace_preview = (graph_context + ("\n\n---\n\n" + vec_context if include_text else ""))[:1200]
        trace = AnswerTrace(
            mode=mode,
            query=query,
            retrieved_chunks=[asdict(c) for c in vec_chunks],
            graph_hits=[asdict(h) for h in graph_hits],
            prompt_context_preview=trace_preview,
            answer_for_eval=final_answer,
        )
        return full_display, trace

    def answer_ontology_graph_only(
        self,
        query: str,
        k_graph: int = 2,
        graph_min_score: float = 0.45,
    ) -> Tuple[str, AnswerTrace]:
        """
        Ablation: OC-RAG with the text-retrieval track removed entirely —
        the model answers using ONLY the ontology's structured knowledge
        (definitions, source-article citations, relations), never seeing
        the raw handbook text. Same graph-search settings as full OC-RAG
        (k_graph, graph_min_score, the 0.08 score_margin baked into
        search_nodes_by_text's default) so this isolates exactly one
        variable: what the ontology alone can answer, versus what the
        ontology + text combination gets you in the full pipeline.
        """
        return self.answer_ontology_contextual_rag(
            query,
            k_graph=k_graph,
            graph_min_score=graph_min_score,
            include_text=False,
            mode="ontology_graph_only",
        )

    def _split_final_answer(self, raw: str) -> Tuple[str, str]:
        """
        Splits the model's marker-delimited output into (final_answer_only,
        full_display_text). Falls back gracefully to treating the whole
        response as both if the model didn't reproduce the markers exactly
        (rare with Claude, but don't let a formatting slip crash a run).
        """
        if self._FINAL_ANSWER_MARKER in raw and self._DERIVATION_MARKER in raw:
            after_marker = raw.split(self._FINAL_ANSWER_MARKER, 1)[1]
            final_answer = after_marker.split(self._DERIVATION_MARKER, 1)[0].strip()
            return final_answer, raw.strip()
        return raw.strip(), raw.strip()


# =========================
# Data loading / building (with embedding cache)
# =========================
def _chunks_fingerprint(items: List[Dict[str, Any]]) -> str:
    """
    Fingerprint covers the embed model + every chunk id/text, so the cache
    auto-invalidates if you edit the handbook OR change chunking.
    """
    h = hashlib.sha1()
    h.update(EMBED_MODEL_NAME.encode("utf-8"))
    for it in items:
        h.update(it["id"].encode("utf-8"))
        h.update(b"\x00")
        h.update(it["text"].encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


def build_vector_index_from_text_file(
    embedder: SentenceTransformer,
    txt_path: str,
    rebuild_cache: bool = False,
) -> Optional[VectorIndex]:
    if not os.path.exists(txt_path):
        print(f"[WARN] No text corpus found at {txt_path}. Basic/hybrid RAG will still run but without chunks.")
        return None

    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Chunking is cheap string work — safe to redo every run. It's the
    # EMBEDDING we're caching, not the chunks.
    chunks = chunk_text(text, chunk_size=1200, overlap=180)
    items = [
        {"id": f"pillar3_txt_chunk_{i}", "text": ch, "source": "pillar3_handbook_source.txt"}
        for i, ch in enumerate(chunks)
    ]
    fingerprint = _chunks_fingerprint(items)
    print(f"Prepared {len(items)} chunks from {os.path.basename(txt_path)}.")

    # --- 1) Try to load cached embeddings (fast path: no encoding at all) ---
    if not rebuild_cache and os.path.exists(EMB_CACHE_NPY) and os.path.exists(EMB_CACHE_META):
        try:
            with open(EMB_CACHE_META, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("fingerprint") == fingerprint:
                embeddings = np.load(EMB_CACHE_NPY)
                if embeddings.shape[0] == len(items):
                    idx = VectorIndex(embedder)
                    idx.build_from_embeddings(items, embeddings)
                    print(f"Loaded {embeddings.shape[0]} cached embeddings from {EMB_CACHE_NPY} — skipping encoding.")
                    return idx
        except Exception as e:
            print(f"[WARN] Embedding cache unreadable ({e}); re-encoding.")
    elif rebuild_cache:
        print("Rebuild requested (--rebuild-cache): ignoring existing cache.")

    # --- 2) Slow path: encode once, WITH progress bar, then save to disk ---
    print(f"Encoding {len(items)} chunks (batch_size={EMBED_BATCH_SIZE}). This happens once; next runs load from cache.")
    idx = VectorIndex(embedder)
    embeddings = idx.build(items)  # shows tqdm progress bar

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(EMB_CACHE_NPY, embeddings)
    with open(EMB_CACHE_META, "w", encoding="utf-8") as f:
        json.dump(
            {
                "fingerprint": fingerprint,
                "embed_model": EMBED_MODEL_NAME,
                "num_items": len(items),
            },
            f,
            indent=2,
        )
    print(f"Saved embedding cache to {EMB_CACHE_NPY}. Next run will load instantly.")
    return idx


def load_eval_set(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"[WARN] No eval set found at {path}. Create eval/eval_set.jsonl to run RAGAS.")
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# =========================
# RAGAS evaluation helper
# =========================
class _LocalSentenceTransformerEmbeddings:
    """
    Minimal LangChain-style embeddings adapter (embed_documents/embed_query/
    aembed_documents/aembed_query) backed directly by the app's own
    already-loaded SentenceTransformer.

    We deliberately do NOT use ragas' built-in HuggingfaceEmbeddings wrapper
    here: in several ragas releases that class is broken and can't even be
    instantiated (see ragas issue #1806 — it never implements the required
    aembed_documents/aembed_query methods, so instantiating it raises
    TypeError regardless of the constructor arguments passed). This tiny
    adapter sidesteps that bug entirely and is wrapped with ragas'
    LangchainEmbeddingsWrapper, which is a thin, stable, provider-agnostic
    adapter rather than a provider-specific implementation.
    """

    def __init__(self, embedder: SentenceTransformer):
        self.embedder = embedder

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.encode(list(texts), normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embedder.encode([text], normalize_embeddings=True)[0].tolist()

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


def build_ragas_llm_and_embeddings(embedder: SentenceTransformer):
    """
    By default ragas.evaluate() reaches for OpenAI for both the judge LLM
    and the embeddings it needs (e.g. for answer_relevancy) — which fails
    here since this project only configures an ANTHROPIC_API_KEY. This
    wires ragas up to use:
      - Claude as the judge LLM, via LangChain's ChatAnthropic wrapped in
        ragas' LangchainLLMWrapper. NOTE: this project deliberately does
        NOT use ragas.llms.llm_factory(provider="anthropic", ...) here —
        that returns a newer "InstructorLLM" object which only implements
        the modern instructor-based interface. The classic metrics we
        import directly from ragas.metrics (faithfulness, answer_relevancy,
        context_precision, context_recall) still expect the older
        BaseRagasLLM interface (agenerate_prompt etc.), which only
        LangchainLLMWrapper provides — using llm_factory here causes every
        judge call to fail silently and all scores to come back as NaN.
      - The same local sentence-transformers embedder the app already
        loads (via a tiny custom adapter — see _LocalSentenceTransformerEmbeddings),
        so no OpenAI embeddings key is needed either.
    Returns (ragas_llm, ragas_embeddings). Either may be None if the
    required package/class isn't available; callers should skip passing
    that kwarg to evaluate() in that case.
    """
    from ragas.llms import LangchainLLMWrapper

    ragas_llm = None
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as e:
        print(
            "[WARN] langchain-anthropic is not installed, so ragas has no "
            f"way to call Claude as its judge LLM ({e}). "
            "Install it with: pip install langchain-anthropic"
        )
    else:
        chat_model = ChatAnthropic(
            model=MODEL_NAME,
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            temperature=0.0,
        )
        ragas_llm = LangchainLLMWrapper(chat_model)

    ragas_embeddings = None
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        ragas_embeddings = LangchainEmbeddingsWrapper(_LocalSentenceTransformerEmbeddings(embedder))
    except Exception as e:  # pragma: no cover
        print(f"[WARN] Could not wrap local embeddings for ragas ({e}); continuing without embeddings=.")
        ragas_embeddings = None

    return ragas_llm, ragas_embeddings


def run_ragas_on_mode(
    system: RAGSystem,
    eval_rows: List[Dict[str, Any]],
    mode: str,
    ragas_llm=None,
    ragas_embeddings=None,
) -> Dict[str, Any]:
    """
    eval_rows expects jsonl lines like:
      {"question": "...", "ground_truth": "..."}
    """
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for r in eval_rows:
        q = r["question"]
        gt = r.get("ground_truth", "")

        if mode == "bare_llm":
            a, tr = system.answer_bare_llm(q)
            ctx = []  # no retrieved contexts
        elif mode == "basic_rag":
            a, tr = system.answer_basic_rag(q)
            ctx = [c["text"] for c in tr.retrieved_chunks]
        elif mode == "hybrid_rag":
            a, tr = system.answer_hybrid_rag(q)
            ctx = [c["text"] for c in tr.retrieved_chunks]
        elif mode == "ontology_graph_only":
            a, tr = system.answer_ontology_graph_only(q)
            # Graph hits only — no text track in this ablation.
            # Score against exactly what the model saw (label + definition + facts +
            # related nodes), not a label/definition-only summary of it.
            ctx = [h.get("frame") or f"{h['label']}: {h.get('definition', '')}".strip() for h in tr.graph_hits]
            if tr.answer_for_eval:
                a = tr.answer_for_eval
        elif mode == "ontology_contextual_rag":
            a, tr = system.answer_ontology_contextual_rag(q)
            # Combine graph and vector contexts as "contexts" for RAGAS.
            graph_ctx = []
            for h in tr.graph_hits:
                graph_ctx.append(h.get("frame") or f"{h['label']}: {h.get('definition', '')}".strip())
            vec_ctx = [c["text"] for c in tr.retrieved_chunks]
            ctx = graph_ctx + vec_ctx
            # Score only the Final Answer section — the derivation/citation
            # text (node IDs, "paraphrased as", quote framing) isn't a
            # factual claim about the handbook and shouldn't be judged as
            # one by faithfulness. See AnswerTrace.answer_for_eval.
            if tr.answer_for_eval:
                a = tr.answer_for_eval
        else:
            raise ValueError("Unknown mode")

        questions.append(q)
        answers.append(a)
        contexts.append(ctx)
        ground_truths.append(gt)

    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    # Note: context_recall needs ground_truth; without it, it's less meaningful.
    evaluate_kwargs: Dict[str, Any] = {
        "metrics": [faithfulness, answer_relevancy, context_precision, context_recall],
    }
    if ragas_llm is not None:
        evaluate_kwargs["llm"] = ragas_llm
    if ragas_embeddings is not None:
        evaluate_kwargs["embeddings"] = ragas_embeddings

    result = evaluate(ds, **evaluate_kwargs)

    return {
        "mode": mode,
        "ragas": result.to_pandas().describe().to_dict(),
        "per_item": result.to_pandas().to_dict(orient="records"),
    }


# =========================
# Interactive demo (no RAGAS here anymore)
# =========================
def interactive_loop(system: RAGSystem):
    print("\n=== Interactive demo ===")
    print("Type a question (or 'exit'):")
    while True:
        q = input("\n> ").strip()
        if q.lower() in ("exit", "quit"):
            break

        for mode in ALL_MODES:
            print("\n" + "=" * 90)
            print(f"MODE: {mode}")
            if mode == "bare_llm":
                a, tr = system.answer_bare_llm(q)
            elif mode == "basic_rag":
                a, tr = system.answer_basic_rag(q)
            elif mode == "hybrid_rag":
                a, tr = system.answer_hybrid_rag(q)
            elif mode == "ontology_graph_only":
                a, tr = system.answer_ontology_graph_only(q)
            else:
                a, tr = system.answer_ontology_contextual_rag(q)

            print("\nANSWER:\n", a)
            # Trace (thesis requirement: show algorithm steps)
            print("\nTRACE (retrieval + ontology hits):")
            print(json.dumps(asdict(tr), indent=2, ensure_ascii=False)[:3000])

    print("\nExited. (RAGAS no longer runs here — use: python app.py --eval)")


# =========================
# Evaluation entry point (separate command)
# =========================
def run_evaluation(system: RAGSystem, modes: List[str], embedder: SentenceTransformer):
    eval_rows = load_eval_set(EVAL_SET)
    if not eval_rows:
        return

    if not RAGAS_AVAILABLE:
        print(
            "\n[WARN] Skipping RAGAS evaluation: ragas failed to import "
            f"({RAGAS_IMPORT_ERROR}). See README for the fix "
            "(pin ragas==0.3.9, or pip install langchain-google-vertexai)."
        )
        return

    print("\n=== Running RAGAS evaluation (standalone mode) ===")
    print("Configuring ragas to use Claude + local embeddings (not OpenAI)...")
    ragas_llm, ragas_embeddings = build_ragas_llm_and_embeddings(embedder)
    if ragas_llm is None:
        print(
            "[WARN] Aborting: no judge LLM available for ragas "
            "(install langchain-anthropic — see warning above)."
        )
        return
    if ragas_embeddings is None:
        print(
            "[WARN] Could not wrap local embeddings for ragas; "
            "answer_relevancy may fail or fall back to OpenAI."
        )

    out_path = "ragas_results.json"
    all_results = []
    for mode in modes:
        print(f"\nEvaluating mode: {mode}")
        res = run_ragas_on_mode(
            system, eval_rows, mode,
            ragas_llm=ragas_llm, ragas_embeddings=ragas_embeddings,
        )
        all_results.append(res)
        # Save after EACH mode so a crash/cancel doesn't lose everything
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"  -> partial results saved to {out_path}")

    print(f"\nSaved RAGAS results to {out_path}")


# =========================
# Retrieval diagnosis (no LLM calls, no API cost)
# =========================
def _ref_coverage(ref_tokens: set, text: str) -> float:
    if not ref_tokens:
        return 0.0
    toks = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(ref_tokens & toks) / len(ref_tokens)


def run_diagnosis(system: RAGSystem, top_n: int = 5):
    """
    For every eval question, print the top graph nodes and top text chunks
    with their scores, plus how much of the ground-truth answer's wording each
    candidate contains ("ref-cov"). Use it to see, per question, whether the
    right evidence ranks first, where junk sits relative to graph_min_score /
    vec_min_score, and which threshold would separate them — instead of tuning
    thresholds blind. Makes no LLM calls.
    """
    rows = load_eval_set(EVAL_SET)
    for r in rows:
        q, gt = r["question"], r.get("ground_truth", "")
        ref = {t for t in re.findall(r"[a-z0-9]+", gt.lower()) if len(t) >= 3}
        print("\n" + "=" * 100)
        print("Q:", q)
        print("REF:", gt[:140])
        if system.ontology:
            print("  -- graph candidates (score | ref-cov | label / frame preview)")
            hits = system.ontology.search_nodes_by_text(q, limit=top_n, min_score=0.0, score_margin=10.0)
            for h in hits:
                prev = (h.label or h.node) + " | " + h.frame.replace("\n", " ")[:90]
                print(f"  {h.score:.3f} | {_ref_coverage(ref, h.frame):.2f} | {prev}")
        if system.vector_index:
            print("  -- text candidates after lexical rescoring (score | ref-cov | preview)")
            cands = system.vector_index.search(q, k=max(9, top_n), min_score=0.0)
            cands = VectorIndex.rescore_lexical(q, cands)[:top_n]
            for c in cands:
                print(f"  {c.score:.3f} | {_ref_coverage(ref, c.text):.2f} | {c.text.replace(chr(10), ' ')[:90]}")


# =========================
# Main
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="LSPU Ontology Contextual RAG")
    p.add_argument(
        "--eval",
        dest="run_eval",
        action="store_true",
        help="Run RAGAS evaluation only, then exit (kept separate from the interactive demo).",
    )
    p.add_argument(
        "--modes",
        type=str,
        default=None,
        help="With --eval: comma-separated subset, e.g. --modes ontology_graph_only,ontology_contextual_rag",
    )
    p.add_argument(
        "--diagnose",
        action="store_true",
        help="Print top graph/text candidates with scores for each eval question (no LLM calls), then exit.",
    )
    p.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force re-embedding even if a valid cache exists (use after editing the handbook).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading models...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    llm = ClaudeClient()

    print("Loading ontology...")
    # Pass the embedder in so node search ranks by semantic similarity
    # instead of lexical substring/token matching (see search_nodes_by_text).
    ontology = OntologyStore(PILLAR3_TTL, embedder=embedder)
    ontology.load()

    print("Preparing vector index (uses cache when possible)...")
    vector_index = build_vector_index_from_text_file(
        embedder, PILLAR3_TXT, rebuild_cache=args.rebuild_cache
    )

    system = RAGSystem(vector_index=vector_index, ontology=ontology, llm=llm)

    if args.diagnose:
        run_diagnosis(system)
        return

    # Eval and interactive are mutually exclusive — pick one per run.
    if args.run_eval:
        modes = [m.strip() for m in args.modes.split(",")] if args.modes else ALL_MODES
        invalid = [m for m in modes if m not in ALL_MODES]
        if invalid:
            raise SystemExit(f"Unknown mode(s): {invalid}. Valid modes: {ALL_MODES}")
        run_evaluation(system, modes, embedder)
        return

    interactive_loop(system)


if __name__ == "__main__":
    main()