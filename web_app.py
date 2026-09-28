"""
web_app.py — thin Flask layer around app.py for the retrieval-visualizer demo.

This does NOT reimplement your RAG logic. It calls the exact same objects
app.py builds (OntologyStore with the shared embedder, VectorIndex,
RAGSystem, ClaudeClient) and narrates each step they take, so the frontend
can animate it slowly and clearly for a live demo. The retrieval pipeline
shown here (pool sizes, thresholds, rescoring, dedup) mirrors
RAGSystem.answer_ontology_contextual_rag / answer_hybrid_rag exactly, so
what's on screen is never different from what actually generated the
answer.

Run:
    python web_app.py
Then open:
    http://127.0.0.1:5000

Works even without an ANTHROPIC_API_KEY in .env — it'll run retrieval-only
(no generated final answers) so you can still demo the retrieval mechanics.
Add a real key to get real Claude-generated answers + derivation traces.
"""
import os
import re
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, url_for
from dotenv import load_dotenv

load_dotenv()

from sentence_transformers import SentenceTransformer

from app import (
    ALL_MODES,
    PILLAR3_TTL,
    PILLAR3_TXT,
    EMBED_MODEL_NAME,
    ClaudeClient,
    OntologyStore,
    RAGSystem,
    VectorIndex,
    build_vector_index_from_text_file,
)

from catalog import COLLEGES

app = Flask(__name__)


@app.context_processor
def inject_catalog():
    """Make the college/program catalog available to every template."""
    return {"colleges": COLLEGES}

# ---------------------------------------------------------------------
# The homepage template references plain "assets/images/..." paths (as
# provided in the design mockup) rather than Flask's /static/ convention.
# Flask does NOT serve arbitrary folders automatically — only whatever is
# configured as static_folder — so without this route every image 404s.
# Put your logo/photos in an "assets/images/" folder next to this file.
# ---------------------------------------------------------------------
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(ASSETS_DIR, filename)


# ---------------------------------------------------------------------
# These must mirror the defaults in RAGSystem.answer_ontology_contextual_rag
# (app.py) exactly — if you tune them there, update them here too, or the
# demo will show a different pipeline than the one that actually runs.
# ---------------------------------------------------------------------
K_VEC = 2
K_GRAPH = 2
VEC_MIN_SCORE = 0.25
GRAPH_MIN_SCORE = 0.40
VEC_CANDIDATE_POOL = 9

# ---------------------------------------------------------------------
# One-time setup at process start (same models/data app.py uses)
# ---------------------------------------------------------------------
print("[web_app] Loading embedding model...")
_embedder = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

print("[web_app] Loading ontology graph (semantic node search via shared embedder)...")
_ontology = OntologyStore(PILLAR3_TTL, embedder=_embedder)
_ontology_loaded = False
try:
    _ontology.load()
    _ontology_loaded = True
    print(f"[web_app] Ontology loaded from {PILLAR3_TTL}")
except Exception as e:
    print(f"[web_app][WARN] Could not load ontology TTL ({e}). Ontology mode will show no graph hits.")

print("[web_app] Preparing vector index (uses cache when available)...")
_vector_index = build_vector_index_from_text_file(_embedder, PILLAR3_TXT)

_llm = None
try:
    _llm = ClaudeClient()
    print("[web_app] ANTHROPIC_API_KEY found — generated answers are ON.")
except Exception as e:
    print(f"[web_app][INFO] No ANTHROPIC_API_KEY ({e}). Retrieval-only demo (answers OFF).")

_system = RAGSystem(vector_index=_vector_index, ontology=_ontology, llm=_llm)


# ---------------------------------------------------------------------
# Helpers: dataclass -> plain dict for JSON
# ---------------------------------------------------------------------
def _chunk_dict(c) -> Dict[str, Any]:
    return {
        "chunk_id": c.chunk_id,
        "source": c.source,
        "text": c.text,
        "score": round(float(c.score), 4),
    }


def _graph_dict(h) -> Dict[str, Any]:
    return {
        "node": h.node,
        "label": h.label,
        "definition": h.definition,
        "source_article": h.source_article,
        "score": round(float(h.score), 4),
        "relations": [{"predicate": p, "object": o} for p, o in h.relations],
    }


MODE_META = {
    "bare_llm": {
        "title": "Bare LLM",
        "subtitle": "No retrieval — Claude answers from parametric knowledge alone",
    },
    "basic_rag": {
        "title": "Basic RAG",
        "subtitle": "Dense vector retrieval over raw handbook text",
    },
    "hybrid_rag": {
        "title": "Hybrid RAG",
        "subtitle": "Dense retrieval + lexical overlap rescoring",
    },
    "ontology_contextual_rag": {
        "title": "Ontology Contextual RAG",
        "subtitle": "Semantic ontology frame + text, with a required derivation trace",
        "proposed": True,
    },
}


# ---------------------------------------------------------------------
# Step-building helpers
#
# Every step dict carries:
#   type        - what the frontend should render (see main.js renderStep)
#   stage       - short name shown on the pipeline strip; consecutive
#                 steps that share a stage are grouped into ONE chip
#                 (e.g. "searching..." followed by "here are the hits"
#                 both belong to the "Ontology Graph Search" stage)
#   mechanism   - "none" | "lexical" | "vector" | "graph_vector" | "merge"
#                 | "generate" -> drives the colored badge on the
#                 frontend, so it's immediately visible WHICH retrieval
#                 technique fired. "vector" = embedding search over raw
#                 text chunks; "graph_vector" = embedding search over
#                 ontology node label+definition — same underlying
#                 technique, different target, kept visually distinct.
#   narration   - one or two plain-English sentences explaining what is
#                 happening right now, shown big above the raw data
# ---------------------------------------------------------------------
def _query_step(question: str) -> Dict[str, Any]:
    return {
        "type": "query",
        "stage": "Query",
        "mechanism": "none",
        "text": question,
        "narration": "The question enters the pipeline.",
    }


def _searching_step(stage: str, mechanism: str, narration: str) -> Dict[str, Any]:
    """A short 'working...' beat shown BEFORE the results of a search,
    so the audience sees that a distinct retrieval mechanism just fired
    before the results land."""
    return {
        "type": "search_start",
        "stage": stage,
        "mechanism": mechanism,
        "narration": narration,
    }


def build_trace(question: str, mode: str) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = [_query_step(question)]

    if mode == "bare_llm":
        steps.append({
            "type": "note",
            "stage": "No Retrieval",
            "mechanism": "none",
            "narration": "This baseline skips retrieval entirely — the question goes straight to Claude, answered only from what it already knows, with no handbook text or ontology context supplied.",
        })

    elif mode == "basic_rag":
        steps.append(_searching_step(
            "Vector Search", "vector",
            "Encoding the question and comparing it against every embedded chunk of raw handbook text using cosine similarity.",
        ))
        chunks = _vector_index.search(question, k=5) if _vector_index else []
        steps.append({
            "type": "vector_search",
            "stage": "Vector Search",
            "mechanism": "vector",
            "label": "Dense vector search — top 5 by cosine similarity",
            "narration": "These are the five closest text chunks by embedding similarity.",
            "chunks": [_chunk_dict(c) for c in chunks],
        })

    elif mode == "hybrid_rag":
        pool = max(12, 5 * 3)  # mirrors answer_hybrid_rag's max(12, k*3) with k=5
        steps.append(_searching_step(
            "Vector Search", "vector",
            f"Stage 1 — encoding the question and pulling the {pool} closest chunks by embedding similarity (a wider net than Basic RAG).",
        ))
        raw = _vector_index.search(question, k=pool) if _vector_index else []
        steps.append({
            "type": "vector_search",
            "stage": "Vector Search",
            "mechanism": "vector",
            "label": f"Stage 1 — Dense vector search (top {pool} candidates)",
            "narration": f"{pool} candidates retrieved by embedding similarity alone, before any lexical rescoring.",
            "chunks": [_chunk_dict(c) for c in raw],
        })

        steps.append(_searching_step(
            "Lexical Rescore", "lexical",
            "Stage 2 — counting how many question words each candidate literally shares with it (lexical overlap), then blending that with the vector score.",
        ))
        q_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
        rescored = []
        for c in raw:
            t_tokens = set(re.findall(r"[a-z0-9]+", c.text.lower()))
            lexical = len(q_tokens.intersection(t_tokens)) / (len(q_tokens) + 1e-6)
            combined = 0.75 * c.score + 0.25 * lexical
            rescored.append((combined, lexical, c))
        rescored.sort(key=lambda x: x[0], reverse=True)
        top = rescored[:5]
        steps.append({
            "type": "lexical_rescore",
            "stage": "Lexical Rescore",
            "mechanism": "lexical",
            "label": "0.75 \u00d7 vector score + 0.25 \u00d7 lexical overlap \u2192 top 5 kept",
            "narration": "The candidates are reordered by the blended score. Notice the ranking can change from Stage 1.",
            "chunks": [
                {
                    **_chunk_dict(c),
                    "combined_score": round(combined, 4),
                    "lexical_overlap": round(lexical, 4),
                }
                for combined, lexical, c in top
            ],
        })

    elif mode == "ontology_contextual_rag":
        # --- ontology graph side: semantic (embedding) search over nodes ---
        steps.append(_searching_step(
            "Ontology Graph Search", "graph_vector",
            f"Embedding the question and comparing it against every ontology node's label + definition using cosine similarity — keeping matches \u2265 {GRAPH_MIN_SCORE:.2f}, top {K_GRAPH}.",
        ))
        graph_hits_raw = (
            _ontology.search_nodes_by_text(question, limit=K_GRAPH, min_score=GRAPH_MIN_SCORE)
            if _ontology_loaded else []
        )
        graph_hits = OntologyStore.dedupe_hits(graph_hits_raw)
        steps.append({
            "type": "graph_search",
            "stage": "Ontology Graph Search",
            "mechanism": "graph_vector",
            "label": f"Ontology node search \u2014 semantic embedding match (top {K_GRAPH}, min score {GRAPH_MIN_SCORE:.2f})",
            "narration": "These ontology nodes matched, each scored by cosine similarity to the question, with near-duplicate nodes merged away.",
            "nodes": [_graph_dict(h) for h in graph_hits],
        })

        # --- supplementary text side: wide embedding pool ---
        steps.append(_searching_step(
            "Supplementary Text Search", "vector",
            f"In parallel, embedding the question against raw handbook chunks and pulling up to {VEC_CANDIDATE_POOL} candidates \u2265 {VEC_MIN_SCORE:.2f} similarity — a wide net, narrowed in the next step.",
        ))
        vec_candidates_raw = (
            _vector_index.search(question, k=VEC_CANDIDATE_POOL, min_score=VEC_MIN_SCORE)
            if _vector_index else []
        )
        steps.append({
            "type": "vector_search",
            "stage": "Supplementary Text Search",
            "mechanism": "vector",
            "label": f"Dense vector search over raw text \u2014 up to {VEC_CANDIDATE_POOL} candidates",
            "narration": "Raw-text candidates retrieved purely by embedding similarity, before lexical rescoring.",
            "chunks": [_chunk_dict(c) for c in vec_candidates_raw],
        })

        # --- rescore + dedupe down to the final k_vec chunks ---
        steps.append(_searching_step(
            "Rescore & Dedupe", "lexical",
            "Blending each candidate's vector score with lexical word-overlap (the same trick used in Hybrid RAG), then dropping near-duplicate chunks.",
        ))
        vec_rescored = VectorIndex.rescore_lexical(question, vec_candidates_raw)
        vec_chunks = VectorIndex.dedupe_chunks(vec_rescored)[:K_VEC]
        steps.append({
            "type": "vector_search",
            "stage": "Rescore & Dedupe",
            "mechanism": "lexical",
            "label": f"Blended score, deduped, top {K_VEC} kept",
            "narration": "This is the final text evidence that gets merged into the contextual frame.",
            "chunks": [_chunk_dict(c) for c in vec_chunks],
        })

        steps.append({
            "type": "frame_assembly",
            "stage": "Assemble Contextual Frame",
            "mechanism": "merge",
            "label": "Contextual knowledge frame",
            "narration": "The matched ontology nodes (definitions, source articles, relations) are merged with the retrieved text chunks into ONE structured context block, sent to Claude with strict grounding instructions.",
            "text": (
                "Ontology node definitions + relations are merged with the retrieved "
                "text chunks into a single structured context block. The model is "
                "instructed to answer only what's explicitly supported, and to output "
                "a separate, verbatim-marked derivation section naming exactly which "
                "nodes and text snippets it used."
            ),
        })

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Final answer (only if an API key is configured)
    answer_text: Optional[str] = None
    derivation_text: Optional[str] = None
    llm_error: Optional[str] = None
    if _llm is not None:
        steps.append({
            "type": "llm_call",
            "stage": "Generate Answer",
            "mechanism": "generate",
            "narration": (
                "Claude receives just the question — no retrieved context — and "
                "answers from its own parametric knowledge alone."
                if mode == "bare_llm" else
                "Claude receives the assembled context plus the question, and must answer using only that context."
            ),
        })
        try:
            if mode == "bare_llm":
                raw_answer, _tr = _system.answer_bare_llm(question)
                answer_text = raw_answer
            elif mode == "basic_rag":
                raw_answer, _tr = _system.answer_basic_rag(question)
                answer_text = raw_answer
            elif mode == "hybrid_rag":
                raw_answer, _tr = _system.answer_hybrid_rag(question)
                answer_text = raw_answer
            else:
                raw_answer, _tr = _system.answer_ontology_contextual_rag(question)
                # Ontology mode's raw answer is marker-delimited
                # (===FINAL_ANSWER=== ... ===DERIVATION=== ...) — split it
                # so the UI can show the clean answer up top and the
                # required derivation trace (which ontology nodes / text
                # quotes were used) as its own, clearly-labeled section.
                final_only, full_display = _system._split_final_answer(raw_answer)
                if final_only != full_display:
                    answer_text = final_only
                    marker = RAGSystem._DERIVATION_MARKER
                    if marker in full_display:
                        derivation_text = full_display.split(marker, 1)[1].strip()
                else:
                    answer_text = full_display
        except Exception as e:  # keep the demo alive even if the API call fails
            llm_error = str(e)

    steps.append({
        "type": "answer",
        "stage": "Final Answer",
        "mechanism": "generate",
        "text": answer_text,
        "derivation": derivation_text,
        "error": llm_error,
    })

    return {
        "mode": mode,
        "meta": MODE_META[mode],
        "steps": steps,
    }


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.route("/")
def index():
    """LSPU-LB student portal homepage — hosts the 'LSPU Knowledge
    Assistant' chat, which talks to /api/chat only (Ontology Contextual
    RAG, no retrieval internals exposed here)."""
    return render_template(
        "index.html",
        llm_enabled=_llm is not None,
    )



# ---------------------------------------------------------------------
# Multi-page site (Explore / Recommendation / About / Contact / Assistant)
# ---------------------------------------------------------------------
@app.route("/explore")
def explore():
    """Colleges and programs, with pictures and details per program."""
    return render_template("explore.html")


@app.route("/recommendation")
def recommendation():
    """Interest checklist -> suggested colleges/programs (client-side)."""
    return render_template("recommendation.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/assistant")
def assistant():
    """Full-page Knowledge Assistant. Talks to the same /api/chat endpoint
    as the homepage chat preview (Ontology Contextual RAG only)."""
    return render_template("assistant.html", llm_enabled=_llm is not None)


@app.route("/programs")
def programs_legacy():
    """Old URL: programs now live on the Explore page."""
    return redirect(url_for("explore") + "#program-details", code=301)


@app.route("/visualizer")
def visualizer():
    """The full multi-architecture retrieval visualizer (Basic RAG /
    Hybrid RAG / Ontology Contextual RAG side-by-side), linked from the
    homepage as its own page rather than a section of the chat."""
    return render_template(
        "visualizer.html",
        llm_enabled=_llm is not None,
        ontology_loaded=_ontology_loaded,
        vector_loaded=_vector_index is not None,
        modes=ALL_MODES,
        mode_meta=MODE_META,
    )


@app.route("/api/status")
def api_status():
    return jsonify({
        "llm_enabled": _llm is not None,
        "ontology_loaded": _ontology_loaded,
        "vector_loaded": _vector_index is not None,
        "modes": ALL_MODES,
        "mode_meta": MODE_META,
    })


@app.route("/api/query", methods=["POST"])
def api_query():
    """Used by the /visualizer page only — runs (and animates) every
    selected architecture's retrieval trace, chunks/graph nodes and all."""
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    modes = data.get("modes") or ALL_MODES
    modes = [m for m in modes if m in ALL_MODES]

    if not question:
        return jsonify({"error": "Question is required."}), 400
    if not modes:
        return jsonify({"error": "No valid modes given."}), 400

    results = [build_trace(question, m) for m in modes]
    return jsonify({"question": question, "results": results})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Used by the homepage 'LSPU Knowledge Assistant' chat ONLY.

    Always runs the Ontology Contextual RAG pipeline (never Basic/Hybrid
    RAG) and returns just the final, student-facing answer — no chunks,
    no vector scores, no ontology nodes, and no derivation trace. That
    retrieval breakdown is intentionally kept out of this tab; it's only
    ever shown on the separate /visualizer page.
    """
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Please type a question."}), 400

    if _llm is None:
        return jsonify({
            "answer": (
                "The Knowledge Assistant isn't fully set up yet — it needs an "
                "ANTHROPIC_API_KEY configured on the server to generate answers. "
                "Please let the site administrator know."
            ),
            "configured": False,
        })

    try:
        raw_answer, _trace = _system.answer_ontology_contextual_rag(question)
        # answer_ontology_contextual_rag's raw text is marker-delimited
        # (===FINAL_ANSWER=== ... ===DERIVATION=== ...); the chat only
        # ever shows the clean final answer, never the derivation section.
        final_answer, _full_display = _system._split_final_answer(raw_answer)
        return jsonify({"answer": final_answer, "configured": True})
    except Exception as e:
        return jsonify({
            "error": f"Something went wrong generating an answer ({e}). Please try again.",
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[web_app] Open http://127.0.0.1:{port} in your browser.\n")
    app.run(debug=True, port=port)