# LSPU Ontology Contextual RAG — Thesis Prototype

Compares four answering modes on your handbook ontology:

- **Bare LLM** — no retrieval
- **Basic RAG** — dense vector retrieval over raw handbook text
- **Hybrid RAG** — dense + lexical rescoring
- **Ontology Contextual RAG** — retrieves relevant ontology (TTL) nodes as a
  "contextual knowledge frame" (definitions, source articles, relations),
  ranked by semantic similarity and filtered by a relevance threshold,
  optionally combined with retrieved text chunks (also threshold-filtered
  and deduped), and forces the model to output both a final answer and a
  step-by-step explanation of how it was derived (which ontology nodes /
  text snippets were used).

Also includes a RAGAS evaluation harness (faithfulness, answer relevancy,
context precision, context recall) across all four modes, run as a separate
command from the interactive demo (see below).

## 1. Files you need to add

Drop these in before running (not included — they're yours):

- `data/pillar3_handbook_ontology.ttl` — your Pillar 3 ontology in Turtle
  format. Expected properties (adjust in `app.py` if yours differ):
  `rdfs:label`, `lspu:definition`, `lspu:sourceArticle`.
- `data/pillar3_handbook_source.txt` — (optional) raw handbook text, used
  for the vector-based baselines (Basic RAG / Hybrid RAG / the text side of
  Ontology Contextual RAG). If you skip this, those modes still run but with
  no retrieved chunks.
- `eval/eval_set.jsonl` — a small QA set for RAGAS scoring. A 3-item example
  is included; expand it to 10–50 items covering your handbook.

Later, when you add Pillar 1 / Pillar 2, drop in more `.txt` / `.ttl` files
and either point the loader at them or ask for the multi-corpus router
version of `app.py` (the single-pillar version here keeps things simple for
now).

## 2. Setup

```bash
cd lspu-rag-thesis
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # then paste your real Anthropic API key into .env
```

## 3. Run

The interactive demo and the RAGAS evaluation are now separate commands, so
a heavy evaluation run never freezes your interactive session (and vice
versa).

```bash
python app.py                                          # interactive demo only (light)
python app.py --rebuild-cache                           # force re-embed after editing the handbook
python app.py --eval                                     # RAGAS, all 4 modes, then exits
python app.py --eval --modes ontology_contextual_rag     # evaluate just one mode
```

- **`python app.py`** — opens an interactive prompt: type a question, and it
  prints the answer plus retrieval/graph trace under all four modes. Type
  `exit` to quit. This no longer runs RAGAS on exit.
- **`--eval`** — runs the RAGAS evaluation over `eval/eval_set.jsonl` and
  writes `ragas_results.json`, then exits. Use `--modes` with a
  comma-separated list (`bare_llm`, `basic_rag`, `hybrid_rag`,
  `ontology_contextual_rag`) to evaluate a subset instead of all four.
  Results are saved after each mode finishes, so a crash or cancel partway
  through doesn't lose earlier results.
- **`--rebuild-cache`** — forces re-embedding of the handbook text even if a
  valid cache exists. Use this after editing
  `data/pillar3_handbook_source.txt`.

### Embedding cache

The first run encodes the handbook text with a visible progress bar and
saves the embeddings to `cache/pillar3_embeddings.npy` (plus a small
metadata file). Every run after that loads the cache instantly instead of
re-embedding. The cache is fingerprinted on the embedding model name and the
exact chunk ids/text, so it auto-invalidates the moment you edit the
handbook — no manual cleanup needed unless you want to force it early with
`--rebuild-cache`.

Ontology node embeddings (label + definition per TTL node) are recomputed
in memory on every run — the ontology is small enough that this is cheap
and doesn't need its own on-disk cache.

If your machine is still tight on RAM during that first encode, lower
`EMBED_BATCH_SIZE` in `app.py` (default `16`; try `8`).

## 4. Notes

- Model used: `claude-haiku-4-5-20251001`. Swap `MODEL_NAME` in `app.py` if
  you want a different Claude model.
- Ontology node lookup in `OntologyStore.search_nodes_by_text` ranks nodes
  by **semantic similarity** (cosine similarity between the query and each
  node's `rdfs:label` + `lspu:definition`, using the same sentence-transformer
  embedder as the vector index), filtered by a minimum score threshold
  rather than always returning a fixed count. This is what keeps
  `context_precision` from being dragged down by loosely-related nodes.
  A lexical substring/token-overlap fallback is still used automatically if
  no embedder is supplied to `OntologyStore`.
- `answer_ontology_contextual_rag` exposes four tunables worth knowing
  about if precision/recall shift as your ontology or eval set grows:
  - `k_graph` / `k_vec` — max number of ontology nodes / text chunks to
    consider (default 3 / 3).
  - `graph_min_score` / `vec_min_score` — minimum relevance score required
    to keep a candidate at all (default 0.35 / 0.30). Raise these if you
    still see irrelevant context slipping in; lower them if `context_recall`
    starts dropping (i.e., real answers are being filtered out).
  - Retrieved text chunks are also deduped (`VectorIndex.dedupe_chunks`) to
    drop near-identical chunks caused by chunk overlap.
- If your TTL uses different predicate names than `lspu:definition` /
  `lspu:sourceArticle`, update the `LSPU` namespace and the predicates
  referenced in `OntologyStore`.
- RAGAS calls the LLM many times per metric, so `--eval` is still slow by
  nature — it's just isolated from the interactive demo now, and you can run
  it overnight or scope it down with `--modes`.
