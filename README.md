# LSPU Ontology Contextual RAG — Thesis Prototype

Compares five answering modes on your LSPU ontology and handbook text:

- **Bare LLM** — no retrieval
- **Basic RAG** — dense vector retrieval over raw handbook text
- **Hybrid RAG** — dense + lexical rescoring
- **Ontology Graph Only** — an ablation: retrieves relevant ontology (TTL)
  nodes only and never sees the raw handbook text
- **Ontology Contextual RAG** — retrieves relevant ontology nodes as a
  "contextual knowledge frame" (definitions, source citations, facts,
  related nodes, parent roll-ups), ranked by semantic similarity and
  filtered by a relevance threshold, combined with retrieved text chunks
  (also threshold-filtered, lexically rescored and deduped). The model is
  forced to output both a final answer and a step-by-step derivation of
  which ontology nodes / text snippets it used.

Also includes a RAGAS evaluation harness (faithfulness, answer relevancy,
context precision, context recall) across the modes, a no-answer-generation
`--diagnose` tool for tuning retrieval thresholds, and Tagalog/Taglish
support (questions are translated to English for retrieval; answers are
always in English). Evaluation is a separate command from the interactive
demo (see below).

## 1. Files you need to add

Drop these in before running (not included — they're yours):

- `data/LSPU_ONTOLOGY.ttl` — your merged ontology in Turtle format. It can
  carry three pillar namespaces, which the code reads automatically:
  - `lspu:` — handbook / conduct policy (Pillar 3): `lspu:definition`,
    `lspu:sourceArticle`
  - `lspuprog:` — program information (Pillar 1): `lspuprog:programOverview`,
    `lspuprog:sourceDocument`
  - `lspuadm:` — admission & Citizen's Charter procedures (Pillar 2):
    `lspuadm:sourceSection`, `lspuadm:sourceDocument`

  Every node's `rdfs:label` is also used. If `rdfs:comment` appears on a
  non-class, non-property node it is used as a fallback definition.

- `data/LSPU_SOURCE.txt` — (optional) raw handbook text, used for the
  vector-based modes (Basic RAG / Hybrid RAG / the text side of Ontology
  Contextual RAG). If you skip this, those modes still run but with no
  retrieved chunks.
- `eval/eval_set.jsonl` — a QA set for RAGAS scoring, one JSON object per
  line: `{"question": "...", "ground_truth": "..."}`. Aim for 10–50+ items
  covering the handbook. `ground_truth` is required for `context_recall`
  to be meaningful.

If your filenames or namespaces differ, adjust `PILLAR3_TTL`, `PILLAR3_TXT`
and `EVAL_SET` in the Config block at the top of `app.py`, and the
namespaces / predicate lists in `OntologyStore`.

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

Evaluation additionally needs `ragas`, `datasets` and `langchain-anthropic`
(ragas uses Claude via LangChain as its judge, so no OpenAI key is needed).
Some ragas / langchain versions conflict on import; if `--eval` reports that
ragas failed to import, try pinning `ragas==0.3.9` or installing
`langchain-google-vertexai`. The interactive demo works without ragas.

## 3. Run

The interactive demo, the RAGAS evaluation and the diagnosis tool are
separate commands, so a heavy evaluation run never freezes your interactive
session (and vice versa). Only one runs per invocation.

```bash
python app.py                                                   # interactive demo (light)
python app.py --rebuild-cache                                   # force re-embed after editing the handbook
python app.py --eval                                            # RAGAS, all 5 modes, then exits
python app.py --eval --modes bare_llm
python app.py --eval --modes basic_rag
python app.py --eval --modes hybrid_rag
python app.py --eval --modes ontology_graph_only
python app.py --eval --modes ontology_contextual_rag

python app.py --diagnose                                        # inspect retrieval scores per eval question
```

- **`python app.py`** — opens an interactive prompt. Type a question in
  English, Tagalog or Taglish; it prints the answer plus a retrieval/graph
  trace (truncated to the first 3000 characters) under all five modes. If
  the question was translated for retrieval, the translated query is shown.
  Type `exit` to quit.
- **`--eval`** — runs RAGAS over `eval/eval_set.jsonl` and writes
  `ragas_results.json`, then exits. Use `--modes` with a comma-separated
  list from `bare_llm`, `basic_rag`, `hybrid_rag`, `ontology_graph_only`,
  `ontology_contextual_rag` to evaluate a subset.
- **`--diagnose`** — for every eval question, prints the top graph nodes and
  top text chunks with their scores and a "ref-cov" value (how much of the
  ground-truth wording each candidate contains). Use it to see where good
  and junk evidence sit relative to `graph_min_score` / `vec_min_score`
  before changing thresholds. It makes one cheap Claude call per question
  (the query translation) and none for answer generation.
- **`--rebuild-cache`** — forces re-embedding of the handbook text even if a
  valid cache exists. Use this after editing `data/LSPU_SOURCE.txt`.

### Evaluation runs: checkpointing and retries

Long `--eval` runs (e.g. 60 questions × 5 modes) are built to survive
failures:

- **Per-question checkpoints.** Each generated answer is appended to
  `ragas_checkpoints/<mode>.gen.jsonl` immediately, and RAGAS scores are
  appended after each batch of 5 rows (`<mode>.scores.jsonl`; change
  `RAGAS_SCORE_BATCH_SIZE` to adjust). If a run crashes or is cancelled,
  re-run the same `--eval` command and it resumes from the checkpoints.
  A saved answer is reused only if it's for the same question at the same
  position. A mode's checkpoint files are deleted once that mode finishes,
  so a normal fresh run never reuses old data.
- **If you change code or the eval set after a crash**, delete the
  `ragas_checkpoints/` folder first, otherwise answers generated by the old
  code would be reused.
- **Per-mode results.** `ragas_results.json` is rewritten after each mode
  finishes, so completed modes are never lost. It holds, per mode, summary
  statistics and per-item scores.
- **Retry with backoff.** `ClaudeClient.generate` retries rate limits,
  connection/timeout errors and 5xx errors up to 6 attempts (2s, 4s, 8s…,
  capped at 60s, with jitter). Other errors (bad request, auth) raise
  immediately.

### Embedding cache

The first run encodes the handbook text with a visible progress bar and
saves the embeddings to `cache/lspu_embeddings.npy` (plus
`cache/lspu_embeddings_meta.json`). Every run after that loads the cache
instantly. The cache is fingerprinted on the embedding model name and the
exact chunk ids/text, so it auto-invalidates when you edit the handbook —
no manual cleanup needed unless you want to force it with `--rebuild-cache`.

Ontology node embeddings are recomputed in memory on every run — the
ontology is small enough that this is cheap.

If your machine is tight on RAM during that first encode, lower
`EMBED_BATCH_SIZE` in `app.py` (default `16`; try `8`).

## 4. How it works

**Chunking.** The handbook text is split on `Chapter N` / `Article N.`
headers first, then into ~1200-character windows with 180 characters of
overlap _within_ each section, so a chunk never straddles two unrelated
articles.

**Multilingual queries.** Both retrieval tracks use the English-trained
`all-MiniLM-L6-v2` embedder, which handles Tagalog/Taglish poorly. So every
retrieval mode first translates the question into a search-friendly English
query with a small Claude call (English input is returned unchanged; if the
call fails it falls back to the raw query). Claude still receives the
student's _original_ question for the final answer, and all answers are
instructed to be in English.

**Ontology retrieval.** `OntologyStore.search_nodes_by_text` ranks nodes by
cosine similarity between the query and each node's text (label +
definition + facts + labeled links). Results must clear both:

- an absolute floor (`min_score`), and
- a relative margin (`score_margin`, default 0.08) — a lower-ranked hit is
  kept only if it's within that margin of the best hit. This stops the many
  near-identical Sanction/SanctionTier nodes from filling slots with loosely
  related matches.

Schema/vocabulary nodes (`owl:Class` / `rdfs:Class`) and property
declarations are never indexed, since they document terms rather than state
facts. Each hit becomes a "frame" containing its definition, source
citation, own facts, related nodes, and (for Step/Tier-level nodes) the
facts of up to two parent nodes so roll-up totals aren't lost. Near-duplicate
hits are removed (`OntologyStore.dedupe_hits`). If no embedder is supplied,
a lexical fallback is used.

**Answer format.** In the ontology modes the model outputs
`===FINAL_ANSWER===` and `===DERIVATION===` sections. Only the final answer
is passed to RAGAS, so node IDs and citation text aren't judged as factual
claims.

**RAGAS setup.** Claude (the same `MODEL_NAME`) acts as judge via
`langchain-anthropic`; embeddings come from the local sentence-transformer
via a small adapter. For each mode the retrieved contexts scored are: none
for `bare_llm`; text chunks for the RAG baselines; graph frames for
`ontology_graph_only`; graph frames + text chunks for
`ontology_contextual_rag`.

## 5. Notes and tunables

- Model used: `claude-haiku-4-5-20251001`. Swap `MODEL_NAME` in `app.py`
  to change the answering and judge model. Embeddings use
  `sentence-transformers/all-MiniLM-L6-v2` (`EMBED_MODEL_NAME`).
- `answer_ontology_contextual_rag` parameters, if precision/recall shift as
  your ontology or eval set grows:
  - `k_graph` / `k_vec` — max ontology nodes / text chunks (default 2 / 3).
  - `graph_min_score` / `vec_min_score` — minimum relevance to keep a
    candidate (default 0.45 / 0.25). Raise them if irrelevant context slips
    in; lower them if `context_recall` drops. Run `--diagnose` after any
    ontology edit to see the new score distribution before retuning.
  - `vec_candidate_pool` (default 9) — text candidates fetched at the looser
    threshold before lexical rescoring cuts them to `k_vec`.
  - Text chunks are deduped (`VectorIndex.dedupe_chunks`) to drop
    near-identical chunks caused by chunk overlap.
- `ontology_graph_only` uses the same graph settings as the full pipeline
  and only removes the text track, so it isolates what the ontology alone can
  answer versus ontology + text.
- Basic RAG and Hybrid RAG default to `k=5` chunks. Hybrid scores are
  `0.75 × embedding score + 0.25 × lexical overlap`.
- If your TTL uses different predicate names, update the namespaces and the
  `_DEFINITION_PREDS` / `_SOURCE_PREDS` lists in `OntologyStore`.
- RAGAS calls the LLM many times per metric, so `--eval` is slow and costs
  real API calls (roughly a thousand or more for 60 questions across all
  modes). Scope it down with `--modes`, or run it overnight — checkpointing
  means an interruption doesn't force a full restart.
