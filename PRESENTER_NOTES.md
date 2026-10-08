# Presenter notes — OC-RAG

Use the 96-second video first, then show one live trace in the website.

## 00:00 — Start with a question

A student asks a campus question. The system must find supporting evidence before the final answer is written.

Example question: “What is the difference between BSCS and BSIT?”

## 00:12 — Prepare the retrieval query

The system reuses a retrieval query, translating it to English when needed. Semantic ontology candidates help identify relevant entities.

The question guides retrieval; similarity alone is not a verified answer.

## 00:24 — Plan, validate, execute

A planner proposes graph queries or traversals using the ontology catalog. Code validates and executes the plan to retrieve recorded relationships.

Invalid plans can be repaired. If execution yields no evidence, an approximate fallback is explicitly flagged.

## 00:36 — Find supporting text

Dense search finds semantically related text. BM25 independently finds keyword matches. Resolved graph entities can expand the text query.

In this implementation, graph retrieval precedes supporting text retrieval.

## 00:48 — Combine the rankings

Reciprocal-rank fusion combines the dense and BM25 result lists. Exact duplicate passages are removed.

RRF adds 1 / (60 + rank) across result lists. Its score is not an answer-confidence percentage.

## 01:00 — Select the evidence

Executed graph evidence and selected text form the answer context. Complete executed graph results are preserved; exact duplicate blocks are removed.

Source and scope qualifiers stay attached. Broader questions can receive a larger text budget.

## 01:12 — Generate from the context

The configured language model is instructed to answer from the supplied evidence, preserve qualifiers, and state gaps or conflicts.

No retrieved evidence → the pipeline returns an insufficient-evidence response. Grounding reduces risk; it does not guarantee correctness.

## 01:24 — Inspect and evaluate

The answer trace records the graph plan, evidence, fallback notices, selected context and answer. The visualizer replays that record without another model call.

Use live traces for concrete examples and evaluation results for measured claims. This walkthrough is an illustration, not a benchmark.

## Suggested closing explanation

“The ontology contributes structured relationships that the system can query and validate. Hybrid text retrieval adds supporting passages. The final model receives selected evidence, while the trace lets us inspect what was actually returned. We evaluate the answers separately; the animation does not establish accuracy.”
