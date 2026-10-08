window.OC_WALKTHROUGH = [
  {
    "stage": "Start with a question",
    "short": "Question",
    "kind": "query",
    "narration": "A student asks a campus question. The system must find supporting evidence before the final answer is written.",
    "detail": "Example question: “What is the difference between BSCS and BSIT?”"
  },
  {
    "stage": "Prepare the retrieval query",
    "short": "Prepare",
    "kind": "prepare",
    "narration": "The system reuses a retrieval query, translating it to English when needed. Semantic ontology candidates help identify relevant entities.",
    "detail": "The question guides retrieval; similarity alone is not a verified answer."
  },
  {
    "stage": "Plan, validate, execute",
    "short": "Graph",
    "kind": "graph",
    "narration": "A planner proposes graph queries or traversals using the ontology catalog. Code validates and executes the plan to retrieve recorded relationships.",
    "detail": "Invalid plans can be repaired. If execution yields no evidence, an approximate fallback is explicitly flagged."
  },
  {
    "stage": "Find supporting text",
    "short": "Text",
    "kind": "text",
    "narration": "Dense search finds semantically related text. BM25 independently finds keyword matches. Resolved graph entities can expand the text query.",
    "detail": "In this implementation, graph retrieval precedes supporting text retrieval."
  },
  {
    "stage": "Combine the rankings",
    "short": "Fuse",
    "kind": "fusion",
    "narration": "Reciprocal-rank fusion combines the dense and BM25 result lists. Exact duplicate passages are removed.",
    "detail": "RRF adds 1 / (60 + rank) across result lists. Its score is not an answer-confidence percentage."
  },
  {
    "stage": "Select the evidence",
    "short": "Context",
    "kind": "context",
    "narration": "Executed graph evidence and selected text form the answer context. Complete executed graph results are preserved; exact duplicate blocks are removed.",
    "detail": "Source and scope qualifiers stay attached. Broader questions can receive a larger text budget."
  },
  {
    "stage": "Generate from the context",
    "short": "Answer",
    "kind": "answer",
    "narration": "The configured language model is instructed to answer from the supplied evidence, preserve qualifiers, and state gaps or conflicts.",
    "detail": "No retrieved evidence → the pipeline returns an insufficient-evidence response. Grounding reduces risk; it does not guarantee correctness."
  },
  {
    "stage": "Inspect and evaluate",
    "short": "Audit",
    "kind": "audit",
    "narration": "The answer trace records the graph plan, evidence, fallback notices, selected context and answer. The visualizer replays that record without another model call.",
    "detail": "Use live traces for concrete examples and evaluation results for measured claims. This walkthrough is an illustration, not a benchmark."
  }
];
