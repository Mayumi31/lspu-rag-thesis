#!/usr/bin/env python3
"""
retrieval_check.py — inspect what the retriever finds WITHOUT any LLM call (zero tokens).

It loads the same embedder, ontology and vector index as app.py and runs the same
retrieval steps and thresholds as answer_ontology_contextual_rag /
answer_ontology_graph_only, but never creates a ClaudeClient (no API key needed).

Because there is no query-translation call, type queries in ENGLISH (for a Tagalog
question, type its English equivalent — that is what the real pipeline searches with).

Run from the project folder (same place you run app.py):

  python retrieval_check.py "Which college offers BS Psychology?"
  python retrieval_check.py "min GWA for BS Computer Science" --frame
  python retrieval_check.py "Which college offers BS Psychology?" --expect "College of Arts and Sciences"
  python retrieval_check.py --eval                 # every question in eval/eval_set.jsonl
  python retrieval_check.py --eval --summary-only  # one line per question
  python retrieval_check.py --node lspuprog:Prog_BSPsych   # exact frame the model would see for a node
  python retrieval_check.py --predicates           # list the ontology's relations (for hop-2 settings)

Useful switches: --top N, --frame, --no-hop2 (A/B the 2nd hop), --no-text (skip the text index),
--k-graph/--graph-min/--k-vec/--vec-min (try thresholds without editing app.py).
"""
import argparse
import inspect
import json
import re
from typing import Any, Dict, List

import app
from app import (
    EMBED_MODEL_NAME, EVAL_SET, PILLAR3_TTL, PILLAR3_TXT,
    OntologyStore, RAGSystem, VectorIndex, SentenceTransformer,
    build_vector_index_from_text_file, load_eval_set,
)


def _tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 3}


def _cov(ref: set, text: str) -> float:
    if not ref:
        return 0.0
    return len(ref & set(re.findall(r"[a-z0-9]+", text.lower()))) / len(ref)


def _defaults() -> Dict[str, Any]:
    p = inspect.signature(RAGSystem.answer_ontology_contextual_rag).parameters
    d = {k: p[k].default for k in ("k_graph", "k_vec", "graph_min_score", "vec_min_score", "vec_candidate_pool")}
    d["score_margin"] = inspect.signature(OntologyStore.search_nodes_by_text).parameters["score_margin"].default
    return d


def check_query(ontology, vector_index, q: str, cfg: Dict[str, Any], top: int, show_frame: bool,
                expects: List[str], ref: set, hop2: bool, quiet: bool = False) -> Dict[str, Any]:
    """Runs graph + text retrieval for one query. Returns numbers for the summary table."""
    result = {"query": q, "kept_graph": [], "kept_text": [], "graph_cov": 0.0, "text_cov": 0.0, "top_score": 0.0}
    if not quiet:
        print("\n" + "=" * 100)
        print("QUERY:", q)

    # ---- graph track (same calls as answer_ontology_contextual_rag) ----
    g_cands, kept_g = [], []
    if ontology:
        g_cands = ontology.search_nodes_by_text(
            q, limit=top, min_score=0.0, score_margin=10.0, expand_second_hop=hop2)
        kept_g = OntologyStore.dedupe_hits(
            ontology.search_nodes_by_text(
                q, limit=cfg["k_graph"], min_score=cfg["graph_min_score"],
                score_margin=cfg["score_margin"], expand_second_hop=hop2))
        kept_ids = {h.node for h in kept_g}
        result["top_score"] = g_cands[0].score if g_cands else 0.0
        result["kept_graph"] = [h.label or h.node for h in kept_g]
        result["graph_cov"] = _cov(ref, "\n".join(h.frame for h in kept_g))
        if not quiet:
            print(f"\n-- GRAPH (keep <= {cfg['k_graph']}, score >= {cfg['graph_min_score']}, "
                  f"within {cfg['score_margin']} of best)   * = would be sent to the model")
            for i, h in enumerate(g_cands, 1):
                mark = "*" if h.node in kept_ids else " "
                cov = f" cov={_cov(ref, h.frame):.2f}" if ref else ""
                print(f" {mark} #{i} {h.score:.3f}{cov}  {h.node}  |  {(h.label or '')[:70]}"
                      f"  [{len(h.neighbors)} rel, {len(h.second_hop)} hop2, {len(h.frame)} chars]")
            if not kept_g:
                print("   (nothing passed the thresholds -> graph-only mode would say it has no context)")

    # ---- text track ----
    t_cands, kept_t = [], []
    if vector_index:
        t_cands = VectorIndex.rescore_lexical(q, vector_index.search(q, k=max(cfg["vec_candidate_pool"], top), min_score=0.0))[:top]
        kept_t = VectorIndex.dedupe_chunks(VectorIndex.rescore_lexical(
            q, vector_index.search(q, k=cfg["vec_candidate_pool"], min_score=cfg["vec_min_score"])))[:cfg["k_vec"]]
        kept_tid = {c.chunk_id for c in kept_t}
        result["kept_text"] = [c.chunk_id for c in kept_t]
        result["text_cov"] = _cov(ref, "\n".join(c.text for c in kept_t))
        if not quiet:
            print(f"\n-- TEXT (full pipeline only; keep <= {cfg['k_vec']}, raw score >= {cfg['vec_min_score']})   * = kept")
            for i, c in enumerate(t_cands, 1):
                mark = "*" if c.chunk_id in kept_tid else " "
                cov = f" cov={_cov(ref, c.text):.2f}" if ref else ""
                print(f" {mark} #{i} {c.score:.3f}{cov}  {c.chunk_id}  |  {c.text.replace(chr(10), ' ')[:80]}")

    if quiet:
        return result

    if ref:
        print(f"\n   reference coverage of what would be sent: graph={result['graph_cov']:.2f}"
              + (f"  text={result['text_cov']:.2f}" if vector_index else ""))

    if show_frame:
        for h in kept_g:
            print("\n" + "-" * 40 + f" FRAME sent to model: {h.node} ({len(h.frame)} chars)")
            print(h.frame)

    for ex in expects:
        exl = ex.lower()
        in_g = [h.label or h.node for h in kept_g if exl in h.frame.lower()]
        in_t = [c.chunk_id for c in kept_t if exl in c.text.lower()]
        if in_g or in_t:
            print(f"\n   [OK]   {ex!r} is in the context: graph={in_g or '-'} text={in_t or '-'}")
        else:
            near = [f"graph #{i} (score {h.score:.3f})" for i, h in enumerate(g_cands, 1) if exl in h.frame.lower()]
            near += [f"text #{i} ({c.chunk_id})" for i, c in enumerate(t_cands, 1) if exl in c.text.lower()]
            if near:
                print(f"\n   [MISS] {ex!r} exists among candidates but was NOT kept: {', '.join(near)}"
                      " -> lower a threshold, or fix ranking / node text")
            else:
                print(f"\n   [MISS] {ex!r} is not in any top-{top} candidate -> retrieval or ontology content problem")
    return result


def show_node(ontology: OntologyStore, ident: str, hop2: bool):
    ident_l = ident.lower()
    exact = [s for s in set(ontology.g.subjects())
             if ident in (ontology._shorten(s), str(s)) or ident_l == ontology._label(s).lower()
             or ident == re.split(r"[#/:]", str(s))[-1]]
    if not exact:
        part = sorted({ontology._shorten(s) for s in set(ontology.g.subjects())
                       if ident_l in ontology._shorten(s).lower() or ident_l in ontology._label(s).lower()})
        print(f"No node matches {ident!r}." + (f" Did you mean: {part[:15]}" if part else ""))
        return
    for s in exact[:3]:
        h = ontology._build_hit(s, 1.0, True, hop2)
        print("=" * 100)
        print(f"NODE {h.node}   ({len(h.frame)} chars)\n")
        print(h.frame)


def show_predicates(ontology: OntologyStore):
    counts: Dict[Any, int] = {}
    for _, p, _ in ontology.g:
        if not ontology._is_structural(p):
            counts[p] = counts.get(p, 0) + 1
    print(f"{'count':>6}  {'readable name':<40} {'uri':<45} flags")
    for p, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        name = ontology._pred_name(p)
        flags = []
        if any(a in name for a in ontology._PRIORITY_LINK_PREDS):
            flags.append("priority")
        if any(a in name for a in ontology._HOP2_FOLLOW_PREDS):
            flags.append("hop2")
        print(f"{n:>6}  {name:<40} {ontology._shorten(p):<45} {','.join(flags)}")
    print("\nEdit _PRIORITY_LINK_PREDS / _HOP2_FOLLOW_PREDS in OntologyStore (app.py) to change the flags.")


def main():
    d = _defaults()
    ap = argparse.ArgumentParser(description="Zero-token retrieval inspector", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("queries", nargs="*", help="English query text(s)")
    ap.add_argument("--eval", action="store_true", help="use every question in the eval set (adds reference coverage)")
    ap.add_argument("--summary-only", action="store_true", help="with --eval: one line per question")
    ap.add_argument("--node", help="print the exact frame for a node id (e.g. lspuprog:Prog_BSPsych) or label")
    ap.add_argument("--predicates", action="store_true", help="list ontology predicates with counts")
    ap.add_argument("--expect", action="append", default=[], help="text that MUST be in the context (repeatable)")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--frame", action="store_true", help="print the full frames that would be sent")
    ap.add_argument("--no-hop2", action="store_true", help="disable 2nd-hop expansion (A/B comparison)")
    ap.add_argument("--no-text", action="store_true", help="skip the text index")
    ap.add_argument("--k-graph", type=int, default=d["k_graph"])
    ap.add_argument("--graph-min", type=float, default=d["graph_min_score"])
    ap.add_argument("--k-vec", type=int, default=d["k_vec"])
    ap.add_argument("--vec-min", type=float, default=d["vec_min_score"])
    args = ap.parse_args()

    if not (args.queries or args.eval or args.node or args.predicates):
        ap.print_help()
        return

    print("Loading embedder + ontology (no LLM, no API calls)...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    ontology = OntologyStore(PILLAR3_TTL, embedder=embedder)
    ontology.load()
    vector_index = None if args.no_text else build_vector_index_from_text_file(embedder, PILLAR3_TXT)
    run(args, d, ontology, vector_index)


def run(args, d, ontology, vector_index):
    cfg = {"k_graph": args.k_graph, "graph_min_score": args.graph_min, "k_vec": args.k_vec,
           "vec_min_score": args.vec_min, "vec_candidate_pool": d["vec_candidate_pool"], "score_margin": d["score_margin"]}
    hop2 = not args.no_hop2

    if args.predicates:
        show_predicates(ontology)
    if args.node:
        show_node(ontology, args.node, hop2)

    for q in args.queries:
        check_query(ontology, vector_index, q, cfg, args.top, args.frame, args.expect, set(), hop2)

    if args.eval:
        rows = load_eval_set(EVAL_SET)
        summary = []
        for r in rows:
            q, gt = r["question"], r.get("ground_truth", "")
            ref = _tokens(gt)
            res = check_query(ontology, vector_index, q, cfg, args.top, args.frame, args.expect, ref, hop2,
                              quiet=args.summary_only)
            if not args.summary_only:
                print("   REF:", gt[:160])
            summary.append(res)
        print("\n" + "=" * 100)
        print(f"SUMMARY  (graph cov = share of reference-answer words present in the kept graph frames)")
        print(f"{'#':>3} {'top':>5} {'kept':>4} {'g-cov':>6} {'t-cov':>6}  question")
        for i, r in enumerate(summary, 1):
            flag = "  <-- nothing kept" if not r["kept_graph"] else ("  <-- low coverage" if r["graph_cov"] < 0.5 else "")
            print(f"{i:>3} {r['top_score']:>5.2f} {len(r['kept_graph']):>4} {r['graph_cov']:>6.2f} {r['text_cov']:>6.2f}  {r['query'][:60]}{flag}")
        if summary:
            n = len(summary)
            print(f"\nmean graph cov = {sum(r['graph_cov'] for r in summary)/n:.2f}   "
                  f"mean text cov = {sum(r['text_cov'] for r in summary)/n:.2f}   "
                  f"questions with nothing kept = {sum(1 for r in summary if not r['kept_graph'])}/{n}")


if __name__ == "__main__":
    main()
