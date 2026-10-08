"""Flask portal and trace-driven OC-RAG presentation.
Run python web_app.py. Uses the existing app.py model and retrieval configuration.
The visualizer replays evidence from the actual answer trace, not a second retrieval.
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
    OpenAIClient,
    OntologyStore,
    RAGSystem,
    VectorIndex,
    build_vector_index_from_text_file,
)

from catalog import COLLEGES
import app as rag_core
from dataclasses import asdict, is_dataclass

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
    _llm = OpenAIClient()
    print("[web_app] OPENAI_API_KEY found — generated answers are ON.")
except Exception as e:
    print(f"[web_app][INFO] No OPENAI_API_KEY ({e}). Retrieval-only demo (answers OFF).")

_system = RAGSystem(vector_index=_vector_index, ontology=_ontology, llm=_llm)


# Presentation uses a single executed answer trace, never a parallel retrieval.
MODE_META = {
    "bare_llm": {"title": "Bare LLM", "subtitle": "Question only; no retrieved evidence"},
    "basic_rag": {"title": "Basic RAG", "subtitle": "Dense text retrieval"},
    "hybrid_rag": {"title": "Hybrid RAG", "subtitle": "Dense + BM25, combined with reciprocal-rank fusion"},
    "ontology_graph_only": {"title": "Ontology Graph Only", "subtitle": "Graph planning and executed evidence; no text retrieval"},
    "ontology_contextual_rag": {"title": "Ontology Contextual RAG", "subtitle": "Executed graph evidence + hybrid text retrieval", "proposed": True},
}


def model_info():
    return {"model": str(getattr(rag_core, "MODEL_NAME", "Configured model")),
            "client": type(_llm).__name__ if _llm is not None else "Not configured",
            "pipeline": str(getattr(rag_core, "PIPELINE_VERSION", "Unspecified"))}


def build_trace(question: str, mode: str) -> Dict[str, Any]:
    meta = MODE_META.get(mode, {"title": mode, "subtitle": "Configured architecture"})
    def step(title, narration, kind="note", **extra):
        return {"stage": title, "narration": narration, "type": kind, **extra}
    steps = [step("Your question", "This is the original input submitted to the system.", "query", text=question)]
    if _llm is None:
        steps.append(step("Model unavailable", "Configure the model client before running a live trace. The illustrated walkthrough remains available.", "error"))
        return {"mode": mode, "meta": meta, "steps": steps, "trace_kind": "unavailable"}
    method = getattr(_system, "answer_" + mode, None)
    if not callable(method):
        raise ValueError("Unsupported architecture: " + mode)
    raw_answer, trace = method(question)  # The ONLY answer/retrieval call.
    record = asdict(trace) if is_dataclass(trace) else (trace if isinstance(trace, dict) else vars(trace))
    if mode != "bare_llm":
        rq = record.get("retrieval_query") or question
        steps.append(step("Retrieval question", "The stored retrieval query is reused across retrieval stages. It may be translated to English.", text=rq))
    else:
        steps.append(step("No retrieval", "This baseline uses the model's existing knowledge; no campus documents or ontology evidence are supplied."))
    if mode in ("ontology_contextual_rag", "ontology_graph_only"):
        steps.append(step("Graph plan", "Recorded graph-query or traversal plan. The system resolves candidates, validates the plan and attempts execution. A stored plan alone does not prove successful execution.", "plan", data=record.get("graph_plan") or {}))
        hits = record.get("graph_hits") or []
        fallback = bool(record.get("approximate_fallback"))
        steps.append(step("Approximate graph evidence" if fallback else "Graph evidence", "Approximate fallback was used; this is not an exhaustive graph result." if fallback else "These graph evidence blocks were returned by the executed pipeline.", "evidence", items=[{"title": h.get("label") or "Graph evidence", "text": h.get("frame") or h.get("definition") or ""} for h in hits], fallback=fallback))
        if record.get("retrieval_errors"):
            steps.append(step("Retrieval notices", "The trace records these validation or retrieval issues, including any repaired attempts.", "evidence", items=[{"title": "Recorded notice", "text": str(e)} for e in record["retrieval_errors"]]))
    if mode in ("basic_rag", "hybrid_rag", "ontology_contextual_rag"):
        hybrid = mode != "basic_rag"
        steps.append(step("Selected text evidence", "The current code combines dense and BM25 rankings with reciprocal-rank fusion, then selects text evidence. Individual candidate rankings are not recorded in this trace." if hybrid else "These are the text chunks returned by dense retrieval.", "evidence", items=[{"title": str(c.get("source") or "Text chunk"), "text": str(c.get("text") or ""), "score": c.get("score"), "score_label": "RRF score (not a probability)" if hybrid else "Similarity score"} for c in record.get("retrieved_chunks", [])]))
    contexts = record.get("selected_contexts")
    if mode != "bare_llm":
        if contexts is not None:
            steps.append(step("Context sent to the model", "Actual selected evidence, in its recorded order. Graph facts and text retain their source qualifiers; exact duplicate blocks are removed.", "evidence", items=[{"title": "Context " + str(i + 1), "text": str(c)} for i, c in enumerate(contexts)]))
        else:
            steps.append(step("Context preview", "This older trace contains only a context preview; it is not the full prompt.", text=record.get("prompt_context_preview") or "No preview recorded."))
    splitter = getattr(_system, "_split_final_answer", None)
    answer = splitter(raw_answer)[0] if splitter else raw_answer
    steps.append(step("Returned answer", "The answer from this same run. Inspect the evidence to assess support; retrieval does not guarantee correctness.", "answer", text=answer))
    return {"mode": mode, "meta": meta, "steps": steps, "trace_kind": "executed",
            "elapsed_seconds": record.get("elapsed_seconds"), "model_info": model_info(),
            "trace": record, "answer": answer}


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
# Multi-page site (Explore / College Details / Others / Assistant)
# ---------------------------------------------------------------------
@app.route("/explore")
def explore():
    """College cards and the campus map."""
    return redirect(url_for("college_details"))


@app.route("/map")
def campus_map():
    return render_template("map.html")


@app.route("/college-details")
def college_details():
    """Full program details for every college (picture, description,
    study areas, and the field/activities/skills/careers blocks).
    A page section is shown/highlighted client-side via the URL hash,
    e.g. /college-details#computer."""
    return render_template("college-details.html")


@app.route("/others")
def others():
    """Recommendation tool + About + Contact, combined into one page
    with #recommendation / #about / #contact sections."""
    return render_template("others.html")


@app.route("/assistant")
def assistant():
    """Full-page Knowledge Assistant. Talks to the same /api/chat endpoint
    as the homepage chat preview (Ontology Contextual RAG only)."""
    return render_template("assistant.html", llm_enabled=_llm is not None)


@app.route("/programs")
def programs_legacy():
    """Old URL: programs now live on the College Details page."""
    return redirect(url_for("college_details"), code=301)


@app.route("/recommendation")
def recommendation_legacy():
    """Old URL: recommendation now lives on the Others page."""
    return redirect(url_for("college_details"))


@app.route("/about")
def about_legacy():
    """Old URL: about now lives on the Others page."""
    return redirect(url_for("others") + "#about", code=301)


@app.route("/contact")
def contact_legacy():
    """Old URL: contact now lives on the Others page."""
    return redirect(url_for("others") + "#contact", code=301)


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
        "model_info": model_info(),
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

    results = []
    for mode in modes:
        try:
            results.append(build_trace(question, mode))
        except Exception:
            app.logger.exception("Visualizer run failed for %s", mode)
            results.append({"mode": mode, "meta": MODE_META.get(mode, {"title": mode}),
                            "trace_kind": "failed", "steps": [{"stage": "Run failed", "type": "error",
                            "narration": "The backend could not complete this run. Check the server log and model connection, then retry."}]})
    return jsonify({"question": question, "results": results})


def chat_sources(trace):
    """Expose actual selected retrieval evidence, never model reasoning."""
    def field(key, default):
        return trace.get(key, default) if isinstance(trace, dict) else getattr(trace, key, default)
    contexts = field("selected_contexts", []) or []
    if contexts:
        return [{"title": "Knowledge base evidence " + str(i + 1), "excerpt": str(text)}
                for i, text in enumerate(contexts)]
    sources = []
    for chunk in field("retrieved_chunks", []) or []:
        sources.append({"title": str(chunk.get("source") or "Retrieved document"),
                        "excerpt": str(chunk.get("text") or "")})
    for hit in field("graph_hits", []) or []:
        sources.append({"title": str(hit.get("source_article") or hit.get("label") or "Ontology evidence"),
                        "excerpt": str(hit.get("frame") or hit.get("definition") or "")})
    return sources


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
                "OPENAI_API_KEY configured on the server to generate answers. "
                "Please let the site administrator know."
            ),
            "configured": False,
        })

    try:
        raw_answer, _trace = _system.answer_ontology_contextual_rag(question)
        # answer_ontology_contextual_rag's raw text is marker-delimited
        # (===FINAL_ANSWER=== ... ===DERIVATION=== ...); the chat only
        # ever shows the clean final answer, never the derivation section.
        splitter = getattr(_system, "_split_final_answer", None)
        final_answer = splitter(raw_answer)[0] if splitter else raw_answer
        return jsonify({"answer": final_answer, "configured": True,
                        "sources": chat_sources(_trace)})
    except Exception as e:
        return jsonify({
            "error": f"Something went wrong generating an answer ({e}). Please try again.",
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[web_app] Open http://127.0.0.1:{port} in your browser.\n")
    app.run(debug=True, port=port)