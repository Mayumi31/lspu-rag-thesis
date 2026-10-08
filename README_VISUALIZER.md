# OC-RAG presentation update

## Install

Back up your project, then merge this package into the folder containing web_app.py. Replace the supplied web_app.py, templates/visualizer.html, static/css/style.css and static/js/main.js. Add static/js/walkthrough-data.js. This package preserves the earlier UI-update routes and chat source panel. Your app.py, catalog, photos and retrieval data are not replaced.

Run `python web_app.py`, open `http://127.0.0.1:5000/visualizer`, and hard-refresh with Ctrl+F5.

## Show your professor

1. Start with **Illustrated walkthrough**. Select Full screen, then Play. Pause or use the arrow controls to explain each stage. Space toggles playback when focus is outside a form control; arrow keys step through scenes.
2. Switch to **Live retrieval trace**, enter a question and choose Ontology Contextual RAG.
3. Click Run retrieval once. When it completes, step through the stored graph plan, returned evidence, selected context and answer.
4. Expand **Inspect the recorded evidence** to show concrete support. Save trace JSON if you need to retain the example.

Replay does not trigger another model call. A single RAG answer run may itself include translation, planning/repair and final-generation calls, as implemented by app.py.

## Video

OC_RAG_Explained.mp4 is a separate 96-second, 1280×720 H.264 video with animated flow indicators and burned-in captions. It is silent so you can narrate it yourself. The video explains the supplied oc-rag-3.6-schema-guidance pipeline; it is labeled as an illustration, not a live run or benchmark. No campus policies, actual retrieved answers, or evaluation scores were invented for it.

## What was corrected

- Removed Claude and Anthropic labels. The model/pipeline label comes from the imported app.py configuration rather than hard-coded provider branding.
- Removed the duplicated, outdated retrieval reconstruction from web_app.py.
- Each live run now invokes the selected answer method once and builds the visual replay from its returned AnswerTrace.
- Supports all five supplied modes, including ontology_graph_only.
- Describes dense + BM25 reciprocal-rank fusion, replacing the old 0.75/0.25 overlap explanation.
- Displays graph plans, actual returned graph evidence, source passages, selected_contexts, validation notices, approximate-fallback flags and the returned answer.
- Does not present model derivations or hidden reasoning as evidence.
- Separates illustrated concepts from executed trace playback. Unrecorded candidate rankings and intermediate timing are not fabricated.

## Scope and configuration

This update targets the app.py supplied in this conversation (oc-rag-3.6-schema-guidance). That file uses OpenAIClient and OPENAI_MODEL. The update does not convert the backend to Ollama or change its model provider. If your local app.py has since been replaced with a different implementation, its constructor and answer-trace interface must remain compatible.

The conceptual walkthrough works without a successful model connection once the Flask site has started. Live retrieval still requires your existing backend dependencies, ontology, text index and working model configuration. If the model is unavailable, the live UI reports that rather than showing invented evidence.

The visualizer is a replay of a completed run, not real-time telemetry. RRF scores are ranking scores, not confidence probabilities. Grounding instructions do not prove the answer is correct; use evidence inspection and evaluation to support your claims.

## Verification

Python and JavaScript syntax checks passed. Mocked Flask tests covered template rendering, model metadata, all five modes, one answer-method call per requested mode, selected-context fidelity, fallback notices, unavailable configuration and isolated failures. DOM interaction tests also passed for playback controls, mode switching, live requests, evidence display and replay without another request. The video was rendered and all eight scene layouts inspected; codec, dimensions and 96-second duration were checked. Real LLM generation and retrieval could not be tested without your configured runtime and data.
