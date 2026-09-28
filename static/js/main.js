const els = {
  question: document.getElementById("question"),
  runBtn: document.getElementById("run-btn"),
  board: document.getElementById("board"),
  boardEmpty: document.getElementById("board-empty"),
  pillVector: document.getElementById("pill-vector"),
  pillOntology: document.getElementById("pill-ontology"),
  pillLlm: document.getElementById("pill-llm"),
  controlsBar: document.getElementById("controls-bar"),
  playPauseBtn: document.getElementById("play-pause-btn"),
  stepBtn: document.getElementById("step-btn"),
  restartBtn: document.getElementById("restart-btn"),
  speedSelect: document.getElementById("speed-select"),
  controlStatus: document.getElementById("control-status"),
  focusExitWrap: document.getElementById("focus-exit-wrap"),
  exitFocusBtn: document.getElementById("exit-focus-btn"),
  modeSelect: document.getElementById("mode-select"),
  modeSelectBtn: document.getElementById("mode-select-btn"),
  modeSelectLabel: document.getElementById("mode-select-label"),
  modeSelectPanel: document.getElementById("mode-select-panel"),
  modeSelectAllBtn: document.getElementById("mode-select-all"),
  modeSelectList: document.getElementById("mode-select-list"),
};

let MODE_META = {};
let MODES = [];
let LAST_RESULTS = null; // so "Replay" doesn't need a re-fetch

// ---------------------------------------------------------------------
// Playback engine
//
// Each "player" owns one mode-column: its full step list, how many
// steps have been rendered so far (currentIndex), and the DOM refs it
// needs to paint into. A single global interval calls tick() on every
// player at once so all columns advance in lockstep, which is what
// makes it easy to narrate ("okay, now watch all three do their vector
// search step..."). Pause/Step/Replay just control that shared timer.
// ---------------------------------------------------------------------
const player = {
  columns: [],       // [{ steps, currentIndex, bodyEl, groups }]
  timer: null,
  playing: false,
  finished: false,
};

function speedMs() {
  return parseInt(els.speedSelect.value, 10) || 1800;
}

function startAutoPlay() {
  stopAutoPlay();
  player.playing = true;
  els.playPauseBtn.textContent = "\u2758\u2758 Pause";
  els.controlStatus.textContent = "Playing…";
  player.timer = setInterval(tickAll, speedMs());
}

function stopAutoPlay() {
  if (player.timer) clearInterval(player.timer);
  player.timer = null;
  player.playing = false;
}

function pauseAutoPlay() {
  stopAutoPlay();
  els.playPauseBtn.textContent = "\u25B6 Play";
  els.controlStatus.textContent = "Paused";
}

function tickAll() {
  let anyAdvanced = false;
  player.columns.forEach((col) => {
    if (col.currentIndex < col.steps.length) {
      const step = col.steps[col.currentIndex];
      const block = renderStep(step);
      if (block) col.bodyEl.appendChild(block);
      col.currentIndex++;
      updatePipeline(col);
      anyAdvanced = true;
    }
  });
  if (!anyAdvanced) {
    player.finished = true;
    stopAutoPlay();
    els.playPauseBtn.textContent = "\u25B6 Play";
    els.controlStatus.textContent = "Done — all steps shown";
  }
}

// ---------------------------------------------------------------------
// Boot: fetch status + mode metadata, build the mode-select dropdown
// ---------------------------------------------------------------------
async function boot() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    MODES = data.modes;
    MODE_META = data.mode_meta;

    setPill(els.pillVector, "vector index", data.vector_loaded);
    setPill(els.pillOntology, "ontology graph", data.ontology_loaded);
    setPill(els.pillLlm, "Claude answers", data.llm_enabled);
    if (!data.llm_enabled) {
      els.pillLlm.title = "Add ANTHROPIC_API_KEY to .env to see generated final answers.";
    }

    buildModeSelectList();
    updateModeSelectLabel();
  } catch (e) {
    console.error("Failed to load /api/status", e);
  }
}

function setPill(el, label, on) {
  el.textContent = label + (on ? "" : " (off)");
  el.classList.toggle("on", !!on);
}

// ---------------------------------------------------------------------
// Single-button mode picker: click opens a small popover with one
// checkbox per architecture + a "select all / none" shortcut. The
// button's own label always summarizes the current selection so you
// don't need the panel open to see what's about to run.
// ---------------------------------------------------------------------
function buildModeSelectList() {
  els.modeSelectList.innerHTML = "";
  MODES.forEach((mode) => {
    const meta = MODE_META[mode] || {};
    const row = document.createElement("label");
    row.className = "mode-option" + (meta.proposed ? " proposed" : "");
    row.innerHTML = `
      <input type="checkbox" value="${mode}" checked />
      <span class="mode-option-text">
        <span class="mode-option-title">${meta.title || mode}</span>
        <span class="mode-option-subtitle">${meta.subtitle || ""}</span>
      </span>
    `;
    row.querySelector("input").addEventListener("change", updateModeSelectLabel);
    els.modeSelectList.appendChild(row);
  });
}

function selectedModes() {
  return Array.from(els.modeSelectList.querySelectorAll("input:checked")).map((i) => i.value);
}

function updateModeSelectLabel() {
  const selected = selectedModes();
  if (selected.length === MODES.length) {
    els.modeSelectLabel.textContent = `All ${MODES.length} architectures`;
  } else if (selected.length === 0) {
    els.modeSelectLabel.textContent = "None selected";
  } else if (selected.length === 1) {
    const meta = MODE_META[selected[0]] || {};
    els.modeSelectLabel.textContent = meta.title || selected[0];
  } else {
    els.modeSelectLabel.textContent = `${selected.length} of ${MODES.length} selected`;
  }
}

function toggleModeSelectPanel(forceOpen) {
  const isHidden = els.modeSelectPanel.hidden;
  const openIt = forceOpen !== undefined ? forceOpen : isHidden;
  els.modeSelectPanel.hidden = !openIt;
}

els.modeSelectBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  toggleModeSelectPanel();
});

els.modeSelectAllBtn.addEventListener("click", () => {
  const boxes = els.modeSelectList.querySelectorAll("input");
  const allChecked = Array.from(boxes).every((b) => b.checked);
  boxes.forEach((b) => (b.checked = !allChecked));
  updateModeSelectLabel();
});

document.addEventListener("click", (e) => {
  if (!els.modeSelect.contains(e.target)) toggleModeSelectPanel(false);
});

// ---------------------------------------------------------------------
// Run a comparison query
// ---------------------------------------------------------------------
async function runQuery() {
  const question = els.question.value.trim();
  if (!question) {
    els.question.focus();
    return;
  }
  const modes = selectedModes();
  if (modes.length === 0) {
    toggleModeSelectPanel(true);
    return;
  }
  toggleModeSelectPanel(false);

  els.runBtn.disabled = true;
  els.runBtn.textContent = "Retrieving…";

  try {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, modes }),
    });
    const data = await res.json();
    if (data.error) {
      alert(data.error);
      return;
    }
    LAST_RESULTS = data.results;
    startPlayback(data.results);
  } catch (e) {
    console.error(e);
    alert("Something went wrong talking to the server. Check the terminal running web_app.py.");
  } finally {
    els.runBtn.disabled = false;
    els.runBtn.textContent = "Run comparison";
  }
}

function startPlayback(results) {
  stopAutoPlay();
  exitFocus();
  els.boardEmpty.style.display = "none";
  els.board.innerHTML = "";
  els.controlsBar.hidden = false;

  player.columns = results.map((result) => buildColumn(result));
  player.finished = false;
  startAutoPlay();
}

// ---------------------------------------------------------------------
// Build one mode column: DOM + pipeline strip + step queue (nothing is
// rendered into the body yet — tickAll() reveals steps one at a time)
// ---------------------------------------------------------------------
function buildColumn(result) {
  const tpl = document.getElementById("tpl-column");
  const frag = tpl.content.cloneNode(true);
  const col = frag.querySelector(".mode-col");
  const title = frag.querySelector(".mode-title");
  const subtitle = frag.querySelector(".mode-subtitle");

  const meta = result.meta || {};
  if (meta.proposed) col.classList.add("proposed");
  title.textContent = meta.title || result.mode;
  subtitle.textContent = meta.subtitle || "";

  els.board.appendChild(frag);
  const liveCol = els.board.lastElementChild;
  const bodyEl = liveCol.querySelector(".mode-col-body");
  const livePipelineEl = liveCol.querySelector(".pipeline-strip");
  const liveFocusBtn = liveCol.querySelector(".focus-btn");

  const groups = buildStageGroups(result.steps);
  groups.forEach((g) => {
    const chip = el("span", "stage-chip pending", g.stage);
    livePipelineEl.appendChild(chip);
    g.chipEl = chip;
  });

  liveFocusBtn.addEventListener("click", () => toggleFocus(liveCol));

  return {
    mode: result.mode,
    steps: result.steps,
    currentIndex: 0,
    colEl: liveCol,
    bodyEl,
    groups,
  };
}

function buildStageGroups(steps) {
  const groups = [];
  steps.forEach((s, i) => {
    const last = groups[groups.length - 1];
    if (last && last.stage === s.stage) {
      last.max = i;
    } else {
      groups.push({ stage: s.stage, min: i, max: i });
    }
  });
  return groups;
}

function updatePipeline(col) {
  const lastRendered = col.currentIndex - 1;
  col.groups.forEach((g) => {
    g.chipEl.classList.remove("active", "done", "pending");
    if (lastRendered < g.min) {
      g.chipEl.classList.add("pending");
    } else if (lastRendered <= g.max) {
      g.chipEl.classList.add("active");
    } else {
      g.chipEl.classList.add("done");
    }
  });
}

// ---------------------------------------------------------------------
// Focus mode — blow up one column full-width for presenting
// ---------------------------------------------------------------------
function toggleFocus(colEl) {
  const alreadyFocused = colEl.classList.contains("is-focused");
  els.board.querySelectorAll(".mode-col").forEach((c) => c.classList.remove("is-focused"));
  if (alreadyFocused) {
    els.board.classList.remove("has-focus");
    els.focusExitWrap.hidden = true;
  } else {
    colEl.classList.add("is-focused");
    els.board.classList.add("has-focus");
    els.focusExitWrap.hidden = false;
  }
}

function exitFocus() {
  els.board.classList.remove("has-focus");
  els.focusExitWrap.hidden = true;
}

// ---------------------------------------------------------------------
// Step renderers
// ---------------------------------------------------------------------
function renderStep(step) {
  switch (step.type) {
    case "query":
      return renderQuery(step);
    case "note":
      return renderNote(step);
    case "search_start":
      return renderSearching(step);
    case "vector_search":
      return renderVectorSearch(step);
    case "lexical_rescore":
      return renderLexicalRescore(step);
    case "graph_search":
      return renderGraphSearch(step);
    case "frame_assembly":
      return renderFrame(step);
    case "llm_call":
      return renderLlmCall(step);
    case "answer":
      return renderAnswer(step);
    default:
      return null;
  }
}

const MECH_LABEL = {
  none: "context",
  lexical: "lexical rescore",
  vector: "text embedding",
  graph_vector: "ontology embedding",
  merge: "merge",
  generate: "generate",
};

function mechBadge(mechanism) {
  if (!mechanism || mechanism === "none") return null;
  return el("span", `mech-badge mech-${mechanism}`, MECH_LABEL[mechanism] || mechanism);
}

function narrationBox(step) {
  if (!step.narration) return null;
  const box = el("div", "narration");
  const badge = mechBadge(step.mechanism);
  if (badge) box.appendChild(badge);
  box.appendChild(el("span", null, step.narration));
  return box;
}

function renderQuery(step) {
  return el("div", "step-block query-block", `\u2753 “${step.text}”`);
}

function renderNote(step) {
  const wrap = el("div", "step-block");
  const n = narrationBox(step);
  if (n) wrap.appendChild(n);
  if (step.text) wrap.appendChild(el("div", "note-block", step.text));
  return wrap;
}

function renderSearching(step) {
  const wrap = el("div", "step-block searching-block");
  wrap.appendChild(el("div", "spinner"));
  const n = narrationBox(step);
  if (n) {
    n.style.margin = "0";
    n.style.flex = "1";
    wrap.appendChild(n);
  }
  return wrap;
}

function renderVectorSearch(step) {
  const wrap = el("div", "step-block");
  const n = narrationBox(step);
  if (n) wrap.appendChild(n);
  wrap.appendChild(el("div", "step-label", step.label || "Vector search results"));
  const maxScore = Math.max(0.0001, ...step.chunks.map((c) => c.score));
  step.chunks.forEach((c, i) => {
    const card = buildChunkCard(c, i === 0);
    wrap.appendChild(card);
    requestAnimationFrame(() => {
      setTimeout(() => {
        const fill = card.querySelector(".score-bar-fill");
        const pct = Math.max(4, (c.score / maxScore) * 100);
        fill.style.width = pct + "%";
      }, 40 + i * 90);
    });
    card.style.animationDelay = (i * 0.06) + "s";
  });
  if (step.chunks.length === 0) {
    wrap.appendChild(el("div", "note-block", "No chunks cleared the similarity threshold for this question — try one of the example questions, or a question closer to the handbook's wording."));
  }
  return wrap;
}

function renderLexicalRescore(step) {
  const wrap = el("div", "step-block");
  const n = narrationBox(step);
  if (n) wrap.appendChild(n);
  wrap.appendChild(el("div", "step-label", step.label || "Lexical rescoring results"));
  const maxScore = Math.max(0.0001, ...step.chunks.map((c) => c.combined_score));
  step.chunks.forEach((c, i) => {
    const card = buildChunkCard(
      { ...c, score: c.combined_score },
      i === 0,
      `overlap ${(c.lexical_overlap * 100).toFixed(0)}%`
    );
    wrap.appendChild(card);
    requestAnimationFrame(() => {
      setTimeout(() => {
        const fill = card.querySelector(".score-bar-fill");
        const pct = Math.max(4, (c.combined_score / maxScore) * 100);
        fill.style.width = pct + "%";
      }, 40 + i * 90);
    });
    card.style.animationDelay = (i * 0.06) + "s";
  });
  return wrap;
}

function buildChunkCard(c, isTop, extraLabel) {
  const tpl = document.getElementById("tpl-chunk");
  const node = tpl.content.cloneNode(true);
  const card = node.querySelector(".chunk-card");
  if (isTop) card.classList.add("top-hit");
  node.querySelector(".chunk-source").textContent = c.source;
  node.querySelector(".chunk-score").textContent = extraLabel
    ? `${c.score.toFixed(3)} · ${extraLabel}`
    : c.score.toFixed(3);
  node.querySelector(".chunk-text").textContent = c.text;
  return card;
}

function renderGraphSearch(step) {
  const wrap = el("div", "step-block graph-wrap");
  const n = narrationBox(step);
  if (n) wrap.appendChild(n);
  wrap.appendChild(el("div", "step-label", step.label || "Ontology graph search results"));

  if (!step.nodes || step.nodes.length === 0) {
    wrap.appendChild(el("div", "note-block", "No ontology node cleared the similarity threshold for this phrasing — try one of the example questions, or ask about a concept closer to how it's defined in the ontology."));
    return wrap;
  }

  const list = el("div", "graph-nodes-list");
  step.nodes.forEach((nItem, i) => {
    const card = el("div", "graph-node-card");
    card.style.animationDelay = (i * 0.12) + "s";

    const head = el("div", "graph-node-head");
    head.appendChild(el("span", "dot"));
    head.appendChild(document.createTextNode(nItem.label || nItem.node));
    if (typeof nItem.score === "number") {
      head.appendChild(el("span", "graph-node-score", nItem.score.toFixed(2)));
    }
    card.appendChild(head);

    if (nItem.definition) card.appendChild(el("p", "graph-node-def", nItem.definition));
    if (nItem.source_article) card.appendChild(el("div", "graph-node-source", nItem.source_article));

    if (nItem.relations && nItem.relations.length) {
      const rels = el("div", "graph-relations");
      nItem.relations.slice(0, 5).forEach((r, j) => {
        const chip = el("span", "rel-chip", `${r.predicate} \u2192 ${r.object}`);
        chip.style.animationDelay = (0.35 + i * 0.12 + j * 0.06) + "s";
        rels.appendChild(chip);
      });
      card.appendChild(rels);
    }

    list.appendChild(card);
  });
  wrap.appendChild(list);
  return wrap;
}

function renderFrame(step) {
  const wrap = el("div", "step-block");
  const n = narrationBox(step);
  if (n) wrap.appendChild(n);
  wrap.appendChild(el("div", "step-label", step.label || "Context assembly"));
  wrap.appendChild(el("div", "frame-block", step.text));
  return wrap;
}

function renderLlmCall(step) {
  const wrap = el("div", "step-block searching-block");
  wrap.appendChild(el("div", "spinner"));
  const n = narrationBox(step);
  if (n) {
    n.style.margin = "0";
    n.style.flex = "1";
    wrap.appendChild(n);
  } else {
    wrap.appendChild(document.createTextNode("Generating answer with Claude…"));
  }
  return wrap;
}

function renderAnswer(step) {
  const tpl = document.getElementById("tpl-answer");
  const node = tpl.content.cloneNode(true);
  const text = node.querySelector(".answer-text");
  const derivationBlock = node.querySelector(".derivation-block");
  const derivationText = node.querySelector(".derivation-text");

  if (step.error) {
    text.textContent = `Could not generate an answer (${step.error}).`;
    text.classList.add("is-error");
  } else if (step.text) {
    text.textContent = step.text;
    if (step.derivation) {
      derivationText.textContent = step.derivation;
      derivationBlock.hidden = false;
    }
  } else {
    text.textContent = "Retrieval-only mode: add ANTHROPIC_API_KEY to .env to see a generated final answer here.";
    text.classList.add("is-note");
  }
  return node.querySelector(".answer-card");
}

// --- tiny dom helper -----------------------------------------------------
function el(tag, className, text) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined && text !== null) e.textContent = text;
  return e;
}

// ---------------------------------------------------------------------
// Wire up events
// ---------------------------------------------------------------------
els.runBtn.addEventListener("click", runQuery);
els.question.addEventListener("keydown", (e) => {
  if (e.key === "Enter") runQuery();
});
document.querySelectorAll(".example-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    els.question.value = chip.dataset.q;
    runQuery();
  });
});

els.playPauseBtn.addEventListener("click", () => {
  if (player.playing) {
    pauseAutoPlay();
  } else {
    if (player.finished) return; // nothing left to play — use Replay
    startAutoPlay();
  }
});

els.stepBtn.addEventListener("click", () => {
  pauseAutoPlay();
  tickAll();
});

els.restartBtn.addEventListener("click", () => {
  if (LAST_RESULTS) startPlayback(LAST_RESULTS);
});

els.speedSelect.addEventListener("change", () => {
  if (player.playing) startAutoPlay(); // restart interval at new speed
});

els.exitFocusBtn.addEventListener("click", exitFocus);

boot();
