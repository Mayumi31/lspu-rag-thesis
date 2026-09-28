/* ==========================================================
   SHARED: mobile navigation (every page using base.html)
   ========================================================== */
const menuBtn = document.getElementById("menuBtn");
const navLinks = document.getElementById("navLinks");

if (menuBtn && navLinks) {
  menuBtn.addEventListener("click", () => {
    const isOpen = navLinks.classList.toggle("open");
    menuBtn.setAttribute("aria-expanded", String(isOpen));
    menuBtn.setAttribute("aria-label", isOpen ? "Close menu" : "Open menu");
  });
  navLinks.querySelectorAll("a").forEach(link => {
    link.addEventListener("click", () => {
      navLinks.classList.remove("open");
      menuBtn.setAttribute("aria-expanded", "false");
      menuBtn.setAttribute("aria-label", "Open menu");
    });
  });
}

/* ==========================================================
   RECOMMENDATION PAGE: interests -> colleges/programs
   Data comes from catalog.py (embedded as JSON by the template).
   ========================================================== */
const recommendationForm = document.getElementById("recommendationForm");
if (recommendationForm) {
  const resultBox = document.getElementById("recommendationResult");
  const resultList = document.getElementById("recommendationList");
  const exploreUrl = recommendationForm.dataset.exploreUrl || "/explore";
  let catalog = [];
  try {
    catalog = JSON.parse(document.getElementById("catalog-data").textContent);
  } catch (err) {
    console.error("Could not read catalog data", err);
  }
  const byId = Object.fromEntries(catalog.map(c => [c.id, c]));

  recommendationForm.addEventListener("submit", event => {
    event.preventDefault();
    const selected = [...recommendationForm.querySelectorAll('input[name="interest"]:checked')]
      .map(input => input.value);
    resultList.replaceChildren();

    if (selected.length === 0) {
      const message = document.createElement("p");
      message.textContent = "Choose at least one interest to see suggested colleges and programs.";
      resultList.appendChild(message);
    } else {
      selected.forEach(key => {
        const college = byId[key];
        if (!college) return;

        const article = document.createElement("article");
        article.className = "suggestion-result";
        const title = document.createElement("h3");
        title.textContent = college.name;
        const description = document.createElement("p");
        description.textContent = college.recommend;
        const label = document.createElement("strong");
        label.textContent = "Related listings: ";
        const programs = document.createElement("p");
        programs.append(label, document.createTextNode(college.programs.map(p => p.name).join("; ")));
        const link = document.createElement("a");
        link.className = "card-link";
        link.href = `${exploreUrl}#${key}`;
        link.textContent = "View college details →";
        article.append(title, description, programs, link);
        resultList.appendChild(article);
      });
    }
    resultBox.hidden = false;
    resultBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
  });
}

/* ==========================================================
   ASSISTANT PAGE: real chat wired to /api/chat
   Conversations are kept in this browser's localStorage.
   ========================================================== */
const chatMessages = document.getElementById("chatMessages");
if (chatMessages) {
  const chatForm = document.getElementById("assistantForm");
  const chatInput = document.getElementById("assistantInput");
  const sendBtn = document.getElementById("sendBtn");
  const progressFill = document.getElementById("progressFill");
  const progressLabel = document.getElementById("progressLabel");
  const historyPanel = document.getElementById("historyPanel");
  const historyToggle = document.getElementById("historyToggle");
  const historyList = document.getElementById("historyList");
  const historyEmpty = document.getElementById("historyEmpty");
  const newChatBtn = document.getElementById("newChatBtn");

  const STORE_KEY = "lspu_conversations_v1";
  const GREETING = "Hi! I'm the LSPU-LB Knowledge Assistant. What would you like to know about the university, its programs, or the handbook?";
  const steps = [
    "Step 1 of 4 · Getting to know you",
    "Step 2 of 4 · Matching your interests",
    "Step 3 of 4 · Comparing programs",
    "Step 4 of 4 · Next steps"
  ];

  let conversations = [];
  let current = null;
  let isSending = false;

  function loadStore() {
    try { conversations = JSON.parse(localStorage.getItem(STORE_KEY)) || []; }
    catch (e) { conversations = []; }
  }
  function saveStore() {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(conversations.slice(0, 30))); }
    catch (e) { /* storage unavailable: chat still works, just not saved */ }
  }

  const timeNow = () => new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  function newConversation() {
    current = { id: String(Date.now()), title: "New conversation", updated: Date.now(), saved: false, messages: [] };
    current.messages.push({ sender: "ai", text: GREETING, time: timeNow() });
  }

  function fmtDate(ts) {
    const d = new Date(ts);
    return d.toDateString() === new Date().toDateString()
      ? "Today"
      : d.toLocaleDateString([], { month: "short", day: "numeric" });
  }

  function renderHistory() {
    historyList.replaceChildren();
    conversations.forEach(conv => {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "history-item" + (current && conv.id === current.id ? " active" : "");
      if (current && conv.id === current.id) btn.setAttribute("aria-current", "true");
      const strong = document.createElement("strong");
      strong.textContent = conv.title;
      const span = document.createElement("span");
      const count = conv.messages.filter(m => m.sender === "student").length;
      span.textContent = `${fmtDate(conv.updated)} · ${count} question${count === 1 ? "" : "s"}`;
      btn.append(strong, span);
      btn.addEventListener("click", () => {
        if (isSending) return;
        current = conv;
        renderAll();
        historyPanel.classList.remove("open");
      });
      li.appendChild(btn);
      historyList.appendChild(li);
    });
    historyEmpty.hidden = conversations.length > 0;
  }

  function buildRow(msg) {
    const row = document.createElement("div");
    row.className = "msg-row " + msg.sender;
    const avatar = document.createElement("div");
    avatar.className = "msg-avatar";
    avatar.textContent = msg.sender === "ai" ? "AI" : "You";
    const content = document.createElement("div");
    content.className = "msg-content";
    const bubble = document.createElement("div");
    bubble.className = "msg-bubble" + (msg.error ? " is-error" : "");
    bubble.textContent = msg.text;
    const time = document.createElement("div");
    time.className = "msg-time";
    time.textContent = msg.time;
    content.append(bubble, time);
    row.append(avatar, content);
    return row;
  }

  function scrollToBottom() { chatMessages.scrollTop = chatMessages.scrollHeight; }

  function updateProgress() {
    const asked = current.messages.filter(m => m.sender === "student").length;
    const idx = Math.min(asked, steps.length - 1);
    progressFill.style.width = ((idx + 1) / steps.length) * 100 + "%";
    progressLabel.textContent = steps[idx];
  }

  function renderAll() {
    chatMessages.replaceChildren(...current.messages.map(buildRow));
    scrollToBottom();
    updateProgress();
    renderHistory();
  }

  function showTyping() {
    const row = document.createElement("div");
    row.className = "msg-row ai typing-row";
    row.id = "typingRow";
    const avatar = document.createElement("div");
    avatar.className = "msg-avatar";
    avatar.textContent = "AI";
    const bubble = document.createElement("div");
    bubble.className = "msg-bubble";
    bubble.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
    row.append(avatar, bubble);
    chatMessages.appendChild(row);
    scrollToBottom();
  }
  function removeTyping() {
    const row = document.getElementById("typingRow");
    if (row) row.remove();
  }

  function setSending(state) {
    isSending = state;
    sendBtn.disabled = state;
    chatInput.disabled = state;
  }

  function persist() {
    current.updated = Date.now();
    if (!current.saved) {
      current.saved = true;
      conversations.unshift(current);
    }
    saveStore();
  }

  async function ask(question) {
    const userMsg = { sender: "student", text: question, time: timeNow() };
    current.messages.push(userMsg);
    if (current.title === "New conversation") {
      current.title = question.length > 38 ? question.slice(0, 38) + "…" : question;
    }
    chatMessages.appendChild(buildRow(userMsg));
    persist();
    updateProgress();
    renderHistory();
    setSending(true);
    showTyping();

    let reply;
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question })
      });
      const data = await res.json();
      reply = data.error
        ? { sender: "ai", text: data.error, time: timeNow(), error: true }
        : { sender: "ai", text: data.answer, time: timeNow() };
    } catch (err) {
      console.error(err);
      reply = {
        sender: "ai",
        text: "Sorry — I couldn't reach the knowledge base just now. Please try again in a moment.",
        time: timeNow(),
        error: true
      };
    }

    removeTyping();
    current.messages.push(reply);
    chatMessages.appendChild(buildRow(reply));
    scrollToBottom();
    persist();
    renderHistory();
    setSending(false);
    chatInput.focus();
  }

  chatForm.addEventListener("submit", event => {
    event.preventDefault();
    const question = chatInput.value.trim();
    if (!question || isSending) return;
    chatInput.value = "";
    ask(question);
  });

  newChatBtn.addEventListener("click", () => {
    if (isSending) return;
    newConversation();
    renderAll();
    historyPanel.classList.remove("open");
    chatInput.focus();
  });

  if (historyToggle) {
    historyToggle.addEventListener("click", () => historyPanel.classList.toggle("open"));
  }

  loadStore();
  newConversation();
  renderAll();
}
