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
  navLinks.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      navLinks.classList.remove("open");
      menuBtn.setAttribute("aria-expanded", "false");
      menuBtn.setAttribute("aria-label", "Open menu");
    });
  });
}

/* ==========================================================
   ASSISTANT PAGE: real chat wired to /api/chat
   Conversations are kept in this browser's localStorage.
   No step indicator / progress bar — just the conversation.
   ========================================================== */
const chatMessages = document.getElementById("chatMessages");
if (chatMessages) {
  const chatForm = document.getElementById("assistantForm");
  const chatInput = document.getElementById("assistantInput");
  const sendBtn = document.getElementById("sendBtn");
  const historyPanel = document.getElementById("historyPanel");
  const historyToggle = document.getElementById("historyToggle");
  const historyList = document.getElementById("historyList");
  const historyEmpty = document.getElementById("historyEmpty");
  const newChatBtn = document.getElementById("newChatBtn");

  const STORE_KEY = "lspu_conversations_v1";
  const GREETING =
    "Hi! I'm the LSPU-LB Knowledge Assistant. What would you like to know about the university, its programs, or the handbook?";

  let conversations = [];
  let current = null;
  let isSending = false;

  function loadStore() {
    try {
      conversations = JSON.parse(localStorage.getItem(STORE_KEY)) || [];
    } catch (e) {
      conversations = [];
    }
  }
  function saveStore() {
    try {
      localStorage.setItem(
        STORE_KEY,
        JSON.stringify(conversations.slice(0, 30)),
      );
    } catch (e) {
      /* storage unavailable: chat still works, just not saved */
    }
  }

  const timeNow = () =>
    new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  function newConversation() {
    current = {
      id: String(Date.now()),
      title: "New conversation",
      updated: Date.now(),
      saved: false,
      messages: [],
    };
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
    conversations.forEach((conv) => {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className =
        "history-item" + (current && conv.id === current.id ? " active" : "");
      if (current && conv.id === current.id)
        btn.setAttribute("aria-current", "true");
      const strong = document.createElement("strong");
      strong.textContent = conv.title;
      const span = document.createElement("span");
      const count = conv.messages.filter((m) => m.sender === "student").length;
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

    const avatar =
      msg.sender === "ai"
        ? document.createElement("img")
        : document.createElement("div");
    avatar.className = "msg-avatar";
    if (msg.sender === "ai") {
      avatar.src = "/assets/images/lspu-chatbot-avatar.png";
      avatar.alt = "";
      avatar.onerror = () => {
        avatar.replaceWith(
          Object.assign(document.createElement("div"), {
            className: "msg-avatar",
            textContent: "AI",
          }),
        );
      };
    } else {
      avatar.textContent = "You";
    }

    const content = document.createElement("div");
    content.className = "msg-content";
    const bubble = document.createElement("div");
    bubble.className = "msg-bubble" + (msg.error ? " is-error" : "");
    ChatUI.format(bubble, msg.text);
    if (msg.sender === "ai" && !msg.error && msg.sources) bubble.append(ChatUI.sources(msg.sources));
    const time = document.createElement("div");
    time.className = "msg-time";
    time.textContent = msg.time;
    content.append(bubble, time);
    row.append(avatar, content);
    return row;
  }

  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function renderAll() {
    chatMessages.replaceChildren(...current.messages.map(buildRow));
    scrollToBottom();
    renderHistory();
  }

  function showTyping() {
    const row = document.createElement("div");
    row.className = "msg-row ai typing-row";
    row.id = "typingRow";
    const avatar = document.createElement("img");
    avatar.className = "msg-avatar";
    avatar.src = "/assets/images/lspu-chatbot-avatar.png";
    avatar.alt = "";
    avatar.onerror = () => {
      avatar.replaceWith(
        Object.assign(document.createElement("div"), {
          className: "msg-avatar",
          textContent: "AI",
        }),
      );
    };
    const bubble = document.createElement("div");
    bubble.className = "msg-bubble";
    bubble.innerHTML =
      '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
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
      current.title =
        question.length > 38 ? question.slice(0, 38) + "…" : question;
    }
    chatMessages.appendChild(buildRow(userMsg));
    persist();
    renderHistory();
    setSending(true);
    showTyping();

    let reply;
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Request failed");
      reply = data.error
        ? { sender: "ai", text: data.error, time: timeNow(), error: true }
        : { sender: "ai", text: data.answer, sources: data.sources || [], time: timeNow() };
    } catch (err) {
      console.error(err);
      reply = {
        sender: "ai",
        text: "Sorry — I couldn't reach the knowledge base just now. Please try again in a moment.",
        time: timeNow(),
        error: true,
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

  chatForm.addEventListener("submit", (event) => {
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
    historyToggle.addEventListener("click", () =>
      historyPanel.classList.toggle("open"),
    );
  }

  loadStore();
  newConversation();
  renderAll();
  chatInput.value = new URLSearchParams(location.search).get("q") || "";
}
