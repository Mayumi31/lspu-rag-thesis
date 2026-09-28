// ---------------------------------------------------------------------
// LSPU-LB homepage script
//   - mobile nav toggle
//   - "LSPU Knowledge Assistant" chat, wired to /api/chat (server-side
//     this endpoint ONLY runs the Ontology Contextual RAG pipeline —
//     no chunk/vector/graph internals are sent to or rendered in this
//     tab; that visual breakdown lives on the separate retrieval
//     visualizer page, linked from the nav / "See how retrieval works").
// ---------------------------------------------------------------------

// Mobile navigation
const menuBtn = document.getElementById("menuBtn");
const navLinks = document.getElementById("navLinks");

menuBtn.addEventListener("click", () => {
  navLinks.classList.toggle("open");
});

navLinks.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", (e) => {
    // Don't collapse the mobile menu for the visualizer link before the
    // new tab has a chance to open — harmless either way, but tidy.
    navLinks.classList.remove("open");
  });
});

// ---------------------------------------------------------------------
// Chat wiring
// ---------------------------------------------------------------------
const form = document.getElementById("chatForm");
const input = document.getElementById("chatInput");
const sendBtn = document.getElementById("chatSendBtn");
const chatBody = document.getElementById("chatBody");
const onlineStatus = document.getElementById("chatOnlineStatus");

let isSending = false;

function addMessage(text, type) {
  const bubble = document.createElement("div");
  bubble.className = "bubble " + type;
  bubble.textContent = text;
  chatBody.appendChild(bubble);
  chatBody.scrollTop = chatBody.scrollHeight;
  return bubble;
}

// Loading visualization shown in the chat itself while the assistant is
// retrieving + generating — kept deliberately generic (no chunk/score/
// graph details here, that's the visualizer page's job).
function addTypingBubble() {
  const bubble = document.createElement("div");
  bubble.className = "bubble bot typing";
  bubble.innerHTML =
    '<span class="typing-dot"></span>' +
    '<span class="typing-dot"></span>' +
    '<span class="typing-dot"></span>';
  chatBody.appendChild(bubble);
  chatBody.scrollTop = chatBody.scrollHeight;
  return bubble;
}

function setSending(sending) {
  isSending = sending;
  input.disabled = sending;
  sendBtn.disabled = sending;
  sendBtn.textContent = sending ? "…" : "Send";
  document.querySelectorAll(".suggestions button").forEach((b) => {
    b.disabled = sending;
  });
}

async function sendQuestion(question) {
  if (!question || isSending) return;

  addMessage(question, "user");
  setSending(true);
  const typingBubble = addTypingBubble();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();

    typingBubble.remove();

    if (data.error) {
      addMessage(data.error, "bot").classList.add("is-error");
    } else {
      addMessage(data.answer, "bot");
    }
  } catch (err) {
    console.error(err);
    typingBubble.remove();
    addMessage(
      "Sorry — I couldn't reach the knowledge base just now. Please try again in a moment.",
      "bot"
    ).classList.add("is-error");
  } finally {
    setSending(false);
    input.focus();
  }
}

function askDemo(question) {
  input.value = question;
  sendQuestion(question);
  input.value = "";
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  sendQuestion(question);
});

// Let the chip buttons call askDemo() via inline onclick in the markup.
window.askDemo = askDemo;
