const sessionId = crypto.randomUUID();
const messagesEl = document.getElementById("chatMessages");
const inputEl    = document.getElementById("userInput");
const sendBtn    = document.getElementById("sendBtn");
const resetBtn   = document.getElementById("resetBtn");
const toggleBtn  = document.getElementById("sidebarToggle");
const overlay    = document.getElementById("sidebarOverlay");
const sidebar    = document.querySelector(".sidebar");

// Sidebar toggle (mobile)
toggleBtn.addEventListener("click", () => {
  sidebar.classList.toggle("open");
  overlay.classList.toggle("open");
});
overlay.addEventListener("click", () => {
  sidebar.classList.remove("open");
  overlay.classList.remove("open");
});

// Quick question buttons
document.querySelectorAll(".quick-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    inputEl.value = btn.dataset.q;
    sidebar.classList.remove("open");
    overlay.classList.remove("open");
    sendMessage();
  });
});

function scrollBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function timeNow() {
  return new Date().toLocaleTimeString("he-IL", { hour: "2-digit", minute: "2-digit" });
}

function renderText(text) {
  return text.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank" rel="noopener">$1</a>');
}

function addMessage(text, role) {
  const welcome = document.querySelector(".welcome-card");
  if (welcome) welcome.remove();

  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "bot" ? "AI" : "אני";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = renderText(text);

  const time = document.createElement("div");
  time.className = "msg-time";
  time.textContent = timeNow();

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  wrapper.appendChild(time);

  // Copy + feedback only for bot messages
  if (role === "bot") {
    const actions = document.createElement("div");
    actions.className = "msg-actions";

    // Copy button
    const copyBtn = document.createElement("button");
    copyBtn.className = "action-btn";
    copyBtn.title = "העתק תשובה";
    copyBtn.textContent = "העתק";
    copyBtn.onclick = () => {
      navigator.clipboard.writeText(bubble.innerText);
      copyBtn.textContent = "הועתק ✓";
      setTimeout(() => copyBtn.textContent = "העתק", 2000);
    };

    // Thumbs up
    const upBtn = document.createElement("button");
    upBtn.className = "action-btn thumb-btn";
    upBtn.title = "תשובה טובה";
    upBtn.textContent = "👍";

    // Thumbs down
    const downBtn = document.createElement("button");
    downBtn.className = "action-btn thumb-btn";
    downBtn.title = "תשובה לא מדויקת";
    downBtn.textContent = "👎";

    [upBtn, downBtn].forEach(btn => {
      btn.onclick = () => {
        fetch("/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId, value: btn.textContent })
        });
        actions.innerHTML = "<span class='feedback-thanks'>תודה על המשוב</span>";
      };
    });

    actions.appendChild(copyBtn);
    actions.appendChild(upBtn);
    actions.appendChild(downBtn);
    wrapper.appendChild(actions);
  }

  messagesEl.appendChild(wrapper);
  scrollBottom();
  return wrapper;
}

function addTyping() {
  const wrapper = document.createElement("div");
  wrapper.className = "message bot typing";
  wrapper.innerHTML = `
    <div class="avatar">AI</div>
    <div class="bubble">
      <span class="dot"></span><span class="dot"></span><span class="dot"></span>
    </div>`;
  messagesEl.appendChild(wrapper);
  scrollBottom();
  return wrapper;
}

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;

  inputEl.value = "";
  sendBtn.disabled = true;
  addMessage(text, "user");

  const typingEl = addTyping();

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: sessionId }),
    });

    const data = await res.json();
    typingEl.remove();
    addMessage(data.reply || "אירעה שגיאה, נסה שוב.", "bot");

  } catch {
    typingEl.remove();
    addMessage("שגיאת תקשורת עם השרת.", "bot");
  }

  sendBtn.disabled = false;
  inputEl.focus();
}

sendBtn.addEventListener("click", sendMessage);
inputEl.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) sendMessage();
});

resetBtn.addEventListener("click", async () => {
  await fetch("/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  messagesEl.innerHTML = `
    <div class="welcome-card">
      <div class="welcome-icon">🎓</div>
      <h2>שיחה חדשה התחילה</h2>
      <p>אנא ציין את פרטיך כדי שאוכל לסייע במדויק.</p>
      <div class="welcome-hint">
        <strong>פרטים נדרשים:</strong>
        <ul>
          <li>תואר ראשון (מה למדת?)</li>
          <li>שנת לימודים (ראשונה / שנייה)</li>
          <li>התמחות מבוקשת</li>
          <li>ממוצע ציונים (אם רלוונטי)</li>
        </ul>
      </div>
    </div>`;
  inputEl.focus();
});
