const sessionId = crypto.randomUUID();
const messagesEl = document.getElementById("chatMessages");
const inputEl    = document.getElementById("userInput");
const sendBtn    = document.getElementById("sendBtn");
const resetBtn   = document.getElementById("resetBtn");
const toggleBtn  = document.getElementById("sidebarToggle");
const overlay    = document.getElementById("sidebarOverlay");
const sidebar    = document.querySelector(".sidebar");

// ── Student profile (built by onboarding wizard) ──────────────────
let studentProfile = {};
let onboardStep = 1;
const MAX_STEPS = 5;

function effectiveTotal() {
  // before the dual question is answered, assume 4 steps
  return studentProfile.dual === "true" ? 5 : 4;
}

function buildProfileContext() {
  if (!studentProfile.role) return "";
  const parts = [studentProfile.role];
  if (studentProfile.year)  parts.push(studentProfile.year);
  if (studentProfile.spec)  parts.push("התמחות ראשית: " + studentProfile.spec);
  if (studentProfile.spec2) parts.push("התמחות משנית: " + studentProfile.spec2);
  return "[פרופיל סטודנט: " + parts.join(" | ") + "]";
}

// Onboarding: handle option click
document.addEventListener("click", e => {
  const btn = e.target.closest(".onboard-btn");
  if (!btn) return;
  const key = btn.dataset.key;
  const val = btn.dataset.val;
  studentProfile[key] = val;

  // Highlight chosen
  btn.closest(".onboard-options").querySelectorAll(".onboard-btn")
    .forEach(b => b.classList.remove("chosen"));
  btn.classList.add("chosen");

  setTimeout(() => advanceOnboarding(), 300);
});

function advanceOnboarding() {
  document.getElementById("step" + onboardStep).style.display = "none";
  onboardStep++;

  // Skip minor-spec step if user chose single specialization
  if (onboardStep === 5 && studentProfile.dual !== "true") {
    document.getElementById("onboardBar").style.width = "100%";
    document.getElementById("onboardLabel").textContent = "סיום!";
    finishOnboarding();
    return;
  }

  const total = effectiveTotal();
  const pct = Math.round((onboardStep - 1) / total * 100);
  document.getElementById("onboardBar").style.width = pct + "%";

  if (onboardStep <= MAX_STEPS) {
    document.getElementById("onboardLabel").textContent = "שלב " + onboardStep + " מתוך " + total;
    document.getElementById("step" + onboardStep).style.display = "block";
  } else {
    document.getElementById("onboardLabel").textContent = "סיום!";
    finishOnboarding();
  }
}

function finishOnboarding() {
  const card = document.getElementById("onboardingCard");
  card.classList.add("onboard-done");
  setTimeout(() => {
    card.remove();
    const spec = studentProfile.spec || "";
    const role = studentProfile.role || "סטודנט";
    const greeting = `שלום! אני כאן לעזור לך.
**פרופיל שלך:** ${buildProfileContext().replace(/[\[\]]/g, "")}

איך אוכל לעזור לך היום?`;
    addMessage(greeting, "bot");
    inputEl.focus();
  }, 400);
}

// ── Schedule builder ──────────────────────────────────────────────
const scheduleData = { "1a": [], "1b": [], "2a": [], "2b": [] };

let coursesList = [];

async function fetchCoursesList() {
  if (coursesList.length > 0) return;
  try {
    const res = await fetch("/courses");
    const data = await res.json();
    coursesList = data.courses || [];
  } catch {}
}

function openScheduleModal() {
  document.getElementById("scheduleModal").style.display = "flex";
  sidebar.classList.remove("open");
  overlay.classList.remove("open");
  renderAllColumns();
  fetchCoursesList();
}

function closeScheduleModal() {
  document.getElementById("scheduleModal").style.display = "none";
}

document.getElementById("scheduleModal").addEventListener("click", e => {
  if (e.target.id === "scheduleModal") closeScheduleModal();
});

document.addEventListener("keydown", e => {
  if (e.key === "Escape") closeScheduleModal();
});

function renderAllColumns() {
  ["1a", "1b", "2a", "2b"].forEach(renderColumn);
}

function renderColumn(cellId) {
  const list = document.getElementById("list-" + cellId);
  if (!list) return;
  list.innerHTML = scheduleData[cellId].map(c =>
    `<div class="sched-course-card">
      <span class="sched-course-name">${c}</span>
      <button class="rm-course" onclick="removeCourse('${cellId}',${JSON.stringify(c)})">✕</button>
    </div>`
  ).join("");
}

function openInlineAdd(cellId) {
  document.querySelectorAll(".sched-input-row").forEach(r => r.remove());
  const col = document.getElementById("col-" + cellId);
  const addBtn = col.querySelector(".sched-add-btn");

  const row = document.createElement("div");
  row.className = "sched-input-row";
  row.innerHTML = `
    <div class="ac-wrap">
      <input id="mi_${cellId}" type="text" placeholder="הקלד שם קורס..." autocomplete="off" />
      <div class="ac-list" id="ac_${cellId}" style="display:none"></div>
    </div>
    <button onclick="saveModalCourse('${cellId}')">הוסף</button>`;
  col.insertBefore(row, addBtn);

  const inp = document.getElementById("mi_" + cellId);
  const acList = document.getElementById("ac_" + cellId);

  inp.focus();

  inp.addEventListener("input", () => {
    const val = inp.value.trim();
    if (!val) { acList.style.display = "none"; acList.innerHTML = ""; return; }
    const matches = coursesList.filter(c => c.includes(val)).slice(0, 8);
    if (!matches.length) { acList.style.display = "none"; return; }
    acList.innerHTML = matches.map(c =>
      `<div class="ac-item" data-name="${c.replace(/"/g, '&quot;')}">${c}</div>`
    ).join("");
    acList.style.display = "block";

    acList.querySelectorAll(".ac-item").forEach(item => {
      item.addEventListener("mousedown", e => {
        e.preventDefault();
        inp.value = item.dataset.name;
        acList.style.display = "none";
      });
    });
  });

  inp.addEventListener("keydown", e => {
    if (e.key === "Enter") { e.preventDefault(); saveModalCourse(cellId); }
    if (e.key === "Escape") row.remove();
  });

  inp.addEventListener("blur", () => setTimeout(() => { acList.style.display = "none"; }, 150));
}

function saveModalCourse(cellId) {
  const inp = document.getElementById("mi_" + cellId);
  if (!inp) return;
  const name = inp.value.trim();
  if (name) scheduleData[cellId].push(name);
  inp.closest(".sched-input-row").remove();
  renderColumn(cellId);
}

function removeCourse(cellId, name) {
  scheduleData[cellId] = scheduleData[cellId].filter(c => c !== name);
  renderColumn(cellId);
}

document.getElementById("clearPlanBtn").addEventListener("click", () => {
  if (!confirm("למחוק את כל הקורסים?")) return;
  Object.keys(scheduleData).forEach(k => { scheduleData[k] = []; renderColumn(k); });
});

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
  // Render markdown then linkify bare URLs not already wrapped in <a>
  let html = marked.parse(text);
  html = html.replace(/(?<!href=")(?<!">)(https?:\/\/[^\s<"]+)/g,
    '<a href="$1" target="_blank" rel="noopener">$1</a>');
  return html;
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
      body: JSON.stringify({ message: buildProfileContext() + " " + text, session_id: sessionId }),
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
