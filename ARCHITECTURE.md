# Architecture Deep Dive
## MBA Academic Advisor Chatbot — Hebrew University Business School

---

## Table of Contents
1. [High-Level Overview](#1-high-level-overview)
2. [Frontend Layer](#2-frontend-layer)
3. [Backend Layer](#3-backend-layer-fastapi)
4. [Scraping Layer](#4-scraping-layer)
5. [AI Layer](#5-ai-layer-google-gemini)
6. [Session Management](#6-session-management)
7. [Data Flow — Step by Step](#7-data-flow--step-by-step)
8. [Key Design Decisions](#8-key-design-decisions)
9. [Limitations](#9-limitations)

---

## 1. High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        STARTUP PHASE                            │
│                                                                 │
│  scrape_runner.py                                               │
│  ┌──────────────┐    fetches     ┌─────────────────────────┐   │
│  │  Playwright  │ ─────────────▶ │  6 HUJI Official Pages  │   │
│  │  (Chromium)  │                │  + Academic PDF         │   │
│  └──────────────┘                └─────────────────────────┘   │
│         │                                                       │
│         │  38,000 chars of text                                 │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │ System Prompt│  = strict rules + all source content         │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       REQUEST PHASE                             │
│                                                                 │
│  Student Browser                                                │
│  ┌──────────────┐   HTTP POST    ┌──────────────────────────┐  │
│  │  Chat UI     │ ─────────────▶ │  FastAPI Server (app.py) │  │
│  │  (HTML/JS)   │ ◀───────────── │  Python + Uvicorn        │  │
│  └──────────────┘   JSON reply   └──────────────────────────┘  │
│                                           │                     │
│                                    Gemini API call              │
│                                           ▼                     │
│                                  ┌─────────────────┐           │
│                                  │ Google Gemini   │           │
│                                  │ 2.5 Flash       │           │
│                                  └─────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Frontend Layer

**Files:** `templates/index.html`, `static/style.css`, `static/script.js`

### What it does
The frontend is a single-page chat interface served by the FastAPI backend. It handles:
- Rendering the chat UI (messages, input box, sidebar)
- Sending user messages to the server via `fetch()` API
- Displaying bot responses
- Managing session ID (unique per browser tab)

### Technology choice: Vanilla HTML/CSS/JS
No React, Vue, or Angular. Reasons:
- The UI is simple — a chat box does not need a framework
- Loads instantly, no build step, no dependencies
- Easy to maintain and modify

### Session ID
Each browser tab generates a unique `sessionId` using `crypto.randomUUID()` on page load:
```javascript
const sessionId = crypto.randomUUID();
```
This ID is sent with every message so the server knows which conversation history belongs to which user.

### Communication Protocol
Every message is sent as a `POST /chat` with JSON body:
```json
{
  "message": "מה תנאי הקבלה?",
  "session_id": "uuid-here"
}
```
The server returns:
```json
{
  "reply": "מבוא: ..."
}
```

### Sidebar — Quick Links
The sidebar links directly to all 6 official HUJI sources so students can verify answers themselves. This is intentional — the bot is a helper, not the final authority.

---

## 3. Backend Layer (FastAPI)

**File:** `app.py`

### What it does
The backend is a Python web server built with FastAPI. It:
1. Serves the HTML chat interface
2. Manages the startup scraping process
3. Routes chat messages to Gemini
4. Maintains conversation sessions in memory

### FastAPI + Uvicorn
- **FastAPI** is a modern Python web framework — fast, async-native, minimal boilerplate
- **Uvicorn** is the ASGI server that runs FastAPI — handles HTTP connections efficiently
- Together they handle multiple concurrent users without blocking

### Lifespan (Startup Event)
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    source_content = await load_all_sources()  # scrape all HUJI pages
    model_instance["model"] = genai.GenerativeModel(
        model_name="gemini-3.0-flash",
        system_instruction=SYSTEM_BASE + source_content,
    )
    yield
```
This runs **once** when the server starts. It:
1. Calls the scraper
2. Builds the full system prompt
3. Initializes the Gemini model with that prompt
4. Then the server becomes ready to handle requests

### Chat Endpoint
```
POST /chat
```
1. Reads the user message and session ID
2. Creates a new chat session if first message from this user
3. Calls Gemini's `send_message()` in a thread executor (avoids blocking the async loop)
4. Returns the response as JSON

### Why Thread Executor?
Gemini's Python SDK is synchronous (blocking). Running it directly inside an `async` function would freeze the server for all other users while waiting for Gemini to respond. Using `run_in_executor` moves it to a background thread:
```python
response = await loop.run_in_executor(
    None, partial(session.send_message, user_message)
)
```

---

## 4. Scraping Layer

**Files:** `scraper.py`, `scrape_runner.py`

### The Problem
The HUJI website uses JavaScript to render content. A basic HTTP request (`requests` library) only gets the HTML skeleton — not the actual content that JavaScript loads dynamically.

### Solution: Playwright (Headless Browser)
Playwright opens a real Chromium browser in the background (headless = no visible window), navigates to each page, waits for JavaScript to finish loading, then extracts the full rendered text.

### Why Two Files?
This is the most important architectural decision in the scraping layer:

**The Problem on Windows:**
Playwright's sync API internally creates its own asyncio event loop. But FastAPI/Uvicorn already runs an asyncio event loop. Two event loops cannot run in the same thread — it causes a `NotImplementedError` on Windows.

**The Solution:**
Split into two files:
- `scraper.py` — called by FastAPI, runs in async context, uses `run_in_executor` to launch a subprocess
- `scrape_runner.py` — a completely standalone Python script with no asyncio, runs in its own process

```python
# scraper.py
def _run_subprocess() -> str:
    result = subprocess.run(
        [sys.executable, "scrape_runner.py"],
        capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout
```

`scrape_runner.py` runs as a fresh Python process, uses sync Playwright freely, and outputs the scraped content to stdout. The parent process reads it back as a string.

### PDF Parsing
The academic regulations PDF is downloaded with `requests` and parsed with `pdfplumber`. Hebrew RTL PDFs sometimes extract text in reversed order — a word-reversal fix is applied to lines detected as backwards Hebrew.

### Content Limits
Each source is capped at 12,000 characters to avoid hitting Gemini's context window limits. Total: ~38,000 chars across all 6 sources.

---

## 5. AI Layer (Google Gemini)

**Model:** `gemini-3.0-flash`

### How Gemini Receives Information
Gemini does not browse the internet. It only knows what we tell it. We use two mechanisms:

**1. System Prompt (permanent context)**
Set once when the model is initialized. Contains:
- The bot's role and rules
- Anti-hallucination instructions
- The mandatory response format
- All 38,000 chars of HUJI source content

**2. Chat History (per session)**
Each `send_message()` call includes the full conversation history of that session, so the bot can answer follow-up questions in context.

### Context Stuffing Pattern
This architecture uses a pattern called **"context stuffing"** (also called RAG-lite):

```
System Prompt = Rules + ALL source documents
```

Instead of a vector database that retrieves relevant chunks, we simply inject everything into every request. This works because:
- The total content (~38,000 chars / ~10,000 tokens) fits in Gemini's context window
- It's simpler — no embedding model, no database, no retrieval logic
- Gemini 2.5 Flash has a 1M token context window — we're using barely 1% of it

### Anti-Hallucination Strategy
The system prompt contains explicit instructions:
```
- Copy ONLY words that appear verbatim in the official content below
- If you cannot find the exact quote — write: "No exact quote found"
- Never invent page numbers
- It is better to say "I don't know" than to give unverified information
```
LLMs can still hallucinate despite these instructions, which is why every answer includes a disclaimer directing students to verify with the secretariat.

### Response Format Enforcement
The system prompt mandates a strict Hebrew response structure:
```
מבוא → פרטי סטודנט → תשובה → ציטוט → קישור → כתב ויתור
```
This ensures consistent, professional responses every time.

---

## 6. Session Management

Sessions are stored in a Python dictionary in memory:
```python
chat_sessions = {}  # { session_id: gemini_chat_object }
```

Each Gemini `ChatSession` object maintains its own message history. When a student sends a follow-up question, Gemini sees the full prior conversation and can answer in context.

### Limitations of In-Memory Sessions
- **Sessions are lost on server restart** — if you restart uvicorn, all conversations reset
- **No persistence** — conversations are not saved to a database
- **Single server only** — cannot scale to multiple server instances

For a production system, sessions would be stored in Redis or a database.

---

## 7. Data Flow — Step by Step

### Startup (runs once)
```
1. uvicorn starts app.py
2. lifespan() is called
3. scraper.py calls _run_subprocess()
4. subprocess.run() launches scrape_runner.py as a new process
5. scrape_runner.py opens Chromium (headless)
6. Chromium visits each of 6 HUJI URLs
7. Waits for JavaScript to load (networkidle / domcontentloaded)
8. Removes noise elements (nav, footer, etc.)
9. Extracts full body text
10. PDF is downloaded and parsed separately
11. All content printed to stdout
12. scraper.py reads stdout → 38,000 char string
13. SYSTEM_BASE + source_content = full system prompt
14. Gemini model initialized with system prompt
15. Server ready → "Application startup complete"
```

### Per Request
```
1. Student types message in browser
2. script.js sends POST /chat with { message, session_id }
3. FastAPI /chat endpoint receives request
4. If new session: create new Gemini ChatSession
5. session.send_message(user_message) in thread executor
6. Gemini receives: system prompt + chat history + new message
7. Gemini searches through 38,000 chars of HUJI content
8. Generates structured Hebrew response
9. FastAPI returns { reply: "..." }
10. script.js displays message in chat bubble
```

---

## 8. Key Design Decisions

| Decision | What we chose | Why |
|---|---|---|
| AI Model | Google Gemini 2.5 Flash | Best Hebrew support, large context window, fast |
| Scraping | Playwright headless browser | HUJI pages use JavaScript — basic requests miss content |
| Subprocess isolation | scrape_runner.py as separate process | Windows asyncio + Playwright conflict |
| Content strategy | Context stuffing (inject all docs) | Simple, no vector DB needed at this scale |
| No framework (frontend) | Vanilla HTML/JS | Simplicity, instant load, no build step |
| In-memory sessions | Python dict | Simple for prototype, sufficient for small scale |
| Async threading | run_in_executor | Gemini SDK is sync — prevents blocking the server |

---

## 9. Limitations

| Limitation | Impact | Solution if scaling |
|---|---|---|
| Content capped at 12,000 chars/source | May miss some content | Increase cap or use chunked RAG |
| PDF RTL extraction | Some text may be reversed | Pre-process PDF manually or use better parser |
| In-memory sessions | Lost on restart | Use Redis or database |
| Content loaded at startup only | Stale if HUJI updates | Add scheduled refresh (every 24h) |
| LLM can hallucinate | Wrong citations | Add citation verification layer |
| Single computer | Goes offline if computer sleeps | Deploy to cloud |
| Playwright subprocess | Slow startup (~30s) | Cache content to file between restarts |

---

*Document prepared: May 2026*
*Stack: Python 3.11 · FastAPI · Uvicorn · Google Gemini 2.5 Flash · Playwright · pdfplumber*
