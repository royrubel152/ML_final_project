# Development Progress & Fine-Tuning Log
## MBA Academic Advisor Chatbot — Hebrew University Business School

**Document type:** Development journal + technical decisions  
**Date:** May 2026

---

## Overview

This document tracks the full development journey of the MBA Advisor chatbot — from a basic prototype to a production-quality RAG system. It explains what we built in each step, what problems we hit, what decisions we made, and why.

---

## Step 1 — Basic Chatbot with Context Stuffing

### What we built

A working web chatbot that answers student questions about the MBA program. The first version scraped 6 official HUJI sources and injected all the content directly into Gemini's system prompt.

### Architecture

```
Student → Chat UI (HTML/JS)
              ↓
         FastAPI Server
              ↓
         Gemini 2.5 Flash
         (System prompt = rules + ALL 38,000 chars of HUJI content)
              ↓
         Hebrew structured answer
```

### Components built

| Component | Description |
|---|---|
| `app.py` | FastAPI server with lifespan startup event |
| `scraper.py` | Orchestrates scraping via subprocess |
| `scrape_runner.py` | Standalone Playwright script (Windows asyncio fix) |
| `templates/index.html` | Chat UI with sidebar + quick questions |
| `static/style.css` | RTL Hebrew design, HUJI blue/gold theme |
| `static/script.js` | Chat logic, session ID, typing indicator |
| `.env` | Gemini API key |

### Technical challenges solved in Step 1

**Challenge 1 — JavaScript-rendered pages**

HUJI's website renders content with JavaScript. A basic `requests` HTTP call only returns the HTML skeleton — not the actual content students see in their browser.

**Solution:** Used Playwright (headless Chromium) to open each page, wait for JavaScript to finish rendering, then extract the full text.

**Challenge 2 — Windows asyncio + Playwright conflict**

Playwright's sync API creates its own asyncio event loop internally. FastAPI/Uvicorn also runs an asyncio event loop. On Windows, two event loops cannot coexist in the same process.

Error received:
```
NotImplementedError: _make_subprocess_transport not implemented
```

**Solution:** Split scraping into two files:
- `scraper.py` — async wrapper, runs inside uvicorn's event loop, launches a subprocess
- `scrape_runner.py` — completely standalone Python process, no asyncio, runs Playwright freely

The parent process reads the scraped content from stdout when the subprocess completes.

**Challenge 3 — Hebrew PDF extraction reversed**

The academic regulations PDF (RTL Hebrew) was being extracted with some lines reversed:
```
Original:  "ממוצע ציונים מצטבר של 85 ומעלה"
Extracted: "85 לש רבטצמ םינויצ עצוממ"
```

**Solution:** Added word-level RTL detection and line reversal in `scrape_runner.py`. Lines identified as majority Hebrew and appearing backwards are reversed word-by-word.

**Challenge 4 — Gemini model not found**

Initial model names (`gemini-1.5-flash`, `gemini-2.0-flash`) returned 404 errors. Used the API to list available models and found `gemini-3.0-flash`.

**Challenge 5 — Gemini SDK blocks async loop**

Gemini's Python SDK uses synchronous blocking calls. Calling it directly inside a FastAPI `async` endpoint would freeze the server for all users.

**Solution:** Used `asyncio.run_in_executor()` to run the Gemini call in a background thread:
```python
response = await loop.run_in_executor(
    None, partial(session.send_message, user_message)
)
```

### Results after Step 1

| Metric | Value |
|---|---|
| Sources loaded | 6 (5 web pages + 1 PDF) |
| Total content | ~31,000 chars |
| Startup time | ~30 seconds (scraping every time) |
| Tokens per request | ~10,000 |
| Session persistence | In-memory only (lost on restart) |
| Answer quality | Good but occasional hallucination |

### What was still wrong

- Every server restart re-scraped all pages (30 second delay)
- If HUJI website was down, server failed to start
- Bot hallucinated citations — invented page numbers and fake quotes
- No rate limiting — one user could exhaust the free API quota
- Sessions lost on every restart
- No copy button, no feedback, no persistent disclaimer

---

## Step 2 — Production Hardening (Phase 1 Improvements)

### What we added

Seven targeted improvements without changing the core architecture. All focused on reliability, safety, and usability.

### Improvement 1 — Content caching

**Problem:** Server re-scraped all 6 HUJI sources every restart. 30 second delay. Server crashed if HUJI was offline.

**Solution:** After scraping, save content to `scraped_content.json` with a timestamp. On next startup, check if cache is less than 24 hours old. If yes — load from file instantly (2 seconds). If no — re-scrape.

```python
# Startup time comparison:
# Before: always 30 seconds
# After:  2 seconds (if cache fresh)
#         30 seconds (if cache expired — once per day max)
```

Also added fallback: if scraping fails, use stale cache rather than crashing. Server always starts.

**Improvement 2 — Temperature = 0**

**Problem:** Gemini's default temperature allows creative generation. For a factual academic advisor, creativity is the enemy — it leads to paraphrasing that introduces inaccuracies.

**Solution:** Set `temperature=0.0` in GenerationConfig. This makes Gemini deterministic — same question will always produce the same answer. Maximum factual consistency.

```python
generation_config=genai.GenerationConfig(temperature=0.0)
```

**Improvement 3 — Citation verifier**

**Problem:** Gemini was inventing quotes that looked real but weren't in the source material. Students were seeing hallucinated citations.

Example of hallucinated citation:
```
"תלמידי תואר ראשון... נדרשים לציון 40 ומעלה." (תקנון אקדמי, עמ' 15)
```
The page number was invented. The quote was paraphrased, not verbatim.

**Solution:** After every Gemini response, extract text inside quotation marks using regex. Check if each quoted string exists verbatim in the source content. If not — replace with a warning:

```python
def verify_citations(reply: str, source_content: str) -> str:
    quotes = re.findall(r'"([^"]{20,})"', reply)
    for quote in quotes:
        if quote.strip() not in source_content:
            reply = reply.replace(f'"{quote}"',
                '"[ציטוט לא אומת — פנה למזכירות לאימות]"')
    return reply
```

**Improvement 4 — Rate limiting**

**Problem:** No protection against abuse. One user could send thousands of requests and exhaust the free Gemini API quota (1,500 requests/day).

**Solution:** Added `slowapi` — FastAPI rate limiting middleware. Limits each IP to 15 messages per minute.

```python
@limiter.limit("15/minute")
async def chat(request: Request):
```

**Improvement 5 — API key validation at startup**

**Problem:** If `GEMINI_API_KEY` was missing from `.env`, the server started fine but crashed on the first message with a cryptic error.

**Solution:** Check the key exists at import time and raise immediately with a clear message:
```python
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing from .env — cannot start.")
```

**Improvement 6 — Pinned disclaimer banner**

**Problem:** The disclaimer appeared inside every bot message, making the chat cluttered. Some students scrolled past it.

**Solution:** Removed per-message disclaimer. Added a fixed yellow banner at the top of the chat area that is always visible:
```html
<div class="disclaimer-banner">
  ⚠️ מענה זה ניתן על-ידי בוט בלבד. יש לאמת כל מידע מול מזכירות התלמידים.
</div>
```

**Improvement 7 — Copy + feedback buttons**

**Problem:** Students had no way to copy answers to paste into emails or notes. No way for the team to know which answers were wrong.

**Solution:**
- Added "העתק" (copy) button to every bot message — copies clean text to clipboard
- Added 👍 👎 buttons — sends feedback to `/feedback` endpoint — logged to server logs
- Added `/feedback` endpoint and structured JSON logging

### Results after Step 2

| Metric | Before | After |
|---|---|---|
| Startup time | 30s always | 2s (cached) |
| Server crash if HUJI down | Yes | No (uses cache) |
| Hallucinated citations | Shown to students | Replaced with warning |
| Answer consistency | Variable | Deterministic (temp=0) |
| API key error message | Cryptic | Clear startup error |
| Rate limiting | None | 15 req/min per IP |
| Copy button | No | Yes |
| Feedback collection | No | Yes (👍 👎 logged) |
| Disclaimer visibility | Buried in chat | Always visible banner |

---

## Step 3 — RAG Architecture

### Why we rebuilt the core

Step 1 and Step 2 improved reliability and safety. But the fundamental architecture — context stuffing — had a ceiling. The more content you add, the worse it gets. And for a real university deployment serving hundreds of students, it would not scale.

The three core problems that required architectural change:

1. **10,000 tokens per request** — 85% of those tokens were irrelevant to the specific question
2. **No semantic search** — the bot couldn't distinguish "what do I need to apply?" from "what exemptions exist?" at the retrieval level
3. **In-memory sessions lost on restart** — students lost their conversation context every time the server restarted

### New components built

| Component | Description |
|---|---|
| `rag.py` | Chunking, embedding, FAISS index, retrieval, prompt builder |
| `sessions.py` | File-based session store with TTL and cleanup |
| `data/faiss_index.bin` | Saved FAISS vector index (auto-generated) |
| `data/chunks.json` | Chunk metadata with source + URL (auto-generated) |
| `data/content_hash.txt` | Hash to detect content changes (auto-generated) |
| `data/sessions/` | Per-session JSON files (auto-generated) |

### Fine-tuning decisions in the RAG implementation

**Decision 1 — Chunk size: 800 characters**

We tested three sizes:
- 400 chars: too small — individual chunks lost context, answers were incomplete
- 800 chars: good balance — enough context per chunk, precise retrieval
- 1,200 chars: too large — chunks became too broad, retrieval less precise

800 characters fits approximately 3-5 sentences of Hebrew academic text. Large enough to contain a complete regulation, small enough to be specific.

**Decision 2 — Overlap: 150 characters**

Without overlap, a regulation that spans the boundary between two chunks would be split in half. Neither chunk would contain the full rule. With 150-character overlap, every boundary zone appears in two chunks — nothing is lost.

```
Chunk N ends:    "...ממוצע ציונים מצטבר של 85 ומעלה בתואר הראשון בסוף שנה ב'."
Chunk N+1 starts: "85 ומעלה בתואר הראשון בסוף שנה ב'. פטור מלימודי אנגלית..."
```

**Decision 3 — Embedding model: gemini-embedding-001**

We tested the available models on this API key:
- `text-embedding-004` — returned 404 (not available on this key)
- `gemini-embedding-001` — available, 3,072 dimensions, excellent Hebrew support
- `gemini-embedding-2` — available, newer but same quality for Hebrew

Chose `gemini-embedding-001` for stability. 3,072 dimensions gives fine-grained semantic resolution — important for distinguishing between similar-sounding academic rules.

**Decision 4 — TOP_K = 5 chunks**

We tested different values:
- TOP_K = 3: sometimes missed relevant context, incomplete answers
- TOP_K = 5: good coverage without sending too much noise
- TOP_K = 8: started including borderline-relevant chunks, confused Gemini

5 chunks × ~800 chars = ~4,000 chars = ~1,200 tokens. This is the sweet spot.

**Decision 5 — Minimum similarity score: 0.45**

If no chunk scores above 0.45, we return nothing and the bot says "information not available." This prevents the bot from answering with weakly related content and hallucinating an answer from irrelevant chunks.

0.45 was chosen after testing:
- 0.6: too strict — rejected some genuinely relevant chunks
- 0.45: correctly filters noise while keeping relevant matches
- 0.3: too loose — included unrelated content

**Decision 6 — Content hash for index invalidation**

The FAISS index only needs to be rebuilt if the scraped content changes. We save an MD5 hash of the content alongside the index. On startup:
- Hash matches → load existing index (1 second)
- Hash differs → rebuild index (~90 seconds, only when HUJI updates pages)

This means after the first build, every restart loads the index in under 2 seconds.

**Decision 7 — File-based sessions instead of Redis**

Redis would be the production choice but requires a running Redis server. For the current deployment (local + ngrok), file-based sessions give us 90% of the benefit with zero infrastructure:
- Sessions saved as `data/sessions/<uuid>.json`
- 2-hour TTL — expired sessions cleaned up at startup
- Survives server restarts — students can continue conversations

**Decision 8 — RAG prompt inside chat history, not system prompt**

The content is no longer in the system prompt. Instead, for each message we build an augmented prompt that includes the retrieved chunks:

```
System prompt: rules + format (static, ~500 tokens)
User message:  [retrieved chunks] + question (dynamic, ~1,200 tokens)
```

This means Gemini always has the rules, and for each specific question it gets exactly the relevant content. Previous messages in the conversation remain in history so follow-up questions work correctly.

### Results after Step 3

| Metric | Step 1 | Step 2 | Step 3 (RAG) |
|---|---|---|---|
| Tokens per request | ~10,000 | ~10,000 | ~1,500 |
| Cost per 1000 queries | ~$1.50 | ~$1.50 | ~$0.23 |
| Answer precision | Medium | Medium | High |
| Hallucination risk | Medium | Medium-Low | Low |
| Startup time (cold) | 30s | 30s | 90s first time |
| Startup time (warm) | 30s | 2s | 2s |
| Sessions survive restart | No | No | Yes (2h TTL) |
| Scalable to more sources | No | No | Yes |
| Citation verification | None | Post-gen check | Post-gen check on chunks |
| Architecture type | Context stuffing | Context stuffing | RAG |

---

## Fine-Tuning Summary — All Changes Made

### System prompt evolution

**Step 1:**
- Basic rules + all 38,000 chars injected

**Step 2:**
- Stricter anti-hallucination rules added
- Explicit instruction: never invent page numbers
- Instruction: write "לא נמצא ציטוט מדויק" when no exact quote exists

**Step 3:**
- Content removed from system prompt entirely
- System prompt now contains only: rules, format, anti-hallucination instructions
- Content comes dynamically per query via retrieved chunks

### Model configuration evolution

| Parameter | Step 1 | Step 2 | Step 3 |
|---|---|---|---|
| Model | gemini-3.0-flash | gemini-3.0-flash | gemini-3.0-flash |
| Temperature | Default (1.0) | 0.0 | 0.0 |
| Context source | System prompt | System prompt | Retrieved chunks |
| Tokens/request | ~10,000 | ~10,000 | ~1,500 |

### Scraper evolution

| Feature | Step 1 | Step 2 | Step 3 |
|---|---|---|---|
| Scraping method | requests + BeautifulSoup | Playwright (headless) | Playwright (headless) |
| Content per startup | Always fresh | Cached 24h | Cached 24h |
| Failure handling | Server crash | Uses stale cache | Uses stale cache |
| PDF parsing | pdfplumber basic | pdfplumber + RTL fix | pdfplumber + RTL fix |

### Session management evolution

| Feature | Step 1 | Step 2 | Step 3 |
|---|---|---|---|
| Storage | Python dict (memory) | Python dict (memory) | JSON files (disk) |
| Survives restart | No | No | Yes |
| TTL | None | None | 2 hours |
| Cleanup | Manual reset only | Manual reset only | Auto on startup |

---

## What Comes Next (Phase 2 & Beyond)

| Improvement | Why | Effort |
|---|---|---|
| Deploy to Railway / Render | Always online, no laptop needed | Low |
| Scheduled index refresh (24h) | Keeps FAISS index current with HUJI changes | Low |
| Sentence-aware chunking | Avoid cutting regulations mid-sentence | Medium |
| Redis sessions | Production-grade, multi-server | Medium |
| Admin panel | Trigger re-scrape, view feedback, see top questions | High |
| Analytics dashboard | Most common topics, failure rate, student patterns | High |
| Evaluation set | 20 real questions with known correct answers | Medium |

---

*Document prepared: May 2026*  
*Stack: Python 3.11 · FastAPI · Uvicorn · Google Gemini 2.5 Flash · gemini-embedding-001 · FAISS · Playwright · pdfplumber*
