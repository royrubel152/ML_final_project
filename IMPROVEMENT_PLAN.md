# Architecture Review & Improvement Plan
## MBA Academic Advisor Chatbot — Hebrew University Business School

**Reviewed by:** Senior AI Systems Architect  
**Date:** May 2026  
**Current stack:** Python · FastAPI · Gemini 2.5 Flash · Playwright · pdfplumber · Vanilla JS

---

## 1. Current Architecture Assessment

### What is good

| Strength | Why it matters |
|---|---|
| Context stuffing at 38K chars | Perfectly sized for Gemini's 1M token window — no retrieval needed yet |
| Subprocess isolation for Playwright | Correct and pragmatic fix for Windows asyncio conflict |
| No framework frontend | Fast, zero dependencies, easy to hand off |
| Lifespan startup event | Clean — scraping happens once, model ready before first request |
| run_in_executor for Gemini SDK | Correct async pattern, prevents server blocking |
| Session ID per browser tab | Stateless server, correct design |
| Anti-hallucination in system prompt | Right instinct, even if not perfect |

### Main risks and weak points

| Risk | Severity | Explanation |
|---|---|---|
| In-memory sessions | HIGH | Server restart = all conversations lost. No persistence. |
| No content caching | HIGH | Every restart re-scrapes (30s delay). If HUJI is down, server fails to start. |
| LLM hallucination | HIGH | Gemini can still invent quotes despite prompt rules |
| No rate limiting | MEDIUM | One user can flood the API and burn through free tier quota |
| No logging | MEDIUM | No way to know what questions are asked or if answers are wrong |
| PDF RTL reversal | MEDIUM | Some PDF text extracted backwards — degrades answer quality |
| Single process | MEDIUM | One server, one machine — if it goes down, everything is down |
| API key in .env only | LOW | Fine for local, risky if ever pushed to Git accidentally |
| No error boundaries | LOW | Any scraping failure crashes the whole startup |

### Is context stuffing acceptable at this scale?

**Yes — for now.** Here is why:

- 38,000 chars ≈ 10,000 tokens. Gemini 2.5 Flash context window = 1,000,000 tokens. You are using 1%.
- The content is static (6 pages, 1 PDF). It does not grow.
- Retrieval adds complexity. At this scale it adds nothing.

**When to move to RAG:** If the university adds more sources (course catalog, FAQ database, past student questions), and total content exceeds ~200,000 tokens, then RAG becomes worth it.

---

## 2. Recommended Improvements by Priority

### Must-Have for Production

| # | Improvement | Why |
|---|---|---|
| 1 | Cache scraped content to a JSON file | Startup in 2s instead of 30s. Server starts even if HUJI is offline. |
| 2 | Rate limiting per IP | Prevents API quota abuse |
| 3 | Structured logging | Know what is broken and what students ask |
| 4 | Graceful scraping failure | If one source fails, use cached version — don't crash |
| 5 | Environment variable validation | Crash early with clear message if API key is missing |

### Nice-to-Have

| # | Improvement | Why |
|---|---|---|
| 6 | Redis for sessions | Sessions survive server restart |
| 7 | Copy answer button in UI | Students need to copy answers to email/notes |
| 8 | Feedback buttons (thumbs up/down) | Tells you which answers are wrong |
| 9 | Scheduled content refresh (every 24h) | Keeps content fresh without restart |
| 10 | Source highlighting in answer | Show which source the answer came from |

### Future Scaling

| # | Improvement | Why |
|---|---|---|
| 11 | Move to RAG with vector DB | When content grows beyond 200K tokens |
| 12 | Docker deployment | Reproducible, deployable anywhere |
| 13 | Analytics dashboard | See most common questions, failure rate |
| 14 | Admin panel | Update sources without restarting server |
| 15 | Multi-language support | English-speaking students |

---

## 3. RAG Improvement Plan

### Should you move from context stuffing to RAG?

**Not yet.** But here is the full plan for when you need it.

### When to switch
- Total source content exceeds 150,000 tokens
- Answer quality degrades (Gemini starts missing content in long prompts)
- You add new sources (course catalog, registration FAQ, etc.)

### RAG Architecture

```
Documents
    ↓
Chunking (split into pieces)
    ↓
Embedding model (text → vectors)
    ↓
Vector Database (store vectors)
    ↓
Query comes in
    ↓
Embed the query
    ↓
Find top-K most similar chunks
    ↓
Pass chunks + query to Gemini
    ↓
Answer with citations
```

### Chunking Strategy

```python
# Recommended settings for Hebrew academic text
CHUNK_SIZE = 800        # characters (not tokens)
CHUNK_OVERLAP = 150     # overlap between chunks to avoid cutting context
MIN_CHUNK_SIZE = 200    # discard tiny chunks

def chunk_document(text: str, source_name: str, source_url: str) -> list[dict]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end]
        chunks.append({
            "text": chunk_text,
            "source": source_name,
            "url": source_url,
            "chunk_index": len(chunks)
        })
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks
```

### Embedding Model

| Option | Language support | Cost | Recommendation |
|---|---|---|---|
| `text-embedding-004` (Google) | Good Hebrew | Free tier | Best fit — same ecosystem as Gemini |
| `text-embedding-3-small` (OpenAI) | Good Hebrew | ~$0.02/1M tokens | Good alternative |
| `intfloat/multilingual-e5-large` | Excellent Hebrew | Free, local | Best quality, runs on your machine |

**Recommended:** `text-embedding-004` from Google — same API key, free tier, good Hebrew.

### Vector Database

| Option | Best for | Cost |
|---|---|---|
| **FAISS** (Facebook) | Local, no server needed | Free |
| **Chroma** | Local or cloud, easy setup | Free |
| **Supabase pgvector** | Production, SQL + vectors | Free tier |
| Pinecone | Large scale | Paid |

**Recommended for this project:** Start with **FAISS** (local, zero cost, zero setup). Move to **Supabase pgvector** when deploying to cloud.

### Retrieval Settings

```python
TOP_K_CHUNKS = 5        # retrieve top 5 most relevant chunks per query
MIN_SIMILARITY = 0.75   # discard chunks below this cosine similarity score
```

### Citations in RAG answers

```python
# Each retrieved chunk carries its source metadata
context = ""
sources_used = []
for chunk in retrieved_chunks:
    context += f"\n[מקור: {chunk['source']} | {chunk['url']}]\n{chunk['text']}\n"
    sources_used.append(chunk['url'])

prompt = f"ענה על השאלה הבאה בהתבסס ONLY על המקורות הבאים:\n{context}\n\nשאלה: {question}"
```

---

## 4. Anti-Hallucination and Citation Verification

### The core problem
Gemini knows Hebrew academic language from its training data. Even when told to answer only from provided content, it can blend its training knowledge with the provided text. This is how fake page numbers and invented quotes appear.

### Strategy 1 — Grounded generation (current, improved)
Make the system prompt extremely explicit:

```
RULE: The "ציטוט" section must contain ONLY text copied character-by-character
from the content below. Do a literal search. If you cannot find the sentence
word-for-word — write exactly: "לא נמצא ציטוט מדויק. לאימות פנה למזכירות."
Never write a page number unless it appears in the text below.
```

### Strategy 2 — Post-generation verification (recommended addition)
After Gemini responds, programmatically check if the quoted text actually appears in the source content:

```python
def verify_citation(reply: str, source_content: str) -> str:
    import re
    # Extract text between quotation marks from the reply
    quotes = re.findall(r'"([^"]{20,})"', reply)
    for quote in quotes:
        # Check if quote exists verbatim in source (allow small tolerance)
        if quote not in source_content:
            reply = reply.replace(
                f'"{quote}"',
                f'[ציטוט לא אומת — לא נמצא במקורות הרשמיים. יש לפנות למזכירות.]'
            )
    return reply
```

### Strategy 3 — "I don't know" fallback
Add to system prompt:
```
If the answer is not clearly present in the source content below,
respond ONLY with:
"שאלה זו אינה מכוסה במקורות הרשמיים שנטענו.
לקבלת מידע מדויק, פנה למזכירות התלמידים: [link]"
Do not try to answer from general knowledge.
```

### Strategy 4 — Temperature = 0
Set Gemini temperature to 0 for maximum factual consistency:
```python
model = genai.GenerativeModel(
    model_name="gemini-3.0-flash",
    system_instruction=SYSTEM_PROMPT,
    generation_config=genai.GenerationConfig(temperature=0.0)
)
```

---

## 5. Scraping and Data Freshness

### Problem: scraping at every startup
Every time you restart the server (even for a code fix), it waits 30 seconds to re-scrape. If HUJI is down, the server crashes.

### Solution: content caching

```python
# scraper.py — cache to file
import json, os, time

CACHE_FILE = "scraped_content.json"
CACHE_MAX_AGE_HOURS = 24

def load_cached_content() -> str | None:
    if not os.path.exists(CACHE_FILE):
        return None
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    age_hours = (time.time() - data["timestamp"]) / 3600
    if age_hours > CACHE_MAX_AGE_HOURS:
        return None  # cache expired
    return data["content"]

def save_cache(content: str):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump({"content": content, "timestamp": time.time()}, f)

async def load_all_sources() -> str:
    cached = load_cached_content()
    if cached:
        print("[scraper] Using cached content (less than 24h old)")
        return cached
    print("[scraper] Cache expired or missing — scraping fresh content...")
    content = await _run_scraper()
    save_cache(content)
    return content
```

### Scheduled refresh
Add a background task that re-scrapes every 24 hours while the server runs:

```python
# In app.py lifespan
import asyncio

async def refresh_content_loop():
    while True:
        await asyncio.sleep(24 * 3600)  # wait 24 hours
        print("[refresh] Refreshing HUJI source content...")
        new_content = await load_all_sources(force=True)
        model_instance["model"] = genai.GenerativeModel(
            model_name="gemini-3.0-flash",
            system_instruction=SYSTEM_BASE + new_content,
        )
        print("[refresh] Done.")

@asynccontextmanager
async def lifespan(app):
    content = await load_all_sources()
    model_instance["model"] = build_model(content)
    asyncio.create_task(refresh_content_loop())
    yield
```

### PDF Hebrew RTL fix
The best fix for reversed Hebrew PDF text is to use `pdfplumber` with explicit RTL layout parameters:

```python
with pdfplumber.open(pdf_bytes) as pdf:
    for page in pdf.pages:
        # Extract words with position data
        words = page.extract_words(x_tolerance=3, y_tolerance=3)
        # Sort by y (line) then by x descending (RTL)
        words.sort(key=lambda w: (round(w['top']/5)*5, -w['x0']))
        line_text = ' '.join(w['text'] for w in words)
```

### Change detection
Before scraping, check if the page has changed since last scrape:

```python
import hashlib

def page_hash(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()

# Save hash with cache, compare on next refresh
# Only re-embed/re-process if hash changed
```

---

## 6. Backend and Session Management

### Replace in-memory sessions with Redis

```python
# pip install redis
import redis, json

r = redis.Redis(host='localhost', port=6379, decode_responses=True)
SESSION_TTL = 3600  # 1 hour

def save_session(session_id: str, history: list):
    r.setex(session_id, SESSION_TTL, json.dumps(history))

def load_session(session_id: str) -> list:
    data = r.get(session_id)
    return json.loads(data) if data else []
```

### Rate limiting

```python
# pip install slowapi
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/chat")
@limiter.limit("10/minute")  # max 10 messages per minute per IP
async def chat(request: Request):
    ...
```

### Structured logging

```python
import logging, json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mba_advisor")

# In the chat endpoint:
logger.info(json.dumps({
    "event": "chat",
    "session_id": session_id[:8],  # partial for privacy
    "question_length": len(user_message),
    "timestamp": datetime.utcnow().isoformat()
}))
```

### Error handling improvements

```python
@app.post("/chat")
async def chat(request: Request):
    try:
        ...
    except genai.types.BlockedPromptException:
        return JSONResponse({"reply": "השאלה נחסמה על ידי מסנן הבטיחות. נסה לנסח מחדש."})
    except genai.types.StopCandidateException:
        return JSONResponse({"reply": "לא הצלחתי לייצר תשובה. נסה שוב."})
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return JSONResponse({"reply": "שגיאה פנימית. נסה שוב בעוד מספר שניות."}, status_code=500)
```

---

## 7. Frontend Improvements

### Copy answer button
```javascript
// Add to each bot bubble
const copyBtn = document.createElement("button");
copyBtn.className = "copy-btn";
copyBtn.textContent = "העתק";
copyBtn.onclick = () => navigator.clipboard.writeText(bubble.innerText);
wrapper.appendChild(copyBtn);
```

### Feedback buttons (thumbs up / down)
```javascript
const feedback = document.createElement("div");
feedback.className = "feedback-row";
feedback.innerHTML = `
  <button class="thumb" data-val="up">👍</button>
  <button class="thumb" data-val="down">👎</button>`;
feedback.querySelectorAll(".thumb").forEach(btn => {
  btn.onclick = () => {
    fetch("/feedback", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, value: btn.dataset.val })
    });
    feedback.innerHTML = "<span style='opacity:0.5'>תודה על המשוב</span>";
  };
});
wrapper.appendChild(feedback);
```

### Disclaimer always visible
Pin the disclaimer at the top of the chat instead of repeating it in every message:
```html
<div class="disclaimer-banner">
  ⚠️ מענה זה ניתן על-ידי בוט. יש לאמת מול מזכירות התלמידים.
</div>
```

### Show which source was used
Parse the bot reply for the URL and render it as a clickable chip:
```javascript
const urlMatch = reply.match(/https?:\/\/[^\s]+/g);
if (urlMatch) {
  urlMatch.forEach(url => {
    const chip = document.createElement("a");
    chip.href = url;
    chip.target = "_blank";
    chip.className = "source-chip";
    chip.textContent = "קישור למקור ↗";
    bubble.appendChild(chip);
  });
}
```

---

## 8. Deployment Recommendation

### Recommended: Railway.app

Reasons: supports Playwright (Chromium), Python-native, $5/month, easy environment variables, no Docker required to start.

### Environment variables (never hardcode)
```bash
GEMINI_API_KEY=your_key_here
REDIS_URL=redis://...         # if using Redis
ENVIRONMENT=production
```

### Dockerfile (for any cloud)
```dockerfile
FROM python:3.11-slim

# Install Playwright system dependencies
RUN apt-get update && apt-get install -y \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxkbcommon0 libxcomposite1 \
    libxdamage1 libxfixes3 libxrandr2 libgbm1 \
    libasound2 libpango-1.0-0 libcairo2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN playwright install chromium

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### requirements.txt for production
```
fastapi
uvicorn[standard]
google-generativeai
python-dotenv
jinja2
python-multipart
requests
beautifulsoup4
pdfplumber
playwright
slowapi
redis
```

---

## 9. Concrete Refactor Plan

### Phase 1 — Quick wins (1-2 days, no architecture change)

- [ ] Add content caching to JSON file → startup goes from 30s to 2s
- [ ] Add `temperature=0` to Gemini config → more consistent answers
- [ ] Add post-generation citation verification → catch hallucinated quotes
- [ ] Add rate limiting (slowapi) → protect API quota
- [ ] Pin disclaimer as top banner, not per-message
- [ ] Add copy button to bot messages
- [ ] Validate `GEMINI_API_KEY` on startup — crash early with clear message

### Phase 2 — Production hardening (3-5 days)

- [ ] Add structured JSON logging
- [ ] Add feedback endpoint (thumbs up/down) and log to file
- [ ] Add graceful degradation — if scraping fails, use cached content
- [ ] Add scheduled 24h content refresh background task
- [ ] Fix PDF RTL extraction with sorted word coordinates
- [ ] Add `/health` endpoint for monitoring
- [ ] Deploy to Railway with Docker

### Phase 3 — Move to proper RAG (1-2 weeks, when content grows)

- [ ] Install FAISS + Google embedding model
- [ ] Chunk all 6 sources into 800-char chunks with 150-char overlap
- [ ] Embed all chunks and save FAISS index to disk
- [ ] Replace context stuffing with: embed query → retrieve top 5 chunks → pass to Gemini
- [ ] Add source URL to each chunk metadata
- [ ] Move sessions to Redis

### Phase 4 — Analytics and evaluation (ongoing)

- [ ] Build simple admin page showing: questions asked today, thumbs down count, most common topics
- [ ] Create a test set of 20 real student questions with known correct answers
- [ ] Run evaluation monthly: what % of answers are correct?
- [ ] Add A/B testing: compare context stuffing vs RAG answer quality

---

## 10. Suggested Folder Structure

```
mba_advisor/
│
├── app/
│   ├── __init__.py
│   ├── main.py              ← FastAPI app, lifespan, routes
│   ├── config.py            ← env vars, settings
│   ├── models.py            ← Pydantic request/response schemas
│   └── dependencies.py      ← shared dependencies (model, rate limiter)
│
├── scraper/
│   ├── __init__.py
│   ├── runner.py            ← scrape_runner.py (subprocess)
│   ├── cache.py             ← read/write scraped_content.json
│   └── pdf_parser.py        ← pdfplumber + RTL fix
│
├── ai/
│   ├── __init__.py
│   ├── gemini.py            ← model init, send_message wrapper
│   ├── prompt.py            ← system prompt builder
│   └── verify.py            ← citation verification
│
├── sessions/
│   ├── __init__.py
│   └── store.py             ← in-memory or Redis session store
│
├── static/
│   ├── style.css
│   └── script.js
│
├── templates/
│   └── index.html
│
├── data/
│   └── scraped_content.json ← cached source content
│
├── logs/
│   └── chat.log             ← structured logs
│
├── tests/
│   ├── test_scraper.py
│   ├── test_citations.py
│   └── eval_questions.json  ← 20 real student Q&A for evaluation
│
├── .env                     ← secrets (never commit)
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 11. Example Code Snippets

### A — Cache scraped content
```python
# scraper/cache.py
import json, os, time

CACHE_FILE = "data/scraped_content.json"
CACHE_TTL_HOURS = 24

def get_cached() -> str | None:
    if not os.path.exists(CACHE_FILE):
        return None
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    if (time.time() - data["timestamp"]) / 3600 > CACHE_TTL_HOURS:
        return None
    print(f"[cache] Using cached content ({data['char_count']} chars)")
    return data["content"]

def set_cache(content: str):
    os.makedirs("data", exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "content": content,
            "timestamp": time.time(),
            "char_count": len(content)
        }, f, ensure_ascii=False)
```

### B — Chunking documents
```python
# ai/chunker.py
def chunk_text(text: str, source: str, url: str,
               size: int = 800, overlap: int = 150) -> list[dict]:
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start + size]
        if len(chunk) > 100:
            chunks.append({"text": chunk, "source": source,
                           "url": url, "index": len(chunks)})
        start += size - overlap
    return chunks
```

### C — Creating embeddings (Google)
```python
# ai/embeddings.py
import google.generativeai as genai

def embed_texts(texts: list[str]) -> list[list[float]]:
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=texts,
        task_type="retrieval_document"
    )
    return result["embedding"]

def embed_query(query: str) -> list[float]:
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=query,
        task_type="retrieval_query"
    )
    return result["embedding"]
```

### D — FAISS retrieval
```python
# ai/retrieval.py
import faiss, numpy as np

def build_index(embeddings: list[list[float]]) -> faiss.Index:
    dim = len(embeddings[0])
    index = faiss.IndexFlatIP(dim)  # inner product = cosine similarity
    vectors = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(vectors)
    index.add(vectors)
    return index

def retrieve(query_embedding: list[float], index: faiss.Index,
             chunks: list[dict], top_k: int = 5) -> list[dict]:
    q = np.array([query_embedding], dtype="float32")
    faiss.normalize_L2(q)
    scores, indices = index.search(q, top_k)
    return [chunks[i] for i in indices[0] if scores[0][list(indices[0]).index(i)] > 0.7]
```

### E — Passing chunks to Gemini
```python
# ai/gemini.py
def build_rag_prompt(question: str, chunks: list[dict]) -> str:
    context_parts = []
    for chunk in chunks:
        context_parts.append(
            f"[מקור: {chunk['source']} | {chunk['url']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    return f"""ענה על השאלה הבאה בהתבסס ONLY על המקורות הבאים.
אם התשובה לא מופיעה במקורות — אמור: "המידע אינו זמין."

מקורות:
{context}

שאלה: {question}"""
```

### F — Redis session store
```python
# sessions/store.py
import redis, json

r = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

def save_history(session_id: str, history: list, ttl: int = 3600):
    r.setex(session_id, ttl, json.dumps(history, ensure_ascii=False))

def load_history(session_id: str) -> list:
    raw = r.get(session_id)
    return json.loads(raw) if raw else []
```

### G — Citation verifier
```python
# ai/verify.py
import re

def verify_and_clean_citations(reply: str, source_content: str) -> str:
    quotes = re.findall(r'"([^"]{20,})"', reply)
    for quote in quotes:
        if quote.strip() not in source_content:
            reply = reply.replace(
                f'"{quote}"',
                '"[ציטוט לא אומת — פנה למזכירות לאימות]"'
            )
    return reply
```

---

## Summary — Top 5 Things to Do Now

| Priority | Action | Time | Impact |
|---|---|---|---|
| 1 | Add content cache to JSON file | 1 hour | Startup: 30s → 2s, resilient to HUJI downtime |
| 2 | Set Gemini temperature=0 | 5 min | More consistent, less creative hallucination |
| 3 | Add citation verifier post-processing | 2 hours | Catches fake quotes before student sees them |
| 4 | Add rate limiting | 30 min | Protects free API quota |
| 5 | Deploy to Railway with Dockerfile | 2 hours | Always online, no laptop needed |

---

*This document is a living plan. Revisit after each phase is complete.*  
*Stack: Python 3.11 · FastAPI · Uvicorn · Google Gemini 2.5 Flash · Playwright · pdfplumber*
