import os
import re
import asyncio
import logging
import json
from contextlib import asynccontextmanager
from datetime import datetime
from functools import partial

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import google.generativeai as genai
from dotenv import load_dotenv

from scraper import load_all_sources
from rag import get_or_build_index, retrieve, build_rag_prompt
from sessions import load_history, save_history, delete_session, cleanup_expired
from models.schemas import ChatRequest, ChatResponse, FeedbackRequest, ResetRequest

# ── Config ────────────────────────────────────────────────────────
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing from .env — cannot start.")

genai.configure(api_key=API_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mba_advisor")

# ── System prompt ─────────────────────────────────────────────────
SYSTEM_PROMPT = """אתה בוט יועץ אקדמי של תוכנית MBA בבית הספר לעסקים של האוניברסיטה העברית.

=====================
חוקים קריטיים
=====================
1. ענה אך ורק על בסיס המקטעים שיסופקו בכל שאלה.
2. אל תמציא, אל תנחש, ואל תשתמש בידע חיצוני.
3. אם המידע לא מופיע במקטעים — כתוב: "המידע אינו זמין במקורות הרשמיים. יש לפנות למזכירות התלמידים."
4. ענה בעברית בלבד.
5. אם השאלה אינה קשורה ל-MBA של האוניברסיטה העברית — ענה: "שאלה זו אינה בתחום הייעוץ שלי."

=====================
פורמט תגובה
=====================

**תשובה:**
תשובה ישירה וברורה. כלול נתונים מהמקטעים כגון קודי קורסים, שמות מרצים, זמנים, ציונים — ישירות בגוף הטקסט, ללא גרשיים.

**פרטים נוספים:**
טבלה או רשימה אם יש מספר פריטים (קורסים, תנאים וכו')

**קישור:**
קישור רלוונטי מהמקטעים אם קיים

**כתב ויתור:**
שימו לב, מענה זה ניתן על-ידי בוט. יש לאמת את התשובה במזכירות התלמידים.
"""

# ── Citation verifier ─────────────────────────────────────────────
def verify_citations(reply: str, context: str) -> str:
    # Only check very long quoted strings (100+ chars) — prevents false positives
    # on Hebrew abbreviations like ד"ר or short course names
    for quote in re.findall(r'"([^"]{100,})"', reply):
        if quote.strip() not in context:
            reply = reply.replace(f'"{quote}"', '[יש לאמת מידע זה מול המקורות הרשמיים]')
    return reply

# ── Global state ──────────────────────────────────────────────────
state = {}
limiter = Limiter(key_func=get_remote_address)

# ── Lifespan ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_expired()

    print("[startup] Loading HUJI source content...")
    content = await load_all_sources()
    print(f"[startup] Content loaded: {len(content)} chars")

    print("[startup] Building / loading RAG index...")
    loop = asyncio.get_event_loop()
    index, chunks = await loop.run_in_executor(None, lambda: get_or_build_index(content))
    print(f"[startup] RAG ready — {len(chunks)} chunks indexed")

    state["index"] = index
    state["chunks"] = chunks
    state["model"] = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(temperature=0.0),
    )

    yield

# ── App ───────────────────────────────────────────────────────────
app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


OFF_TOPIC_REPLY = (
    "שאלה זו אינה בתחום הייעוץ שלי. "
    "אני מסייע אך ורק בנושאי תוכנית MBA של האוניברסיטה העברית — "
    "קבלה, מסלולים, פטורים, שכר לימוד ותקנון אקדמי. "
    "לשאלות אחרות, פנה למזכירות התלמידים."
)


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("15/minute")
async def chat(request: Request, body: ChatRequest):
    user_message = body.message
    session_id = body.session_id

    logger.info(json.dumps({
        "event": "chat",
        "session_id": session_id[:8],
        "q_len": len(user_message),
        "ts": datetime.utcnow().isoformat()
    }))

    try:
        model = state["model"]
        index = state["index"]
        chunks = state["chunks"]

        # 1 — Retrieve relevant chunks (in executor — embedding call is sync)
        loop = asyncio.get_event_loop()
        relevant = await loop.run_in_executor(
            None, lambda: retrieve(user_message, index, chunks)
        )

        # 2 — Off-topic gate: no matching chunks means the question is out of scope
        if not relevant:
            logger.info(json.dumps({
                "event": "off_topic",
                "session_id": session_id[:8],
                "ts": datetime.utcnow().isoformat()
            }))
            return JSONResponse({
                "reply": OFF_TOPIC_REPLY,
                "sources_used": [],
                "chunks_found": 0,
            })

        # 3 — Build augmented prompt with retrieved context
        augmented = build_rag_prompt(user_message, relevant)

        # 4 — Load conversation history (capped to last 10 turns) and rebuild chat
        history = load_history(session_id)
        session = model.start_chat(history=history)

        # 5 — Send augmented message (sync SDK in executor)
        response = await loop.run_in_executor(
            None, partial(session.send_message, augmented)
        )
        reply = response.text

        # 6 — Verify citations against retrieved context
        context_text = " ".join(c["text"] for c in relevant)
        reply = verify_citations(reply, context_text)

        # 7 — Persist updated history
        updated_history = list(session.history)
        serializable = [
            {"role": m.role, "parts": [p.text for p in m.parts]}
            for m in updated_history
        ]
        save_history(session_id, serializable)

        return JSONResponse({
            "reply": reply,
            "sources_used": list({c["url"] for c in relevant if c.get("url")}),
            "chunks_found": len(relevant),
        })

    except Exception as e:
        logger.error(f"[ERROR] {e}")
        return JSONResponse(
            {"reply": "שגיאה פנימית. נסה שוב בעוד מספר שניות."},
            status_code=500
        )


@app.post("/feedback")
async def feedback(request: Request, body: FeedbackRequest):
    logger.info(json.dumps({
        "event": "feedback",
        "session_id": body.session_id[:8],
        "value": body.value,
        "ts": datetime.utcnow().isoformat()
    }))
    return JSONResponse({"status": "ok"})


@app.post("/reset")
async def reset(body: ResetRequest):
    delete_session(body.session_id)
    return JSONResponse({"status": "ok"})


@app.get("/courses")
async def courses():
    chunks = state.get("chunks", [])
    names = set()
    pattern = re.compile(r"^\s*-\s+\d+:\s+(.+?)\s+\(")
    for chunk in chunks:
        for line in chunk["text"].split("\n"):
            m = pattern.match(line)
            if m:
                name = m.group(1).strip()
                if name:
                    names.add(name)
    return JSONResponse({"courses": sorted(names)})


@app.get("/health")
async def health():
    return JSONResponse({
        "status": "ok",
        "chunks_indexed": len(state.get("chunks", [])),
        "rag_ready": "index" in state,
    })
