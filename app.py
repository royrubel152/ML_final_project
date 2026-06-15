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
from rag import (get_or_build_index, retrieve_with_context,
                 build_rag_prompt_with_context, enrich_chunks_metadata)
from sessions import load_history, load_state, save_history, delete_session, cleanup_expired
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
3. אם המידע לא מופיע במקטעים — הבחן בין שני מקרים:
   א. אם יש מקטעים רלוונטיים לנושא אך אף אחד לא מכיל את הפרט המבוקש (למשל: שואלים על קורסים ביום מסוים אך אף קורס לא מוזכר בהם ביום הזה) — ענה: "על פי המידע הזמין, אין קורסים [הנושא המבוקש] בתוכנית ה-MBA." ואפשר לפרט מה כן קיים.
   ב. אם אין כלל מקטעים רלוונטיים לשאלה — כתוב: "המידע אינו זמין במקורות הרשמיים. יש לפנות למזכירות התלמידים."
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

# ── Specialization alias map ──────────────────────────────────────
# Maps Hebrew/English keywords that appear in user questions → (spec_name, spec_code)
# spec_name must match SHNATON_SPECIALIZATIONS keys (used as chunk source names)
SPEC_ALIASES: dict[str, tuple[str, str]] = {
    "אנליטיקה של נתוני עתק": ("התמחות באנליטיקה של נתוני עתק - ראשית", "3663"),
    "נתוני עתק":              ("התמחות באנליטיקה של נתוני עתק - ראשית", "3663"),
    "ביג דאטה":               ("התמחות באנליטיקה של נתוני עתק - ראשית", "3663"),
    "big data":               ("התמחות באנליטיקה של נתוני עתק - ראשית", "3663"),
    "אנליטיקה":               ("התמחות באנליטיקה של נתוני עתק - ראשית", "3663"),
    "מדע המידע":              ("התמחות במדע המידע בניהול - ראשית",       "3662"),
    "information science":    ("התמחות במדע המידע בניהול - ראשית",       "3662"),
    "פינטק":                  ("התמחות בפינטק - ראשית",                  "3798"),
    "fintech":                ("התמחות בפינטק - ראשית",                  "3798"),
    "מימון ובנקאות":          ("התמחות במימון ובנקאות - ראשית",          "3113"),
    "בנקאות":                 ("התמחות במימון ובנקאות - ראשית",          "3113"),
    "שיווק":                  ("התמחות בשיווק - ראשית",                  "3123"),
    "אסטרטגיה ויזמות":       ("התמחות באסטרטגיה ויזמות - ראשית",       "3443"),
    "יזמות":                  ("התמחות באסטרטגיה ויזמות - ראשית",       "3443"),
    "אסטרטגיה":              ("התמחות באסטרטגיה ויזמות - ראשית",       "3443"),
    "משאבי אנוש":            ("התמחות בהתנהגות ארגונית ומנהל משאבי אנוש - ראשית", "3333"),
    "התנהגות ארגונית":       ("התמחות בהתנהגות ארגונית ומנהל משאבי אנוש - ראשית", "3333"),
    "חקר ביצועים":           ("התמחות בחקר ביצועים - ראשית",            "3552"),
    "ניהול פיננסי":          ("התמחות ראשית בניהול פיננסי לחשבונאים",   "3114"),
    'נדל"ן':                  ('התמחות במימון נדל"ן - ראשית',            "3795"),
    "נדלן":                   ('התמחות במימון נדל"ן - ראשית',            "3795"),
    "ביו-רפואי":             ("התמחות בניהול ביו-רפואי - ראשית",        "3664"),
    "ביו רפואי":             ("התמחות בניהול ביו-רפואי - ראשית",        "3664"),
    "ניהול רפואי":           ("התמחות בניהול, חדשנות ויזמות רפואית - ראשית", "3234"),
}

CORRECTION_PATTERNS = [
    "למה לא אמרת", "לא ציינת", "שכחת", "פספסת", "יש עוד", "ויש גם",
    "אז למה", "הייתי צריך", "3 אופציות", "שלוש אופציות", "יש 3",
    "לא הזכרת", "חסר", "לא הצגת", "לא הופיע", "גם אופציה", "גם זה",
    "לא ציינת", "הצגת רק", "רק שתיים", "רק אחד", "חסרה",
]

EXPLICIT_SWITCH_PATTERNS = [
    "עכשיו שאלה על", "תחליף נושא", "שאלה אחרת", "נושא אחר",
    "לגבי התמחות אחרת", "עזוב את",
]


def detect_intent(message: str, state: dict) -> dict:
    """
    Fast regex/keyword intent detection. Returns:
      spec_name, spec_code  — detected or carried from state
      topic                 — seminar / lecturer / schedule / courses / general
      is_correction         — user is correcting a previous answer
      is_follow_up          — short message continuing same specialization
    """
    msg = message

    # Check for explicit topic switch — clear specialization context
    explicit_switch = any(pat in msg for pat in EXPLICIT_SWITCH_PATTERNS)

    # Detect specialization from message (longer aliases checked first → avoids partial matches)
    detected_spec_name = detected_spec_code = None
    for alias, (sname, scode) in sorted(SPEC_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in msg or alias.lower() in msg.lower():
            detected_spec_name, detected_spec_code = sname, scode
            break

    # When an active specialization is already set, only switch to a new one if the
    # user explicitly says "התמחות של X" / "בהתמחות X" — otherwise a keyword like
    # "מדע המידע" might be a course name inside the active specialization, not a
    # request to switch to the Information Science specialization.
    has_active = bool(state.get("active_spec_code"))
    explicit_spec_reference = any(
        marker in msg for marker in ["בהתמחות", "התמחות של", "התמחות ב", "לגבי ההתמחות",
                                     "כהתמחות", "התמחות ראשית", "התמחות משנית"]
    )
    if has_active and detected_spec_code and not explicit_spec_reference and not explicit_switch:
        # Keep active spec — don't switch just because a keyword appeared
        detected_spec_name = state.get("active_spec_name")
        detected_spec_code = state.get("active_spec_code")

    # Carry active specialization when nothing was detected
    if not detected_spec_code and not explicit_switch:
        detected_spec_name = state.get("active_spec_name")
        detected_spec_code = state.get("active_spec_code")

    # Topic detection
    topic = "general"
    if any(k in msg for k in ["סמינריון", "סמינר", "seminar"]):
        topic = "seminar"
    elif any(k in msg for k in ["מרצה", "מלמד", "מי מלמד", "מי מרצה"]):
        topic = "lecturer"
    elif any(k in msg for k in ["מועד", "שעה", "יום", "סמסטר", "מתי"]):
        topic = "schedule"
    elif any(k in msg for k in ["ציון", "ציונים", "חלוקת ציונים", "בחינה"]):
        topic = "grades"
    elif any(k in msg for k in ["פטור", "פטורים"]):
        topic = "exemption"
    elif any(k in msg for k in ["קבלה", "תנאי קבלה", "הרשמה"]):
        topic = "admission"

    is_correction = any(pat in msg for pat in CORRECTION_PATTERNS)
    words = msg.split()
    is_follow_up = (
        not explicit_switch
        and not detected_spec_code != state.get("active_spec_code")
        and len(words) < 18
        and bool(state.get("active_spec_code"))
    )

    # Detect questions about general MBA mandatory courses (not specialization-specific)
    is_general_mandatory = bool(
        re.search(r"חובה.{0,30}(כל|כלל).{0,20}(סטודנט|תלמיד|תוכנית|ה?תואר)", msg) or
        re.search(r"(כל|כלל).{0,20}(סטודנט|תלמיד).{0,30}חובה", msg) or
        re.search(r"חובה.{0,30}(ללא|בלי) קשר.{0,20}התמחות", msg) or
        re.search(r"חובה.{0,20}(כלליים|משותפ|הכלל)", msg) or
        re.search(r"(חובות|חובה).{0,20}(תכנית|תוכנית|ה?תואר).{0,20}(בכלל|כולם|ה?כל)", msg)
    )

    return {
        "spec_name":            detected_spec_name,
        "spec_code":            detected_spec_code,
        "topic":                topic,
        "is_correction":        is_correction,
        "is_follow_up":         is_follow_up,
        "is_general_mandatory": is_general_mandatory,
    }


# ── Citation verifier ─────────────────────────────────────────────
def verify_citations(reply: str, context: str) -> str:
    # Only flag long quoted strings (100+ chars) where the " is NOT part of a
    # Hebrew abbreviation like נ"ז or ד"ר (i.e. not preceded/followed by Hebrew letter)
    for quote in re.findall(r'(?<![א-ת])"([^"]{100,})"(?![א-ת])', reply):
        if quote.strip() not in context:
            reply = reply.replace(f'"{quote}"', '[יש לאמת מידע זה מול המקורות הרשמיים]')
    return reply

# ── Global state ──────────────────────────────────────────────────
state = {}
limiter = Limiter(key_func=get_remote_address)

# ── Mandatory courses summary builder ────────────────────────────
def _build_mandatory_courses_chunk(chunks: list[dict]) -> dict:
    """
    Scan roadmap chunks for the 'שנה 1 - חובה' section and build a single
    clean text block listing all general mandatory courses. This avoids
    passing 7+ mid-sentence continuation fragments to the LLM.
    """
    import re as _re
    course_re = _re.compile(r"-\s+(\d{5}):\s+([^\(]+)\s+\((\d+)\s+נ\"ז,\s*([^\)]+)\)")
    ROADMAP_SRC = "מנהל עסקים, מחקרי"

    # Find the index of the first mandatory header chunk
    start_idx = None
    for i, c in enumerate(chunks):
        if c["source"] == ROADMAP_SRC:
            hp = c["text"].find("שנה 1 - חובה")
            if hp != -1 and hp < 300:
                start_idx = i
                break

    if start_idx is None:
        return None

    # Collect course lines from start_idx through the next 8 chunks (same source)
    courses = []
    seen = set()
    for i in range(start_idx, min(start_idx + 9, len(chunks))):
        if chunks[i]["source"] != ROADMAP_SRC:
            break
        for m in course_re.finditer(chunks[i]["text"]):
            code, name, credits, semester = m.group(1), m.group(2).strip(), m.group(3), m.group(4).strip()
            if code not in seen:
                seen.add(code)
                courses.append(f"  - {code}: {name} ({credits} נ\"ז, {semester})")

    if not courses:
        return None

    text = (
        "קורסי חובה כלליים לכלל תלמידי MBA (חובה במסגרת לימודי MBA, ציון עובר 60):\n"
        "בעבור פטור מקורס חובה כללי יש ללמוד קורס בחירה כללי להשלמת נ\"ז.\n\n"
        + "\n".join(courses)
    )
    return {
        "text": text,
        "source": "מנהל עסקים (קורסי חובה כלליים)",
        "url": "https://shnaton.huji.ac.il/roadmap/322-3220",
        "chunk_id": -1,
        "score": 0.95,
    }


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

    # Build spec_map for metadata enrichment (spec_name → spec_code)
    spec_map = {name: code for name, code in [v for v in SPEC_ALIASES.values()]}
    enrich_chunks_metadata(chunks, spec_map)
    print(f"[startup] Chunk metadata enriched — "
          f"{sum(1 for c in chunks if c.get('spec_code'))} chunks tagged with spec_code")

    # Pre-build a clean synthetic chunk listing all general mandatory courses.
    # The raw data is spread across 7 overlapping chunks (mid-sentence fragments),
    # making it hard for the LLM to parse. We extract a clean course list once here.
    state["mandatory_courses_chunk"] = _build_mandatory_courses_chunk(chunks)

    state["index"] = index
    state["chunks"] = chunks
    state["model"] = genai.GenerativeModel(
        model_name="gemini-3.0-flash",
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
        loop = asyncio.get_event_loop()

        # 1 — Load session state and detect intent
        session_state = load_state(session_id)
        intent = detect_intent(user_message, session_state)

        logger.info(json.dumps({
            "event": "intent",
            "session_id": session_id[:8],
            "spec_code": intent["spec_code"],
            "topic": intent["topic"],
            "is_correction": intent["is_correction"],
            "is_follow_up": intent["is_follow_up"],
            "ts": datetime.utcnow().isoformat(),
        }))

        # 2 — Retrieve relevant chunks with specialization context boosting
        relevant = await loop.run_in_executor(
            None,
            lambda: retrieve_with_context(
                user_message, index, chunks,
                active_spec_code=intent["spec_code"],
            ),
        )

        # 2b — Fallback: if spec-filtered retrieval returned nothing, retry without
        #      the spec constraint (user may be asking a general question mid-conversation,
        #      or a secondary-track question whose data is in the bschool section).
        if not relevant and intent["spec_code"]:
            relevant = await loop.run_in_executor(
                None,
                lambda: retrieve_with_context(
                    user_message, index, chunks,
                    active_spec_code=None,
                ),
            )

        # 2b2 — For secondary-specialization queries, also try without spec boosting
        #       even when primary retrieval found something — the answer may be in a
        #       different section (bschool התמחויות page, not Shnaton spec chunks).
        if "משנית" in user_message and relevant:
            general_results = await loop.run_in_executor(
                None,
                lambda: retrieve_with_context(
                    user_message, index, chunks,
                    active_spec_code=None,
                ),
            )
            seen_ids = {r["chunk_id"] for r in relevant}
            for r in general_results:
                if r["chunk_id"] not in seen_ids and "משנית" in r["text"]:
                    relevant.append(r)
                    seen_ids.add(r["chunk_id"])
            relevant.sort(key=lambda x: x["score"], reverse=True)
            relevant = relevant[:5]

        # 2c — For general mandatory questions, prepend the pre-built synthetic chunk
        #      that lists all 8 mandatory courses cleanly (built at startup).
        if intent.get("is_general_mandatory"):
            mandatory_chunk = state.get("mandatory_courses_chunk")
            if mandatory_chunk and not any(r.get("chunk_id") == -1 for r in relevant):
                relevant = [mandatory_chunk] + relevant
            relevant.sort(key=lambda x: x["score"], reverse=True)

        # 3 — Off-topic gate
        if not relevant:
            logger.info(json.dumps({
                "event": "off_topic",
                "session_id": session_id[:8],
                "ts": datetime.utcnow().isoformat(),
            }))
            return JSONResponse({
                "reply": OFF_TOPIC_REPLY,
                "sources_used": [],
                "chunks_found": 0,
            })

        # 4 — Build context-aware augmented prompt
        augmented = build_rag_prompt_with_context(
            user_message,
            relevant,
            active_spec_name=intent["spec_name"],
            is_correction=intent["is_correction"],
        )

        # 5 — Load history and send to model
        history = load_history(session_id)
        session = model.start_chat(history=history)
        response = await loop.run_in_executor(
            None, partial(session.send_message, augmented)
        )
        reply = response.text

        # 6 — Verify citations
        context_text = " ".join(c["text"] for c in relevant)
        reply = verify_citations(reply, context_text)

        # 7 — Persist history + updated state
        updated_history = list(session.history)
        serializable = [
            {"role": m.role, "parts": [p.text for p in m.parts]}
            for m in updated_history
        ]
        new_state = {
            "active_spec_name": intent["spec_name"],
            "active_spec_code": intent["spec_code"],
            "active_topic":     intent["topic"],
            "turn_count":       session_state.get("turn_count", 0) + 1,
        }
        save_history(session_id, serializable, state=new_state)

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
    return JSONResponse({"status": "ok", "state_cleared": True})


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
