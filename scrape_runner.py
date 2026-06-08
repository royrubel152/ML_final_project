"""Runs as a standalone process — no asyncio conflict."""
import sys
import io
import json
import requests
import pdfplumber
from playwright.sync_api import sync_playwright

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ── bschool.huji.ac.il sources (Playwright-rendered) ─────────────
SOURCES = {
    "תקנון אקדמי (PDF)": "https://bschool.huji.ac.il/sites/default/files/business/files/academic_regulations_for_masters_programs_01.pdf",
    "פטורים": "https://bschool.huji.ac.il/ptorim",
    "קבלה": "https://bschool.huji.ac.il/mba/admittance",
    "התמחויות": "https://bschool.huji.ac.il/mba/specializations",
    "מסלול מואץ": "https://bschool.huji.ac.il/Accelerated-study-track",
    "מסלול מואץ - כל הקורסים": "https://bschool.huji.ac.il/Accelerated-study-track-all",
}

# ── shnaton.huji.ac.il מוסמך programs (JSON API) ─────────────────
SHNATON_YEAR = 2026
SHNATON_API = "https://shnaton.huji.ac.il/api"
SHNATON_THRESHOLD_PARAMS = f"activeYear={SHNATON_YEAR}&thresholdYear={SHNATON_YEAR}&include=2"

SHNATON_ROADMAPS = {
    "מנהל עסקים עם התמחות ביזמות וחדשנות, עיוני":                  "943-3894",
    "מנהל עסקים, מחקרי":                                            "322-3220",
    "מנהל עסקים, עיוני":                                            "322-3222",
    "תואר בינלאומי במנהל עסקים - ניהול חדשנות ויזמות רפואית (ללא)": "826-3254",
    "תואר בינלאומי במנהל עסקים - ניהול חדשנות ויזמות רפואית, עיוני": "826-3255",
    "מנהל עסקים (322-8168)":                                         "322-8168",
    "מנהל עסקים (322-3792)":                                         "322-3792",
}

SHNATON_SPECIALIZATIONS = {
    "התמחות במדע המידע בניהול - משנית":                      "3661",
    "התמחות במימון ובנקאות - משנית":                          "3111",
    "התמחות במימון ובנקאות - ראשית":                          "3113",
    "התמחות ראשית בניהול פיננסי לחשבונאים":                  "3114",
    "התמחות בשיווק - משנית":                                  "3121",
    "התמחות בשיווק - ראשית":                                  "3123",
    "התמחות בהתנהגות ארגונית ומנהל משאבי אנוש - משנית":      "3331",
    "התמחות בשיווק (3228)":                                   "3228",
    "התמחות במדע המידע בניהול - ראשית":                       "3662",
    "התמחות באנליטיקה של נתוני עתק - ראשית":                 "3663",
    "התמחות בפינטק - ראשית":                                  "3798",
    "התמחות בהתנהגות ארגונית ומנהל משאבי אנוש (3224)":       "3224",
    "התמחות בהתנהגות ארגונית ומנהל משאבי אנוש - ראשית":      "3333",
    "התמחות באסטרטגיה ויזמות - משנית":                       "3441",
    "התמחות באסטרטגיה ויזמות - ראשית":                       "3443",
    "התמחות בחקר ביצועים - משנית":                           "3551",
    "התמחות בחקר ביצועים - ראשית":                           "3552",
    "התמחות בניהול, חדשנות ויזמות רפואית - ראשית":           "3234",
    "התמחות בניהול ביו-רפואי - ראשית":                       "3664",
    'התמחות במימון נדל"ן - ראשית':                            "3795",
}

NOISE_SELECTORS = [
    "nav", "footer", "header", ".menu", ".breadcrumb",
    ".social", "script", "style", "noscript", ".search-block",
    "#toolbar-administration", ".contextual",
]

SHNATON_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://shnaton.huji.ac.il/",
    "Accept": "application/json",
}

HEBREW_DAYS = {1: "ראשון", 2: "שני", 3: "שלישי", 4: "רביעי", 5: "חמישי", 6: "שישי", 7: "שבת"}


# ── Helpers ───────────────────────────────────────────────────────

def ms_to_time(ms):
    """Convert milliseconds-from-midnight to HH:MM string."""
    if not ms:
        return ""
    total_min = ms // 1000 // 60
    return f"{total_min // 60:02d}:{total_min % 60:02d}"


def fix_rtl_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return line
    hebrew = sum(1 for c in stripped if 'א' <= c <= 'ת')
    if hebrew > len(stripped) * 0.3:
        if stripped and ('א' <= stripped[-1] <= 'ת' or stripped[0] in '.,;:)]-'):
            words = stripped.split()
            if len(words) > 1:
                return ' '.join(reversed(words))
    return line


def fix_pdf_text(text: str) -> str:
    return "\n".join(fix_rtl_line(l) for l in text.splitlines())


def fetch_pdf(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://bschool.huji.ac.il/",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    if b"%PDF" not in resp.content[:10]:
        raise ValueError("Not a PDF")
    import io as _io
    parts = []
    with pdfplumber.open(_io.BytesIO(resp.content)) as pdf:
        for page in pdf.pages:
            t = page.extract_text(x_tolerance=3, y_tolerance=3)
            if t:
                parts.append(t)
    return fix_pdf_text("\n".join(parts))


def fetch_html(browser, url):
    page = browser.new_page()
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        for sel in NOISE_SELECTORS:
            page.evaluate(f"document.querySelectorAll('{sel}').forEach(e=>e.remove())")
        text = page.inner_text("body")
        lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 3]
        return "\n".join(lines)
    finally:
        page.close()


def fetch_syllabus(course_code: str) -> dict:
    """Fetch syllabus for a course — returns grade breakdown, description, teacher info."""
    try:
        r = requests.get(
            f"{SHNATON_API}/syllabus?courseCode={course_code}&year={SHNATON_YEAR}",
            headers=SHNATON_HEADERS, timeout=15
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


def format_grade_breakdown(syllabus: dict) -> str:
    """Build a compact grade breakdown string from syllabus data."""
    components = []
    mappings = [
        ("writtenExamPercentage",       "מבחן בכתב"),
        ("oralExamPercentage",          "מבחן בעל פה"),
        ("midtermExamsPercentage",      "תרגילים/מבחן ביניים"),
        ("assignmentPercentage",        "עבודה"),
        ("presentationPercentage",      "הצגה"),
        ("activeParticipationPercentage", "השתתפות פעילה"),
        ("grade7Percentage",            "מרכיב נוסף"),
    ]
    for field, label in mappings:
        pct = syllabus.get(field, 0) or 0
        if pct:
            components.append(f"{label} {pct}%")
    return " | ".join(components) if components else ""


def _he(obj) -> str:
    """Safely extract Hebrew string from a possibly-None dict."""
    return (obj or {}).get("he") or "" if isinstance(obj, dict) or obj is None else str(obj)


def format_schedule(groups: list) -> str:
    """Extract lecturer names and class schedule from groups data."""
    info = []
    seen_teachers = set()
    seen_sessions = set()

    for group in groups:
        for session in group.get("studySessions", []):
            for teacher in session.get("displayTeachers", []):
                name = _he(teacher.get("name"))
                title = _he(teacher.get("title"))
                if name and name not in seen_teachers:
                    seen_teachers.add(name)
                    info.append(f"מרצה: {f'{title} {name}'.strip()}")

            day = HEBREW_DAYS.get(session.get("dayOfWeek"), "")
            start = ms_to_time(session.get("startTime"))
            end = ms_to_time(session.get("endTime"))
            stype = _he(group.get("studySessionTypeName"))
            if day and start:
                key = f"{day}{start}{end}"
                if key not in seen_sessions:
                    seen_sessions.add(key)
                    info.append(f"מפגש: יום {day} {start}-{end}" + (f" ({stype})" if stype else ""))

    return " | ".join(info)


def format_course(course: dict, fetch_full: bool = False) -> str:
    """Format a single course with optional syllabus enrichment."""
    name_he = _he(course.get("name"))
    code = course.get("code", "")
    pts = course.get("academicPoints", "")
    period_he = _he(course.get("coursePeriodName"))
    remark = (course.get("remark") or "").strip()

    line = f"  - {code}: {name_he} ({pts} נ\"ז, {period_he})"

    schedule = format_schedule(course.get("groups", []))
    if schedule:
        line += f"\n    {schedule}"

    if fetch_full and code:
        syllabus = fetch_syllabus(code)
        grade = format_grade_breakdown(syllabus)
        if grade:
            line += f"\n    ציון סופי: {grade}"
        desc_he = (_he(syllabus.get("courseDescription"))).strip()
        if desc_he:
            line += f"\n    תיאור: {desc_he[:200]}"
        office_he = (_he(syllabus.get("officeHours"))).strip()
        if office_he:
            line += f"\n    שעות קבלה: {office_he[:120]}"

    if remark and not fetch_full:
        line += f"\n    הערה: {remark[:120]}"

    return line


def format_threshold(threshold: dict, fetch_syllabus_for_mandatory: bool = True) -> str:
    """Format one threshold (course group) as readable Hebrew text."""
    sub = threshold.get("subChapter", {})
    sub_name = sub.get("he", "") if isinstance(sub, dict) else str(sub)
    year = threshold.get("displayAcademicYear", "")
    min_naz = threshold.get("minNaz")
    note = (threshold.get("hebrewNote") or "").strip()

    if threshold.get("notIncludeInShnaton") and not threshold.get("courses"):
        return ""

    lines = []
    header = f"שנה {year} - {sub_name}"
    if min_naz:
        header += f" (סה\"כ {min_naz} נ\"ז)"
    lines.append(header)

    if note:
        lines.append(f"הערה: {note[:300]}")

    courses = threshold.get("courses", [])
    is_mandatory = sub_name in ("חובה", "חובת קורסי השלמה", "חובת בחירה", "חובת בחירה סמינרים")
    is_free_elective = sub_name == "בחירה" and not min_naz

    # Free electives: show compact list capped at 20
    if is_free_elective and len(courses) > 20:
        for course in courses[:20]:
            name_he = course.get("name", {}).get("he", "") if isinstance(course.get("name"), dict) else ""
            code = course.get("code", "")
            pts = course.get("academicPoints", "")
            lines.append(f"  - {code}: {name_he} ({pts} נ\"ז)")
        lines.append(f"  ... ועוד {len(courses) - 20} קורסי בחירה נוספים")
    else:
        # Mandatory/required electives: full detail including syllabus
        for course in courses:
            fetch_full = is_mandatory and fetch_syllabus_for_mandatory
            lines.append(format_course(course, fetch_full=fetch_full))

    return "\n".join(lines)


def fetch_shnaton_roadmap(code: str) -> str:
    """Fetch full program roadmap via shnaton API and format as text."""
    url = f"{SHNATON_API}/yearly-roadmaps/{code}/thresholds?{SHNATON_THRESHOLD_PARAMS}"
    r = requests.get(url, headers=SHNATON_HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()

    roadmap = data.get("roadmap", {}) if isinstance(data, dict) else {}
    name_he = roadmap.get("name", {}).get("he", code) if isinstance(roadmap.get("name"), dict) else code
    scope = data.get("scopeName", {}).get("he", "") if isinstance(data.get("scopeName"), dict) else ""
    degree = data.get("diplomaDegreeTypeName", {}).get("he", "") if isinstance(data.get("diplomaDegreeTypeName"), dict) else ""
    dept = data.get("departmentName", {}).get("he", "") if isinstance(data.get("departmentName"), dict) else ""

    lines = [
        f"שם המסלול: {name_he}",
        f"סוג: {scope} | תואר: {degree} | חוג: {dept}",
        f"קוד: {code}",
        "",
    ]
    for threshold in data.get("thresholds", []):
        block = format_threshold(threshold)
        if block:
            lines.append(block)
            lines.append("")

    return "\n".join(lines)


def fetch_shnaton_specialization(code: str) -> str:
    """Fetch specialization via shnaton API and format as text."""
    url = f"{SHNATON_API}/specializations/{code}/thresholds?{SHNATON_THRESHOLD_PARAMS}"
    r = requests.get(url, headers=SHNATON_HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()

    name_he = data.get("name", {}).get("he", code) if isinstance(data.get("name"), dict) else code
    lines = [
        f"שם ההתמחות: {name_he}",
        f"קוד: {code}",
        "",
    ]
    for threshold in data.get("thresholds", []):
        block = format_threshold(threshold)
        if block:
            lines.append(block)
            lines.append("")

    return "\n".join(lines)


# ── Main scraping ─────────────────────────────────────────────────

sections = []

# 1. Original bschool sources via Playwright
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    for name, url in SOURCES.items():
        try:
            if url.endswith(".pdf"):
                content = fetch_pdf(url)
            else:
                content = fetch_html(browser, url)
            content = content[:12000]
            sections.append(f"=== {name} ===\nURL: {url}\n\n{content}")
            print(f"[scraper] OK: {name} ({len(content)} chars)", file=sys.stderr)
        except Exception as e:
            print(f"[scraper] FAILED: {name} -> {e}", file=sys.stderr)
            sections.append(f"=== {name} ===\nURL: {url}\n\n[שגיאה בטעינה]")
    browser.close()

# 2. Full MBA degree programs via shnaton API
print(f"[scraper] Fetching {len(SHNATON_ROADMAPS)} MBA roadmaps (with syllabus enrichment)...", file=sys.stderr)
for name, code in SHNATON_ROADMAPS.items():
    url = f"https://shnaton.huji.ac.il/roadmap/{code}"
    try:
        content = fetch_shnaton_roadmap(code)
        content = content[:14000]
        sections.append(f"=== {name} ===\nURL: {url}\n\n{content}")
        print(f"[scraper] OK: {name} ({len(content)} chars)", file=sys.stderr)
    except Exception as e:
        print(f"[scraper] FAILED: {name} ({code}) -> {e}", file=sys.stderr)
        sections.append(f"=== {name} ===\nURL: {url}\n\n[שגיאה בטעינה]")

# 3. Specializations via shnaton API
print(f"[scraper] Fetching {len(SHNATON_SPECIALIZATIONS)} specializations (with syllabus enrichment)...", file=sys.stderr)
for name, code in SHNATON_SPECIALIZATIONS.items():
    url = f"https://shnaton.huji.ac.il/specialization/{code}"
    try:
        content = fetch_shnaton_specialization(code)
        content = content[:10000]
        sections.append(f"=== {name} ===\nURL: {url}\n\n{content}")
        print(f"[scraper] OK: {name} ({len(content)} chars)", file=sys.stderr)
    except Exception as e:
        print(f"[scraper] FAILED: {name} ({code}) -> {e}", file=sys.stderr)
        sections.append(f"=== {name} ===\nURL: {url}\n\n[שגיאה בטעינה]")

print("\n\n".join(sections))
