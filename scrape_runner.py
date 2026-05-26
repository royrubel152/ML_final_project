"""Runs as a standalone process — no asyncio conflict."""
import sys
import io
import requests
import pdfplumber
from playwright.sync_api import sync_playwright

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SOURCES = {
    "תקנון אקדמי (PDF)": "https://bschool.huji.ac.il/sites/default/files/business/files/academic_regulations_for_masters_programs_01.pdf",
    "פטורים": "https://bschool.huji.ac.il/ptorim",
    "קבלה": "https://bschool.huji.ac.il/mba/admittance",
    "התמחויות": "https://bschool.huji.ac.il/mba/specializations",
    "מסלול מואץ": "https://bschool.huji.ac.il/Accelerated-study-track",
    "מסלול מואץ - כל הקורסים": "https://bschool.huji.ac.il/Accelerated-study-track-all",
}

NOISE_SELECTORS = [
    "nav", "footer", "header", ".menu", ".breadcrumb",
    ".social", "script", "style", "noscript", ".search-block",
    "#toolbar-administration", ".contextual",
]


def fix_rtl_line(line: str) -> str:
    """Reverse lines that were extracted backwards from RTL PDFs."""
    import unicodedata
    stripped = line.strip()
    if not stripped:
        return line
    # Check if majority of chars are Hebrew (RTL)
    hebrew = sum(1 for c in stripped if '֐' <= c <= '׿')
    if hebrew > len(stripped) * 0.3:
        # Check if line appears reversed: first char is punctuation or last char is a Hebrew letter
        if stripped and ('֐' <= stripped[-1] <= '׿' or stripped[0] in '.,;:)]-'):
            # Reverse words (not chars) to fix RTL extraction
            words = stripped.split()
            if len(words) > 1:
                return ' '.join(reversed(words))
    return line


def fix_pdf_text(text: str) -> str:
    lines = text.splitlines()
    fixed = [fix_rtl_line(l) for l in lines]
    return "\n".join(fixed)


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
    raw = "\n".join(parts)
    return fix_pdf_text(raw)


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


sections = []
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

print("\n\n".join(sections))
