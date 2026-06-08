"""
extractor.py — Convert raw HTML, JSON (Shnaton API), and PDF bytes into clean text + structure metadata.

Three public functions:
  extract_html(path)  → dict   (bschool Drupal pages)
  extract_json(path)  → dict   (Shnaton API JSON responses)
  extract_pdf(path)   → dict   (regulations PDF)

All return a common schema:
  {
    text          : str   — clean body text, headings marked with ## / ###
    title         : str   — page/document title
    headings      : list  — [{level, text}, ...]
    tables_count  : int
    list_items_count : int
    paragraphs    : list[str]
    word_count    : int
    char_count    : int
    extraction_warnings : list[str]
  }
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Tags to strip entirely (navigation, UI, scripts) ──────────────────────────
HTML_NOISE_TAGS = [
    "nav", "footer", "header", "script", "style",
    "noscript", "aside", "form", "button", "iframe",
    "svg", "img", "figure",
]

# ── Hebrew common function-words used to check text direction ─────────────────
# If these appear in the text, we assume RTL rendering is correct.
HEBREW_DIRECTION_WORDS = ["של", "על", "את", "הוא", "לא", "עם", "כי", "אם", "גם"]
MIN_DIRECTION_WORDS_FOR_CONFIDENCE = 3


# ─────────────────────────────────────────────────────────────────────────────
# HTML extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_html(path: Path) -> dict:
    """
    Parse an HTML file into clean text and structure metadata.

    Strips navigation, footers, scripts, and styling.
    Preserves heading hierarchy with ## / ### markers in the text.
    Tables are counted but their cell content is also included as text rows.
    """
    from bs4 import BeautifulSoup

    warnings: List[str] = []
    path = Path(path)

    raw_bytes = path.read_bytes()
    # BeautifulSoup with lxml handles encoding detection well
    soup = BeautifulSoup(raw_bytes, "lxml")

    # Strip noise tags before any text extraction
    for tag_name in HTML_NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Also strip elements that look like cookie banners or GDPR notices
    for tag in soup.find_all(class_=re.compile(r"cookie|gdpr|banner|popup|modal", re.I)):
        tag.decompose()
    for tag in soup.find_all(id=re.compile(r"cookie|gdpr|banner|popup|modal", re.I)):
        tag.decompose()

    # Title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Headings
    headings: List[Dict] = []
    for level in range(1, 7):
        for h in soup.find_all(f"h{level}"):
            text = h.get_text(separator=" ", strip=True)
            if text:
                headings.append({"level": level, "text": text})

    # Table count
    tables = soup.find_all("table")
    tables_count = len(tables)

    # List items count
    list_items = soup.find_all("li")
    list_items_count = len(list_items)

    # Build clean text with heading markers
    text_lines: list[str] = []
    body = soup.find("body") or soup
    _collect_text_lines(body, text_lines)
    full_text = "\n".join(line for line in text_lines if line.strip())

    # Split into paragraphs (blank-line separated after joining)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", full_text) if p.strip()]

    if not full_text.strip():
        warnings.append("Empty text after extraction")

    return {
        "text": full_text,
        "title": title,
        "headings": headings,
        "tables_count": tables_count,
        "list_items_count": list_items_count,
        "paragraphs": paragraphs,
        "word_count": len(full_text.split()),
        "char_count": len(full_text),
        "extraction_warnings": warnings,
    }


def _collect_text_lines(tag, lines: List[str]) -> None:
    """Recursively walk BeautifulSoup tree, emit text with heading markers."""
    from bs4 import NavigableString, Tag

    for child in tag.children:
        if isinstance(child, NavigableString):
            text = str(child).strip()
            if text:
                lines.append(text)
        elif isinstance(child, Tag):
            tag_name = child.name.lower() if child.name else ""

            if tag_name in ("h1",):
                text = child.get_text(separator=" ", strip=True)
                if text:
                    lines.append(f"\n# {text}\n")
            elif tag_name in ("h2",):
                text = child.get_text(separator=" ", strip=True)
                if text:
                    lines.append(f"\n## {text}\n")
            elif tag_name in ("h3", "h4", "h5", "h6"):
                text = child.get_text(separator=" ", strip=True)
                if text:
                    lines.append(f"\n### {text}\n")
            elif tag_name == "p":
                text = child.get_text(separator=" ", strip=True)
                if text:
                    lines.append(text)
                    lines.append("")  # blank line after paragraph
            elif tag_name in ("tr",):
                # Emit table rows as pipe-separated text
                cells = [
                    td.get_text(separator=" ", strip=True)
                    for td in child.find_all(["td", "th"])
                ]
                row_text = " | ".join(c for c in cells if c)
                if row_text:
                    lines.append(row_text)
            elif tag_name in ("li",):
                text = child.get_text(separator=" ", strip=True)
                if text:
                    lines.append(f"- {text}")
            elif tag_name in ("br",):
                lines.append("")
            else:
                # Recurse into any other container tag
                _collect_text_lines(child, lines)


# ─────────────────────────────────────────────────────────────────────────────
# JSON extractor (Shnaton API responses)
# ─────────────────────────────────────────────────────────────────────────────

def extract_json(path: Path) -> dict:
    """
    Convert a Shnaton API JSON response into the common text + structure schema.

    Handles two JSON shapes:
      - Yearly roadmap: has roadmapCode, scopeName, departmentName, specializations,
        thresholds_detail, course_combinations
      - Specialization: has code, name, academicInfo, thresholds_detail

    Renders the structured data as Hebrew-first plain text with heading markers
    so downstream EDA treats it the same as any other document.
    """
    import json as _json

    warnings: List[str] = []
    path = Path(path)

    try:
        data = _json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _empty_result(f"JSON parse error: {exc}")

    lines: List[str] = []
    headings: List[Dict] = []
    tables_count = 0
    list_items_count = 0

    def _he(obj) -> str:
        """Extract Hebrew string from a {he: ..., en: ...} name object."""
        if isinstance(obj, dict):
            return obj.get("he") or obj.get("en") or ""
        return str(obj) if obj else ""

    # ── Roadmap shape ─────────────────────────────────────────────────────────
    if "roadmapCode" in data:
        code = data.get("roadmapCode", "")
        roadmap_name = _he(data.get("roadmap", {}).get("name", {})) if data.get("roadmap") else ""
        scope = _he(data.get("scopeName", {}))
        dept = _he(data.get("departmentName", {}))
        faculty = _he(data.get("facultyName", {}))
        degree = _he(data.get("diplomaDegreeTypeName", {}))
        prog_name = _he(data.get("learningProgramDisplayName", {}))

        title = roadmap_name or f"מסלול {code}"
        lines.append(f"# {title}\n")
        headings.append({"level": 1, "text": title})

        if scope:
            lines.append(f"סוג מסלול: {scope}")
        if prog_name:
            lines.append(f"תוכנית לימודים: {prog_name}")
        if degree:
            lines.append(f"תואר: {degree}")
        if dept:
            lines.append(f"מחלקה: {dept}")
        if faculty:
            lines.append(f"פקולטה: {faculty}")
        lines.append(f"קוד מסלול: {code}")
        lines.append("")

        # Thresholds_detail contains course groups with embedded course lists.
        # Shape: [{subChapter: {he}, minNaz, courses: [{code, name.he, academicPoints, coursePeriodName.he}]}]
        thresholds_detail = data.get("thresholds_detail") or []
        if thresholds_detail:
            lines.append("\n## קורסי התוכנית\n")
            headings.append({"level": 2, "text": "קורסי התוכנית"})
            tables_count += 1
            for group in thresholds_detail:
                if not isinstance(group, dict):
                    continue
                group_name = _he(group.get("subChapter", {}))
                min_naz = group.get("minNaz")
                year_label = group.get("displayAcademicYear", "")
                note = _he(group.get("hebrewNote") or {}) if isinstance(group.get("hebrewNote"), dict) else (group.get("hebrewNote") or "")
                header = group_name or "קבוצת קורסים"
                if year_label:
                    header = f"שנה {year_label} — {header}"
                lines.append(f"\n### {header}\n")
                headings.append({"level": 3, "text": header})
                if min_naz is not None and min_naz > 0:
                    lines.append(f"חובה ללמוד לפחות {min_naz} נ\"ז מקבוצה זו.")
                if note and note.strip():
                    lines.append(note.strip())
                courses = group.get("courses") or []
                for course in courses:
                    if not isinstance(course, dict):
                        continue
                    c_code = course.get("code", "")
                    c_name = _he(course.get("name", {}))
                    c_credits = course.get("academicPoints", "")
                    c_semester = _he(course.get("coursePeriodName", {}))
                    c_remark = (course.get("remark") or "").strip()
                    row = f"{c_code} | {c_name} | {c_credits} נ\"ז | {c_semester}"
                    if c_remark:
                        row += f" | {c_remark}"
                    lines.append(f"- {row}")
                    list_items_count += 1
            lines.append("")

        # Legacy thresholds field (admission/GPA requirements — usually empty for MBA)
        plain_thresholds = [t for t in (data.get("thresholds") or []) if not isinstance(t, dict) or not t.get("courses")]
        if plain_thresholds:
            lines.append("## דרישות סף\n")
            headings.append({"level": 2, "text": "דרישות סף"})
            for t in plain_thresholds:
                t_name = _he(t.get("name", {})) if isinstance(t, dict) else str(t)
                t_val = t.get("value", "") if isinstance(t, dict) else ""
                lines.append(f"- {t_name}: {t_val}")
                list_items_count += 1
            lines.append("")

        # Specialization IDs mentioned in the roadmap
        spec_ids = data.get("specializationIds") or []
        if spec_ids:
            lines.append(f"## התמחויות במסלול\n")
            headings.append({"level": 2, "text": "התמחויות במסלול"})
            lines.append(f"מספר התמחויות: {len(spec_ids)}")

    # ── Specialization shape ──────────────────────────────────────────────────
    elif "code" in data and "name" in data and "learningProgramName" in data:
        spec_code = str(data.get("code", ""))
        spec_name = _he(data.get("name", {}))
        prog = _he(data.get("learningProgramDisplayName", {}))

        title = spec_name or f"התמחות {spec_code}"
        lines.append(f"# {title}\n")
        headings.append({"level": 1, "text": title})

        lines.append(f"קוד התמחות: {spec_code}")
        if prog:
            lines.append(f"תוכנית לימודים: {prog}")
        lines.append("")

        # Academic info (credit requirements, scope description)
        acad = data.get("academicInfo")
        if acad and isinstance(acad, dict):
            lines.append("## מידע אקדמי\n")
            headings.append({"level": 2, "text": "מידע אקדמי"})
            scope_info = acad.get("scopeAcademicInfo")
            if scope_info:
                lines.append(_he(scope_info))
            min_pts = acad.get("minPointsTotal")
            max_pts = acad.get("maxPointsTotal")
            if min_pts is not None:
                lines.append(f"מינימום נקודות זכות: {min_pts}")
            if max_pts is not None:
                lines.append(f"מקסימום נקודות זכות: {max_pts}")
            lines.append("")

        # Thresholds_detail contains the course groups for this specialization
        thresholds_detail = data.get("thresholds_detail") or []
        if thresholds_detail:
            lines.append("\n## קורסי ההתמחות\n")
            headings.append({"level": 2, "text": "קורסי ההתמחות"})
            tables_count += 1
            for group in thresholds_detail:
                if not isinstance(group, dict):
                    continue
                group_name = _he(group.get("subChapter", {}))
                min_naz = group.get("minNaz")
                year_label = group.get("displayAcademicYear", "")
                note = group.get("hebrewNote") or ""
                if isinstance(note, dict):
                    note = _he(note)
                header = group_name or "קבוצת קורסים"
                if year_label:
                    header = f"שנה {year_label} — {header}"
                lines.append(f"\n### {header}\n")
                headings.append({"level": 3, "text": header})
                if min_naz is not None and min_naz > 0:
                    lines.append(f"חובה ללמוד לפחות {min_naz} נ\"ז מקבוצה זו.")
                if note and note.strip():
                    lines.append(note.strip())
                courses = group.get("courses") or []
                for course in courses:
                    if not isinstance(course, dict):
                        continue
                    c_code = course.get("code", "")
                    c_name = _he(course.get("name", {}))
                    c_credits = course.get("academicPoints", "")
                    c_semester = _he(course.get("coursePeriodName", {}))
                    c_remark = (course.get("remark") or "").strip()
                    row = f"{c_code} | {c_name} | {c_credits} נ\"ז | {c_semester}"
                    if c_remark:
                        row += f" | {c_remark}"
                    lines.append(f"- {row}")
                    list_items_count += 1
            lines.append("")

    else:
        warnings.append(f"Unrecognized JSON shape; keys: {list(data.keys())[:10]}")
        title = str(path.stem)
        lines.append(str(data))

    full_text = "\n".join(line for line in lines if line is not None)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", full_text) if p.strip()]

    return {
        "text": full_text,
        "title": title if "title" in dir() else path.stem,  # type: ignore[name-defined]
        "headings": headings,
        "tables_count": tables_count,
        "list_items_count": list_items_count,
        "paragraphs": paragraphs,
        "word_count": len(full_text.split()),
        "char_count": len(full_text),
        "extraction_warnings": warnings,
    }


def _empty_result(warning: str) -> dict:
    return {
        "text": "",
        "title": "",
        "headings": [],
        "tables_count": 0,
        "list_items_count": 0,
        "paragraphs": [],
        "word_count": 0,
        "char_count": 0,
        "extraction_warnings": [warning],
    }


# ─────────────────────────────────────────────────────────────────────────────
# PDF extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf(path: Path) -> dict:
    """
    Extract text from a PDF file.

    Tries pypdf first; falls back to pdfplumber if pypdf returns suspiciously
    short text (a sign of extraction failure for complex Hebrew PDFs).

    Also runs a Hebrew text-direction sanity check and logs a warning if the
    text appears to be reversed (a common artifact in some Hebrew PDF renderers).
    """
    warnings: List[str] = []
    path = Path(path)

    text_pypdf, pypdf_warnings = _extract_with_pypdf(path)
    text_pdfplumber = None

    # If pypdf gives very little text, try pdfplumber as a fallback
    if len(text_pypdf.split()) < 50:
        warnings.append(
            f"pypdf returned only {len(text_pypdf.split())} words; trying pdfplumber fallback"
        )
        text_pdfplumber, plumber_warnings = _extract_with_pdfplumber(path)
        warnings.extend(plumber_warnings)

        # Choose the richer extraction
        if len(text_pdfplumber.split()) > len(text_pypdf.split()):
            text = text_pdfplumber
            warnings.append("Using pdfplumber output (richer than pypdf)")
        else:
            text = text_pypdf
            warnings.append("Keeping pypdf output (pdfplumber not better)")
    else:
        text = text_pypdf

    warnings.extend(pypdf_warnings)

    # Hebrew direction sanity check
    direction_ok, direction_msg = _check_hebrew_direction(text)
    if not direction_ok:
        warnings.append(f"Hebrew direction warning: {direction_msg}")
        logger.warning("PDF direction issue in %s: %s", path.name, direction_msg)

    # Structure metadata (headings heuristic: lines in ALL_CAPS or starting with digit+dot)
    headings = _infer_pdf_headings(text)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    return {
        "text": text,
        "title": path.stem,
        "headings": headings,
        "tables_count": 0,  # PDFs: tables are not reliably detectable without layout analysis
        "list_items_count": text.count("\n-") + text.count("\n•"),
        "paragraphs": paragraphs,
        "word_count": len(text.split()),
        "char_count": len(text),
        "extraction_warnings": warnings,
    }


def _extract_with_pypdf(path: Path) -> Tuple[str, List[str]]:
    """Extract text from PDF using pypdf."""
    warnings: List[str] = []
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        pages_text = []
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            pages_text.append(page_text)
        return "\n\n".join(pages_text), warnings
    except Exception as exc:
        warnings.append(f"pypdf extraction error: {exc}")
        return "", warnings


def _extract_with_pdfplumber(path: Path) -> Tuple[str, List[str]]:
    """Extract text from PDF using pdfplumber (better for complex layouts)."""
    warnings: List[str] = []
    try:
        import pdfplumber
        pages_text = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                pages_text.append(page_text)
        return "\n\n".join(pages_text), warnings
    except Exception as exc:
        warnings.append(f"pdfplumber extraction error: {exc}")
        return "", warnings


def _check_hebrew_direction(text: str) -> Tuple[bool, str]:
    """
    Heuristic check: count how many common Hebrew function-words appear in
    their correct left-to-right Unicode order within the string.

    If fewer than MIN_DIRECTION_WORDS_FOR_CONFIDENCE are found, the text
    may be reversed (a common artifact in Hebrew PDF extraction).

    Returns (is_ok: bool, message: str).
    """
    found = sum(1 for word in HEBREW_DIRECTION_WORDS if word in text)
    total_hebrew_chars = sum(1 for c in text if "\u0590" <= c <= "\u05FF")

    if total_hebrew_chars < 100:
        return True, "Too little Hebrew text to assess direction"

    if found >= MIN_DIRECTION_WORDS_FOR_CONFIDENCE:
        return True, f"Found {found}/{len(HEBREW_DIRECTION_WORDS)} expected Hebrew function-words"

    return (
        False,
        f"Only {found}/{len(HEBREW_DIRECTION_WORDS)} expected Hebrew function-words found. "
        "Text may be reversed or have encoding issues.",
    )


def _infer_pdf_headings(text: str) -> List[Dict]:
    """
    Heuristic heading detection for plain-text PDF output.

    Treats as headings:
    - Lines that are ALL-CAPS Hebrew/English (likely section titles)
    - Lines matching numbered section pattern: "1.", "1.1", "א.", etc.
    """
    headings = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or len(stripped) < 3:
            continue
        # Numbered section (e.g. "1.", "1.2", "א.")
        if re.match(r"^[\d\u05D0-\u05EA]{1,3}\.[\d\.]?\s+\S", stripped):
            headings.append({"level": 2, "text": stripped})
        # Short ALL-CAPS line (likely a heading in Latin script)
        elif stripped.isupper() and 3 <= len(stripped) <= 80:
            headings.append({"level": 2, "text": stripped})

    return headings
