"""
Full EDA — what data do we actually have?
Organized by: programs, specializations, courses, and content coverage.
Run: python eda.py
"""
import json, re, sys
from collections import defaultdict, Counter
sys.stdout.reconfigure(encoding="utf-8")

with open("data/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

COURSE_RE   = re.compile(r"-\s+(5\d{4}):\s+([^\(\n]+)\s*\((\d+)\s+נ\"ז,\s*([^\)]+)\)")
LECTURE_RE  = re.compile(r"מרצה:\s*([^\|]+)")
SCHEDULE_RE = re.compile(r"מפגש:\s*([^\|]+)")
GRADE_RE    = re.compile(r"מבחן בכתב (\d+)%")

def extract_courses(text):
    courses = []
    for m in COURSE_RE.finditer(text):
        code, name, credits, semester = m.group(1), m.group(2).strip(), m.group(3), m.group(4).strip()
        # Find lecturer for this course (next מרצה: after course line)
        pos = m.start()
        sub = text[pos:pos+600]
        lec = LECTURE_RE.search(sub)
        sch = SCHEDULE_RE.search(sub)
        grade = GRADE_RE.search(sub)
        courses.append({
            "code": code,
            "name": name,
            "credits": credits,
            "semester": semester,
            "lecturer": lec.group(1).strip() if lec else None,
            "schedule": sch.group(1).strip() if sch else None,
            "has_grade": bool(grade),
        })
    return courses

# Group chunks by source
by_source = defaultdict(list)
for c in chunks:
    by_source[c["source"]].append(c)

sep = "=" * 65

# ─────────────────────────────────────────────────────────────
print(sep)
print("  MBA ADVISOR — FULL DATA INVENTORY")
print(sep)
print(f"  Total chunks: {len(chunks)} | Unique sources: {len(by_source)}")

# ─────────────────────────────────────────────────────────────
print(f"\n{sep}")
print("  SECTION 1 — MBA PROGRAMS (roadmaps)")
print(sep)

PROGRAM_SRCS = [s for s in by_source if "התמחות" not in s and s not in ("התמחויות", "פטורים", "קבלה", "תקנון אקדמי (PDF)")]
for src in sorted(PROGRAM_SRCS):
    all_text = "\n".join(c["text"] for c in by_source[src])
    courses = extract_courses(all_text)
    unique = {c["code"]: c for c in courses}
    mandatory = [c for c in unique.values() if any(
        "חובה" in ch["text"][:80] and c["code"] in ch["text"]
        for ch in by_source[src]
    )]
    url = by_source[src][0].get("url", "")
    print(f"\n  📋 {src}")
    print(f"     URL:            {url}")
    print(f"     Chunks:         {len(by_source[src])}")
    print(f"     Unique courses: {len(unique)}")
    if unique:
        has_lec   = sum(1 for c in unique.values() if c["lecturer"])
        has_sched = sum(1 for c in unique.values() if c["schedule"])
        has_grade = sum(1 for c in unique.values() if c["has_grade"])
        print(f"     With lecturer:  {has_lec}/{len(unique)}")
        print(f"     With schedule:  {has_sched}/{len(unique)}")
        print(f"     With grades:    {has_grade}/{len(unique)}")

# ─────────────────────────────────────────────────────────────
print(f"\n{sep}")
print("  SECTION 2 — SPECIALIZATIONS")
print(sep)

spec_srcs = [s for s in by_source if "התמחות" in s]
primary   = sorted([s for s in spec_srcs if "ראשית" in s])
secondary = sorted([s for s in spec_srcs if "משנית" in s])

def show_spec(src):
    all_text = "\n".join(c["text"] for c in by_source[src])
    courses = extract_courses(all_text)
    unique = {c["code"]: c for c in courses}
    has_lec   = sum(1 for c in unique.values() if c["lecturer"])
    has_sched = sum(1 for c in unique.values() if c["schedule"])
    url = by_source[src][0].get("url", "")
    label = "✅" if len(unique) >= 5 else "⚠️ "
    print(f"\n  {label} {src}")
    print(f"     Chunks: {len(by_source[src])} | Courses: {len(unique)} | Lecturer: {has_lec}/{len(unique)} | Schedule: {has_sched}/{len(unique)}")
    # List courses
    for c in list(unique.values())[:8]:
        lec = c["lecturer"].split("|")[0][:30] if c["lecturer"] else "—"
        sch = c["schedule"][:25] if c["schedule"] else "—"
        print(f"       {c['code']}  {c['name'][:35]:<35}  {c['credits']} נ\"ז  {lec}")
    if len(unique) > 8:
        print(f"       ... and {len(unique)-8} more")

print("\n  ── PRIMARY SPECIALIZATIONS (ראשית) ──")
for src in primary:
    show_spec(src)

print(f"\n  ── SECONDARY SPECIALIZATIONS (משנית) ──")
for src in secondary:
    show_spec(src)

# ─────────────────────────────────────────────────────────────
print(f"\n{sep}")
print("  SECTION 3 — REGULATORY & ADMISSION CONTENT")
print(sep)

for src in ["תקנון אקדמי (PDF)", "פטורים", "קבלה"]:
    if src in by_source:
        all_text = "\n".join(c["text"] for c in by_source[src])
        words = len(all_text.split())
        print(f"\n  📄 {src}")
        print(f"     Chunks: {len(by_source[src])} | Words: ~{words}")
        print(f"     Preview: {all_text[:200].strip()!r}")

# ─────────────────────────────────────────────────────────────
print(f"\n{sep}")
print("  SECTION 4 — OVERALL COURSE INVENTORY")
print(sep)

all_courses = {}
for c in chunks:
    for course in extract_courses(c["text"]):
        code = course["code"]
        if code not in all_courses:
            all_courses[code] = {**course, "source": c["source"]}
        elif not all_courses[code]["lecturer"] and course["lecturer"]:
            all_courses[code]["lecturer"] = course["lecturer"]

print(f"\n  Total unique courses across all sources: {len(all_courses)}")
has_lec   = sum(1 for c in all_courses.values() if c["lecturer"])
has_sched = sum(1 for c in all_courses.values() if c["schedule"])
has_grade = sum(1 for c in all_courses.values() if c["has_grade"])
print(f"  With lecturer info:   {has_lec}/{len(all_courses)} ({100*has_lec//len(all_courses)}%)")
print(f"  With schedule:        {has_sched}/{len(all_courses)} ({100*has_sched//len(all_courses)}%)")
print(f"  With grade breakdown: {has_grade}/{len(all_courses)} ({100*has_grade//len(all_courses)}%)")

# Missing lecturer
missing_lec = [c for c in all_courses.values() if not c["lecturer"]]
if missing_lec:
    print(f"\n  Courses missing lecturer ({len(missing_lec)}):")
    for c in missing_lec[:10]:
        print(f"    {c['code']}: {c['name']}")
    if len(missing_lec) > 10:
        print(f"    ... and {len(missing_lec)-10} more")

# ─────────────────────────────────────────────────────────────
print(f"\n{sep}")
print("  SECTION 5 — DATA GAPS & QUALITY ISSUES")
print(sep)

# Short chunks
short = [c for c in chunks if len(c["text"]) < 150]
print(f"\n  Short chunks (<150 chars): {len(short)}")
for c in short[:5]:
    print(f"    [{c['source']}] {c['text'][:80]!r}")

# Garbled text (reversed Hebrew — PDF issue)
garbled = [c for c in chunks if re.search(r"[ץף][^֐-׿]{0,5}[ךמן]", c["text"])]
print(f"\n  Potentially garbled/reversed text chunks: {len(garbled)}")
for c in garbled[:3]:
    print(f"    [{c['source']}] {c['text'][:80]!r}")

# Near-duplicates
seen = Counter(c["text"][:120] for c in chunks)
dups = [(k, v) for k, v in seen.items() if v > 1]
print(f"\n  Near-duplicate chunk groups: {len(dups)}")

print(f"\n{sep}")
print("  EDA COMPLETE")
print(sep)
