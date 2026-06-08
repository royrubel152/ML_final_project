"""
Replace the 6 bschool sections in scraped_content.json with Bar's clean EDA data.
Run once before rebuilding the index.

Bar's repo: https://github.com/royrubel152/ML_final_project/tree/main/RAG-EDA/data/clean
Files are named sha1(url).hexdigest() + ".txt"
"""

import hashlib
import json
import os
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

RAW_BASE = "https://raw.githubusercontent.com/royrubel152/ML_final_project/main/RAG-EDA/data/clean"
CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scraped_content.json")
HASH_FILE  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "content_hash.txt")

BSCHOOL_SOURCES = {
    "https://bschool.huji.ac.il/mba/admittance":           "קבלה",
    "https://bschool.huji.ac.il/mba/specializations":      "התמחויות",
    "https://bschool.huji.ac.il/ptorim":                   "פטורים",
    "https://bschool.huji.ac.il/Accelerated-study-track":  "מסלול מואץ",
    "https://bschool.huji.ac.il/Accelerated-study-track-all": "מסלול מואץ - כל הקורסים",
    "https://bschool.huji.ac.il/sites/default/files/business/files/academic_regulations_for_masters_programs_01.pdf": "תקנון אקדמי (PDF)",
}


def sha1(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()


def fetch_bar_txt(url: str) -> str:
    h = sha1(url)
    raw_url = f"{RAW_BASE}/{h}.txt"
    print(f"  Fetching {raw_url}")
    try:
        with urllib.request.urlopen(raw_url, timeout=15) as r:
            return r.read().decode("utf-8")
    except Exception as e:
        print(f"  WARN: Could not fetch {url} — {e}")
        return None


def parse_sections(content: str) -> list[dict]:
    """Parse === Name === / URL: ... / text blocks."""
    sections = []
    name = url = None
    lines = []
    for line in content.splitlines():
        if line.startswith("=== ") and line.endswith(" ==="):
            if name is not None:
                sections.append({"name": name, "url": url or "", "text": "\n".join(lines)})
            name = line[4:-4]
            url = None
            lines = []
        elif line.startswith("URL: "):
            url = line[5:].strip()
        else:
            lines.append(line)
    if name is not None:
        sections.append({"name": name, "url": url or "", "text": "\n".join(lines)})
    return sections


def reassemble(sections: list[dict]) -> str:
    parts = []
    for s in sections:
        parts.append(f"=== {s['name']} ===")
        if s["url"]:
            parts.append(f"URL: {s['url']}")
        parts.append(s["text"])
        parts.append("")
    return "\n".join(parts)


def main():
    print(f"[patch] Loading {CACHE_FILE}")
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)

    sections = parse_sections(cache["content"])
    print(f"[patch] Found {len(sections)} sections")

    replaced = 0
    for s in sections:
        if s["url"] in BSCHOOL_SOURCES:
            source_name = BSCHOOL_SOURCES[s["url"]]
            print(f"\n[patch] Replacing '{source_name}' with Bar's clean version")
            bar_text = fetch_bar_txt(s["url"])
            if bar_text is not None:
                old_len = len(s["text"])
                s["text"] = bar_text
                print(f"  {old_len} chars → {len(bar_text)} chars")
                replaced += 1
            else:
                print(f"  Keeping original text for '{source_name}'")

    print(f"\n[patch] Replaced {replaced}/6 bschool sources")

    new_content = reassemble(sections)
    cache["content"] = new_content
    cache["char_count"] = len(new_content)
    cache["timestamp"] = time.time()
    cache["bar_patch"] = True

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    print(f"[patch] Saved {len(new_content)} chars to {CACHE_FILE}")

    # Delete content hash so the server rebuilds the FAISS index on next start
    if os.path.exists(HASH_FILE):
        os.remove(HASH_FILE)
        print("[patch] Deleted content_hash.txt → index will rebuild on server restart")

    print("[patch] Done. Restart the server to rebuild the index.")


if __name__ == "__main__":
    main()
