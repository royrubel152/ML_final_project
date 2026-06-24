"""
chunkers.py — chunking strategies + metadata enrichment + URL-preserving dedup.

Strategies (Stage A arms):
  baseline      : reuse the production header-aware chunker in rag.chunk_content.
  breadcrumb    : baseline chunks, each prefixed with a structural breadcrumb
                  (source / specialization) so a split chunk still carries its
                  context for the embedder.
  per_group     : one chunk per year/section header block (credit-requirement
                  context never splits across chunks).
  parent_child  : small children (~280 chars) are the retrieval units; the full
                  parent block is carried in ``context_text`` for generation.

Every unit is a dict with at least:
  text          -> the text that is embedded and retrieved on
  context_text  -> the text shown to the generator (== text, except parent_child)
  source, url, chunk_id (str), plus enrichment metadata.

URL-preserving dedup removes duplicate *text* but keeps every source URL in
``dup_urls`` so recall/citation is never silently dropped (EDA near-duplicate
specialization pages, section 12). Heavy deps are imported lazily.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_CONTENT_FILE = ROOT / "scraped_content.json"

# ── Domain regexes (course codes, credits, semester) ────────────────────────
_COURSE_CODE = re.compile(r"\b\d{5}\b")
_CREDITS = re.compile(r"(\d+(?:\.\d+)?)\s*נ[\"״׳']?ז")
_SEMESTER = re.compile(r"סמסטר\s*[א-ת]|סמסטר\s*\d|שנתי")
_SECTION_HDR = re.compile(r"(?:^|\n)(שנה\s+\d+\s*[-–][^\n]*)")


def preprocess_hebrew(text: str) -> str:
    """NFC normalization + nikud strip + whitespace collapse (matches EDA)."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u0591-\u05C7]", "", text)  # Hebrew diacritics / nikud
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def load_raw_content() -> str:
    """Load the concatenated scraped content string (key 'content')."""
    import json
    with open(RAW_CONTENT_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("content", "")


# ── Enrichment ──────────────────────────────────────────────────────────────

def enrich(chunk: dict) -> dict:
    """Attach structured metadata used for filtering and reporting."""
    text = chunk.get("text", "")
    source = chunk.get("source", "")
    url = chunk.get("url", "")

    chunk["course_codes"] = sorted(set(_COURSE_CODE.findall(text)))
    chunk["credits"] = sorted(set(_CREDITS.findall(text)))
    chunk["semester"] = bool(_SEMESTER.search(text))
    chunk["source_type"] = _classify_source(url, source)
    spec_code, spec_name = _parse_spec(url, source)
    chunk["spec_code"] = spec_code
    chunk["spec_name"] = spec_name
    return chunk


def _classify_source(url: str, source: str) -> str:
    if "specialization" in url:
        return "shnaton_specialization"
    if "roadmap" in url:
        return "shnaton_roadmap"
    if "bschool" in url:
        return "bschool"
    if source.startswith("FAQ"):
        return "faq"
    return "other"


def _parse_spec(url: str, source: str) -> tuple[str | None, str | None]:
    m = re.search(r"/specialization/(\d+)", url)
    code = m.group(1) if m else None
    name = source.strip() or None
    return code, name


# ── URL-preserving dedup ────────────────────────────────────────────────────

def dedup_preserve_urls(chunks: list[dict]) -> list[dict]:
    """Drop exact-text duplicates, merging their URLs into ``dup_urls``."""
    seen: dict[str, dict] = {}
    out: list[dict] = []
    for ch in chunks:
        key = ch["text"]
        if key in seen:
            keeper = seen[key]
            urls = set(keeper.get("dup_urls", [keeper.get("url", "")]))
            urls.add(ch.get("url", ""))
            keeper["dup_urls"] = sorted(u for u in urls if u)
        else:
            ch["dup_urls"] = [ch.get("url", "")] if ch.get("url") else []
            seen[key] = ch
            out.append(ch)
    return out


def _finalize(chunks: list[dict], strategy: str, enrich_meta: bool) -> list[dict]:
    """Assign stable ids, set context_text, enrich, dedup."""
    for i, ch in enumerate(chunks):
        ch.setdefault("context_text", ch["text"])
        ch["chunk_id"] = f"{strategy}_{i}"
        if enrich_meta:
            enrich(ch)
    return dedup_preserve_urls(chunks)


# ── Strategies ──────────────────────────────────────────────────────────────

def _baseline_chunks(content: str) -> list[dict]:
    """Production header-aware chunks from rag.chunk_content."""
    import rag
    chunks = rag.chunk_content(content)
    for c in chunks:
        c["text"] = preprocess_hebrew(c["text"])
    return chunks


def _breadcrumb_chunks(content: str) -> list[dict]:
    """Baseline chunks, each prefixed with a 'source > spec' breadcrumb."""
    chunks = _baseline_chunks(content)
    for c in chunks:
        code, name = _parse_spec(c.get("url", ""), c.get("source", ""))
        crumbs = [x for x in (c.get("source", ""), name) if x]
        breadcrumb = " > ".join(dict.fromkeys(crumbs))  # de-dup, preserve order
        if breadcrumb:
            c["text"] = f"[{breadcrumb}]\n{c['text']}"
    return chunks


def _per_group_chunks(content: str) -> list[dict]:
    """One chunk per year/section header block (no cross-section bleed)."""
    import rag
    chunks: list[dict] = []
    for section in rag._parse_sections(content):
        text = preprocess_hebrew(section["text"])
        blocks = _split_on_headers(text)
        for block in blocks:
            if len(block) > 80:
                chunks.append({
                    "text": block,
                    "source": section["source"],
                    "url": section["url"],
                })
    return chunks


def _split_on_headers(text: str) -> list[str]:
    parts = _SECTION_HDR.split(text)
    blocks: list[str] = []
    if parts[0].strip():
        blocks.append(parts[0].strip())
    i = 1
    while i < len(parts) - 1:
        header, body = parts[i].strip(), parts[i + 1].strip()
        blocks.append(f"{header}\n{body}" if body else header)
        i += 2
    return blocks


def _parent_child_chunks(content: str, child_size: int = 280) -> list[dict]:
    """Children are retrieval units; full parent block is carried as context."""
    parents = _baseline_chunks(content)
    units: list[dict] = []
    for parent in parents:
        ptext = parent["text"]
        start = 0
        while start < len(ptext):
            piece = ptext[start:start + child_size].strip()
            if len(piece) >= 60:
                units.append({
                    "text": piece,
                    "context_text": ptext,   # generation sees the full parent
                    "source": parent.get("source", ""),
                    "url": parent.get("url", ""),
                })
            start += child_size
    return units


_STRATEGIES = {
    "baseline": _baseline_chunks,
    "breadcrumb": _breadcrumb_chunks,
    "per_group": _per_group_chunks,
    "parent_child": _parent_child_chunks,
}


def build_chunks(strategy: str, content: str | None = None, enrich_meta: bool = True) -> list[dict]:
    """Build the retrieval units for a chunking strategy (enriched + deduped)."""
    if strategy not in _STRATEGIES:
        raise ValueError(f"unknown chunking strategy '{strategy}'. Choose from {list(_STRATEGIES)}.")
    if content is None:
        content = load_raw_content()
    chunks = _STRATEGIES[strategy](content)
    return _finalize(chunks, strategy, enrich_meta)
