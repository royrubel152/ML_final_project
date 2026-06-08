"""
RAG module — chunking, embedding, FAISS index, retrieval.
Embedding model: gemini-embedding-001 (dim=3072, Hebrew support)
Vector store: FAISS IndexFlatIP (cosine similarity via L2-normalized vectors)
"""

import os
import json
import hashlib
import numpy as np
import faiss
import google.generativeai as genai

CHUNKS_FILE  = os.path.join("data", "chunks.json")
INDEX_FILE   = os.path.join("data", "faiss_index.bin")
HASH_FILE    = os.path.join("data", "content_hash.txt")
EMBED_MODEL  = "models/gemini-embedding-001"
CHUNK_SIZE   = 900   # max chars for a single chunk when fallback-splitting long sections
CHUNK_OVERLAP = 100  # overlap only used when a semantic section exceeds CHUNK_SIZE
TOP_K        = 5
MIN_SCORE    = 0.40

# Matches year/section headers in the Shnaton roadmap and specialization data.
# Examples: "שנה 1 - חובה", "שנה 2 - בחירה", "שנה 1 - חובת בחירה סמינרים"
import re as _re
_SECTION_HDR = _re.compile(
    r"(?:^|\n)(שנה\s+\d+\s*[-–]\s*(?:חובה|בחירה|חובת בחירה)[^\n]*)",
)

# Matches section-level headers in bschool regulatory/exemptions content.
# Used only for the פטורים source to split on logical section boundaries.
_REGULATORY_HDR = _re.compile(
    r"(?:^|\n)((?:פטור מ|מסלול הגשה|תנאי|נוהל|הגדרת|דרישות)[^\n]{5,60})",
)

# bschool navigation boilerplate — present on every bschool HTML page.
# Shnaton structural headings are legitimate content and must NOT be stripped.
_BSCHOOL_NAV = _re.compile(
    r"^(דילוג לתוכן העיקרי|ניגודיות צבעים גבוהה|תפריט ראשי"
    r"|אתר האוניברסיטה העברית|נגישות|English)\s*$",
    _re.MULTILINE,
)

# Hebrew morphological prefixes used for query expansion.
_HEB_PREFIXES = ("ב", "ל", "ה", "ש", "ו", "מ", "כ")


# ── Chunking ──────────────────────────────────────────────────────

def _expand_query(query: str) -> str:
    """
    Expand a Hebrew query with prefix variants (ב/ל/ה/ש/ו/מ/כ) of each word.
    Handles 1,693 prefix-variant pairs found in the corpus — e.g. a query for
    "פינטק" also becomes "בפינטק", "לפינטק" so retrieval doesn't miss prefixed forms.
    Runtime-only: no re-embedding required.
    """
    words = query.split()
    expanded = list(words)
    for w in words:
        if len(w) >= 3:
            for p in _HEB_PREFIXES:
                variant = p + w
                if variant not in expanded:
                    expanded.append(variant)
    return " ".join(expanded)


def _parse_sections(content: str) -> list[dict]:
    """Split scraped content string into per-source sections."""
    sections, name, url, lines = [], None, None, []
    for line in content.splitlines():
        if line.startswith("=== ") and line.endswith(" ==="):
            if name and lines:
                raw_text = "\n".join(lines)
                # Strip bschool nav boilerplate (not Shnaton structural headings)
                if "bschool" in (url or ""):
                    raw_text = _BSCHOOL_NAV.sub("", raw_text).strip()
                sections.append({"source": name, "url": url or "", "text": raw_text})
            name, url, lines = line[4:-4], None, []
        elif line.startswith("URL: "):
            url = line[5:].strip()
        else:
            lines.append(line)
    if name and lines:
        raw_text = "\n".join(lines)
        if "bschool" in (url or ""):
            raw_text = _BSCHOOL_NAV.sub("", raw_text).strip()
        sections.append({"source": name, "url": url or "", "text": raw_text})
    return sections


def _split_on_headers(text: str) -> list[str]:
    """
    Split text on Shnaton year-section headers (שנה X - חובה/בחירה).
    Each returned block starts with its header (if any) and contains the
    courses belonging to that section — no bleed between sections.
    """
    parts = _SECTION_HDR.split(text)
    # parts alternates: [pre_header_text, header1, body1, header2, body2, ...]
    blocks = []
    if parts[0].strip():
        blocks.append(parts[0].strip())
    i = 1
    while i < len(parts) - 1:
        header = parts[i].strip()
        body   = parts[i + 1].strip()
        block  = f"{header}\n{body}" if body else header
        if block:
            blocks.append(block)
        i += 2
    return blocks


def _split_on_headers_with_pattern(text: str, pattern) -> list[str]:
    """Generic version of _split_on_headers for any compiled regex pattern."""
    parts = pattern.split(text)
    blocks = []
    if parts[0].strip():
        blocks.append(parts[0].strip())
    i = 1
    while i < len(parts) - 1:
        header = parts[i].strip()
        body   = parts[i + 1].strip()
        block  = f"{header}\n{body}" if body else header
        if block:
            blocks.append(block)
        i += 2
    return blocks


def _fixed_split(text: str, source: str, url: str, all_chunks: list):
    """Fallback: split a long block with overlapping fixed-size windows.

    Continuation chunks (all except the first) get the block's first line
    prepended so that the section header (e.g. 'שנה 1 - חובה (12 נ"ז)') is
    present in every chunk and the embedding model can match retrieval queries
    even when a long mandatory-course block is split across multiple windows.
    """
    # Use the first ~200 chars as context prefix (includes section header + הערה note).
    # This ensures continuation chunks carry the specialization name, e.g.
    # "אנליטיקה של נתוני עתק", so the embedding matches the right specialization query.
    context_prefix = text[:200].strip()
    start = 0
    chunk_num = 0
    while start < len(text):
        piece = text[start:start + CHUNK_SIZE].strip()
        if chunk_num > 0 and context_prefix:
            piece = f"{context_prefix}\n...\n{piece}"
        if len(piece) > 80:
            all_chunks.append({
                "text": piece,
                "source": source,
                "url": url,
                "chunk_id": len(all_chunks),
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP
        chunk_num += 1


def chunk_content(content: str) -> list[dict]:
    """
    Semantic chunking: split each source section on year/type headers
    (שנה 1 - חובה, שנה 1 - בחירה, etc.) so mandatory and elective courses
    never share a chunk. Blocks that exceed CHUNK_SIZE are further split
    with a small fixed-size window (elective lists can be very long).

    Special handling:
    - פטורים source: split on regulatory section headers (_REGULATORY_HDR)
      because the exemptions page is 20,021 words with 304 RTL/LTR switches.
    - Post-processing: exact-text deduplication removes near-identical chunks
      from מחקרי/עיוני MBA pair (Jaccard 0.969 per Bar's EDA).
    """
    all_chunks = []
    for section in _parse_sections(content):
        text, source, url = section["text"], section["source"], section["url"]

        # Choose header pattern: regulatory for exemptions, semantic for everything else
        if "פטורים" in source:
            blocks = _split_on_headers_with_pattern(text, _REGULATORY_HDR)
        else:
            blocks = _split_on_headers(text)

        for block in blocks:
            if len(block) <= CHUNK_SIZE:
                if len(block) > 80:
                    all_chunks.append({
                        "text": block,
                        "source": source,
                        "url": url,
                        "chunk_id": len(all_chunks),
                    })
            else:
                _fixed_split(block, source, url, all_chunks)

    # P1: Drop exact-text duplicates (handles מחקרי/עיוני near-identical pair).
    # Keep first occurrence (which will be מחקרי if sorted alphabetically by source).
    seen_texts: dict[str, bool] = {}
    merged = []
    for ch in all_chunks:
        if ch["text"] not in seen_texts:
            seen_texts[ch["text"]] = True
            merged.append(ch)
    # Re-assign sequential chunk_ids after dedup
    for i, ch in enumerate(merged):
        ch["chunk_id"] = i
    return merged


# ── Embedding ─────────────────────────────────────────────────────

def _embed(text: str, task: str) -> list[float]:
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type=task,
    )
    return result["embedding"]


def embed_document(text: str) -> list[float]:
    return _embed(text, "retrieval_document")


def embed_query(text: str) -> list[float]:
    return _embed(text, "retrieval_query")


# ── Index management ──────────────────────────────────────────────

def _content_hash(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()


def _index_is_fresh(content: str) -> bool:
    if not all(os.path.exists(f) for f in [CHUNKS_FILE, INDEX_FILE, HASH_FILE]):
        return False
    with open(HASH_FILE, "r") as f:
        return f.read().strip() == _content_hash(content)


def build_index(content: str) -> tuple:
    """Chunk → embed → build FAISS index → save to disk."""
    chunks = chunk_content(content)
    print(f"[rag] Embedding {len(chunks)} chunks (this takes ~1 min)...")

    embeddings = []
    for i, chunk in enumerate(chunks):
        emb = embed_document(chunk["text"])
        embeddings.append(emb)
        if (i + 1) % 10 == 0:
            print(f"[rag] {i + 1}/{len(chunks)} embedded")

    vectors = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    with open(HASH_FILE, "w") as f:
        f.write(_content_hash(content))

    print(f"[rag] Index saved — {len(chunks)} chunks, dim={vectors.shape[1]}")
    return index, chunks


def load_index() -> tuple | None:
    if not all(os.path.exists(f) for f in [CHUNKS_FILE, INDEX_FILE]):
        return None
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"[rag] Loaded existing index ({len(chunks)} chunks)")
    return index, chunks


def get_or_build_index(content: str) -> tuple:
    """Load index from disk if content hasn't changed, else rebuild."""
    if _index_is_fresh(content):
        result = load_index()
        if result:
            return result
    return build_index(content)


# ── Chunk metadata enrichment ─────────────────────────────────────

def enrich_chunks_metadata(chunks: list[dict], spec_map: dict) -> list[dict]:
    """
    Attach spec_code + spec_name to each chunk based on its source field.
    spec_map: {spec_name: spec_code}  (from SHNATON_SPECIALIZATIONS)
    Called once at startup — no re-embedding needed.
    """
    import re as _re
    seminar_re = _re.compile(r"סמינר|seminar", _re.IGNORECASE)
    for chunk in chunks:
        source = chunk.get("source", "")
        chunk.setdefault("spec_code", None)
        chunk.setdefault("spec_name", None)
        for spec_name, spec_code in spec_map.items():
            if spec_name in source:
                chunk["spec_code"] = spec_code
                chunk["spec_name"] = spec_name
                break
        chunk["has_seminar"] = bool(seminar_re.search(chunk["text"]))
    return chunks


# ── Retrieval ─────────────────────────────────────────────────────

def retrieve(query: str, index, chunks: list[dict], top_k: int = TOP_K) -> list[dict]:
    return retrieve_with_context(query, index, chunks, active_spec_code=None, top_k=top_k)


def retrieve_with_context(
    query: str,
    index,
    chunks: list[dict],
    active_spec_code: str = None,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Retrieve chunks with optional specialization boosting.
    When active_spec_code is set, fetches 4× more candidates,
    boosts chunks from the active specialization (+0.20),
    and mildly penalises chunks from other specializations (-0.08).
    """
    # P2: Expand query with Hebrew prefix variants before embedding.
    query = _expand_query(query)

    candidates = min(top_k * 4, len(chunks)) if active_spec_code else top_k

    q_vec = np.array([embed_query(query)], dtype="float32")
    faiss.normalize_L2(q_vec)
    scores, indices = index.search(q_vec, candidates)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        raw = float(score)
        if raw < MIN_SCORE * 0.75:
            continue
        chunk = dict(chunks[idx])
        adjusted = raw
        if active_spec_code:
            cs = chunk.get("spec_code")
            if cs == active_spec_code:
                adjusted += 0.20
            elif cs and cs != active_spec_code:
                adjusted -= 0.08
        chunk["score"] = round(adjusted, 3)
        results.append(chunk)

    results.sort(key=lambda x: x["score"], reverse=True)
    results = [r for r in results if r["score"] >= MIN_SCORE]

    # P1: Runtime dedup — drop chunks whose 120-char token-Jaccard with an already-kept
    # result is ≥ 0.70 (handles מחקרי/עיוני overlap at 0.969).
    # Continuation chunks have a shared context prefix ending with "...\n"; skip past it
    # so dedup keys on unique content rather than the shared preamble.
    seen_prefixes: list[set] = []
    deduped = []
    for r in results:
        text = r["text"]
        marker = "...\n"
        pos = text.find(marker)
        content_start = pos + len(marker) if pos >= 0 else 0
        key = set(text[content_start:content_start + 120].split())
        if not any(len(key & s) / max(len(key | s), 1) >= 0.70 for s in seen_prefixes):
            seen_prefixes.append(key)
            deduped.append(r)
    return deduped[:top_k]


# ── Prompt builder ────────────────────────────────────────────────

def build_rag_prompt(question: str, chunks: list[dict]) -> str:
    return build_rag_prompt_with_context(question, chunks)


def build_rag_prompt_with_context(
    question: str,
    chunks: list[dict],
    active_spec_name: str = None,
    is_correction: bool = False,
) -> str:
    if not chunks:
        return (
            f"שאלה: {question}\n\n"
            "לא נמצאו מקטעים רלוונטיים במקורות הרשמיים. "
            "יש לפנות למזכירות התלמידים."
        )

    parts = []
    for c in chunks:
        parts.append(f"[מקור: {c['source']} | {c['url']}]\n{c['text']}")
    context = "\n\n---\n\n".join(parts)

    spec_block = ""
    if active_spec_name:
        spec_block = (
            f"\n⚠️ הקשר פעיל: ההתמחות הנוכחית בשיחה היא «{active_spec_name}».\n"
            f"ענה אך ורק על מידע הרלוונטי להתמחות זו. "
            f"אל תזכיר קורסים או סמינרים מהתמחויות אחרות.\n"
        )

    correction_block = ""
    if is_correction:
        correction_block = (
            "\n⚠️ המשתמש מציין שחסרה מידע בתשובה הקודמת. "
            "עיין בכל המקטעים והצג את הרשימה המלאה של האופציות הרלוונטיות — "
            "אל תשמיט אף אחת. הודה בקיצור שהתשובה הקודמת הייתה חלקית.\n"
        )

    return (
        f"ענה על השאלה הבאה בהתבסס ONLY על המקטעים הבאים.\n"
        f"אם התשובה לא מופיעה — אמור: 'המידע אינו זמין במקורות הרשמיים.'\n"
        f"{spec_block}"
        f"{correction_block}"
        f"\nמקטעים רלוונטיים:\n{context}\n\n"
        f"שאלה: {question}"
    )
