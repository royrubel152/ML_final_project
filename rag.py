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
CHUNK_SIZE   = 800
CHUNK_OVERLAP = 150
TOP_K        = 5
MIN_SCORE    = 0.45


# ── Chunking ──────────────────────────────────────────────────────

def _parse_sections(content: str) -> list[dict]:
    """Split scraped content string into per-source sections."""
    sections, name, url, lines = [], None, None, []
    for line in content.splitlines():
        if line.startswith("=== ") and line.endswith(" ==="):
            if name and lines:
                sections.append({"source": name, "url": url or "", "text": "\n".join(lines)})
            name, url, lines = line[4:-4], None, []
        elif line.startswith("URL: "):
            url = line[5:].strip()
        else:
            lines.append(line)
    if name and lines:
        sections.append({"source": name, "url": url or "", "text": "\n".join(lines)})
    return sections


def chunk_content(content: str) -> list[dict]:
    """Return flat list of overlapping chunks with source metadata."""
    all_chunks = []
    for section in _parse_sections(content):
        text, source, url = section["text"], section["source"], section["url"]
        start = 0
        while start < len(text):
            piece = text[start:start + CHUNK_SIZE].strip()
            if len(piece) > 80:
                all_chunks.append({
                    "text": piece,
                    "source": source,
                    "url": url,
                    "chunk_id": len(all_chunks),
                })
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return all_chunks


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


# ── Retrieval ─────────────────────────────────────────────────────

def retrieve(query: str, index, chunks: list[dict], top_k: int = TOP_K) -> list[dict]:
    q_vec = np.array([embed_query(query)], dtype="float32")
    faiss.normalize_L2(q_vec)
    scores, indices = index.search(q_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and float(score) >= MIN_SCORE:
            chunk = dict(chunks[idx])
            chunk["score"] = round(float(score), 3)
            results.append(chunk)
    return results


# ── Prompt builder ────────────────────────────────────────────────

def build_rag_prompt(question: str, chunks: list[dict]) -> str:
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
    return (
        f"ענה על השאלה הבאה בהתבסס ONLY על המקטעים הבאים.\n"
        f"אם התשובה לא מופיעה — אמור: 'המידע אינו זמין במקורות הרשמיים.'\n\n"
        f"מקטעים רלוונטיים:\n{context}\n\n"
        f"שאלה: {question}"
    )
