"""
experiments/retrieval_experiment.py
────────────────────────────────────
Hebrew MBA RAG — 4-Pipeline Retrieval Comparison

Benchmarks 4 retrieval pipelines against the 100-question ground-truth CSV,
evaluating Recall@k (k=1,3,5,10), MRR, and optionally LLM-as-judge context
relevance. Outputs a markdown comparison table + two PNG charts.

Pipelines:
  P1  Baseline   — Fixed-size chunks (500 chars) + MiniLM-L12 + pure dense search
  P2  Production — Semantic chunks (900 chars) + gemini-embedding-001 + query expansion
  P3  Hybrid     — Semantic chunks + E5-large + BM25 RRF fusion + entity boost
  P4  Ultimate   — Hybrid top-20 candidates → bge-reranker-v2-m3 cross-encoder

Usage:
  python experiments/retrieval_experiment.py
  python experiments/retrieval_experiment.py --skip-p2   # skip Gemini API (saves cost)
  python experiments/retrieval_experiment.py --judge      # add LLM context-relevance metric
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

# Block TensorFlow and JAX before any transformers/sentence-transformers import.
# Without this, transformers tries to import TF even when PyTorch is the backend,
# causing a protobuf version crash on this environment.
os.environ.setdefault("USE_TF",  "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import pandas as pd
import faiss

# ── Path setup — project root must be importable ───────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

# ── Config ─────────────────────────────────────────────────────────────────────
CHUNKS_FILE  = ROOT / "data" / "chunks.json"
SCRAPE_FILE  = ROOT / "scraped_content.json"   # lives at project root, not in data/
GT_CSV       = ROOT / "evaluation_framework" / "ground_truth_mba_qa.csv"
CACHE_DIR    = Path(__file__).parent / "cache"
RESULTS_DIR  = Path(__file__).parent / "results"
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

K_VALUES      = [1, 3, 5, 10]
TOP_K_FETCH   = 10   # final results returned to caller
TOP_K_RERANK  = 70   # legacy — not used by retrieve_p4 anymore (kept for P3)

# P4 independent stage parameters
P4_K_PARENTS  = 20   # unique parents passed to cross-encoder (sweet spot: 15-25)
P4_K_FETCH    = 10   # final results returned from P4

HEB_PREFIXES = ("ב", "ל", "ה", "ש", "ו", "מ", "כ")

SPECIALIZATIONS = [
    "פינטק", "שיווק", "אסטרטגיה", "יזמות", "מימון", "ביו-רפואי",
    "אנליטיקה", "מדע המידע", "ניהול פיננסי",
]

# ═══════════════════════════════════════════════════════════════════════════════
# 1. HEBREW PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_hebrew(text: str) -> str:
    """NFC normalization + nikud strip + whitespace collapse."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[֑-ׇ]", "", text)   # Hebrew diacritics / nikud
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ENTITY EXTRACTION (Phase 1 — static analysis over chunk corpus)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_entities(chunks: list[dict]) -> dict:
    """Regex entity extraction across all chunk texts."""
    all_text = " ".join(c["text"] for c in chunks)

    def find(pat: str) -> list[str]:
        return re.findall(pat, all_text)

    course_codes  = list(set(re.findall(r"\b\d{5}\b", all_text)))
    credit_vals   = find(r'\d+(?:\.\d+)?\s*נ[״"ז]{1,2}')
    lecturer_raw  = find(r'(?:מרצה|מלמד)\s*:\s*([^\n,;]{3,30})')
    deadlines     = find(r'עד\s+\d{1,2}\.\d{1,2}\.\d{4}')
    prerequisites = find(r'(?:תנאי קדם|ידע קודם)\s*:\s*([^\n]{3,60})')
    grades        = find(r'ציון\s+\w*\s*\d{2,3}')

    spec_counts = {s: all_text.count(s) for s in SPECIALIZATIONS if all_text.count(s) > 0}

    return {
        "course_codes":    {"count": len(course_codes),  "examples": course_codes[:6]},
        "credit_values":   {"count": len(credit_vals),   "examples": credit_vals[:5]},
        "lecturer_names":  {"count": len(lecturer_raw),  "examples": lecturer_raw[:5]},
        "deadlines":       {"count": len(deadlines),     "examples": deadlines[:5]},
        "prerequisites":   {"count": len(prerequisites), "examples": prerequisites[:4]},
        "grade_thresholds":{"count": len(grades),        "examples": grades[:5]},
        "specializations": {"count": sum(spec_counts.values()), "by_name": spec_counts},
    }


def print_entity_report(entities: dict):
    print("\n" + "═" * 62)
    print("  PHASE 1 — ENTITY EXTRACTION REPORT")
    print("═" * 62)
    labels = {
        "course_codes": "Course Codes (5-digit)",
        "credit_values": "Credit Values (נ\"ז)",
        "lecturer_names": "Lecturer Names",
        "deadlines": "Deadlines",
        "prerequisites": "Prerequisites",
        "grade_thresholds": "Grade Thresholds",
    }
    for key, label in labels.items():
        d = entities[key]
        ex = " | ".join(str(e)[:25] for e in d["examples"][:4])
        print(f"  {label:<28} {d['count']:>4}   e.g. {ex}")

    print(f"\n  Specialization Mentions:")
    for spec, cnt in sorted(entities["specializations"]["by_name"].items(), key=lambda x: -x[1]):
        bar = "█" * min(cnt // 5, 20)
        print(f"    {spec:<20} {cnt:>4}  {bar}")

    print("\n  Entity leverage proposals:")
    print("  • Course codes  → BM25 exact match + 1.3× entity boost in P3/P4")
    print("  • Specializations → score boost already applied in production (rag.py)")
    print("  • Deadlines     → flag as time-sensitive; exclude from long-lived cache")
    print("  • Prerequisites → prerequisite graph for course-planning Q&A")
    print("═" * 62)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CHUNKING
# ═══════════════════════════════════════════════════════════════════════════════

def load_semantic_chunks() -> list[dict]:
    """Pre-built production chunks (header-split, 900 chars, context prefix)."""
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    for c in chunks:
        c["text"] = preprocess_hebrew(c["text"])
    return chunks


def build_fixed_size_chunks(size: int = 500, overlap: int = 50) -> list[dict]:
    """P1 baseline: naive fixed-size chunking from raw scraped content, with URL tracking."""
    raw_path = ROOT / "scraped_content.json"
    with open(raw_path, encoding="utf-8") as f:
        data = json.load(f)
    raw_text = data.get("content", "")

    # Split into per-source sections; each starts with "=== ... ===\nURL: <url>\n"
    section_pat = re.compile(r"=== .+? ===\nURL: (https?://\S+)", re.DOTALL)
    sections: list[tuple[int, str]] = []   # (char_offset, url)
    for m in section_pat.finditer(raw_text):
        sections.append((m.start(), m.group(1)))

    def url_at(pos: int) -> str:
        url = ""
        for offset, u in sections:
            if offset <= pos:
                url = u
            else:
                break
        return url

    chunks = []
    cid = 0
    start = 0
    while start < len(raw_text):
        piece = preprocess_hebrew(raw_text[start:start + size])
        if len(piece) >= 60:
            chunks.append({
                "text":     piece,
                "source":   "scraped",
                "url":      url_at(start),
                "chunk_id": cid,
            })
            cid += 1
        start += size - overlap
    return chunks


def build_parent_child_chunks(
    parent_chunks: list[dict],
    child_size: int = 250,
) -> tuple[list[dict], list[dict]]:
    """
    Small-to-big (parent-child) chunking for P4.

    Each production semantic chunk (900 chars) becomes the PARENT.
    Each parent is further split into CHILD chunks (~250 chars) which are
    embedded and searched. Retrieval finds the best children, then we
    surface the full parent text to the reranker for richer context.

    Returns:
        parents  — same list as parent_chunks (with added 'parent_id' == chunk_id)
        children — smaller chunks with 'parent_id' pointing to their parent
    """
    children: list[dict] = []
    child_id = 0
    for parent in parent_chunks:
        text     = parent["text"]
        pid      = parent["chunk_id"]
        start    = 0
        while start < len(text):
            piece = text[start:start + child_size].strip()
            if len(piece) >= 60:
                children.append({
                    "text":      piece,
                    "source":    parent["source"],
                    "url":       parent["url"],
                    "chunk_id":  child_id,
                    "parent_id": pid,
                })
                child_id += 1
            start += child_size   # no overlap on children — parent provides the context
    return parent_chunks, children


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EMBEDDING & CACHING
# ═══════════════════════════════════════════════════════════════════════════════

def _fingerprint(chunks: list[dict]) -> str:
    import hashlib
    sample = "".join(c["text"][:40] for c in chunks[:30])
    return hashlib.md5(sample.encode()).hexdigest()[:10]


def _load_or_embed(cache_name: str, chunks: list[dict], embed_fn) -> np.ndarray:
    path = CACHE_DIR / f"{cache_name}.npy"
    if path.exists():
        print(f"    [cache] {path.name}")
        return np.load(str(path)).astype("float32")
    vecs = embed_fn(chunks)
    np.save(str(path), vecs)
    return vecs.astype("float32")


def embed_with_minilm(chunks: list[dict], model) -> np.ndarray:
    print(f"    Embedding {len(chunks)} chunks with MiniLM...")
    return model.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    ).astype("float32")


def embed_with_e5(chunks: list[dict], model) -> np.ndarray:
    print(f"    Embedding {len(chunks)} chunks with E5-large...")
    texts = ["passage: " + c["text"] for c in chunks]
    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=8,
    ).astype("float32")


def embed_with_gemini(chunks: list[dict]) -> np.ndarray:
    import google.generativeai as genai
    print(f"    Embedding {len(chunks)} chunks with gemini-embedding-001 (API calls)...")
    vecs = []
    for i, chunk in enumerate(chunks):
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=chunk["text"],
            task_type="retrieval_document",
        )
        vecs.append(result["embedding"])
        if (i + 1) % 100 == 0:
            print(f"      {i+1}/{len(chunks)}")
        time.sleep(0.06)   # ~15 req/s limit
    return np.array(vecs, dtype="float32")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. INDEX BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    vecs = vectors.copy().astype("float32")
    faiss.normalize_L2(vecs)
    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)
    return idx


def build_bm25_index(chunks: list[dict]):
    from rank_bm25 import BM25Okapi
    tokenized = [c["text"].split() for c in chunks]
    return BM25Okapi(tokenized)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. QUERY UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def expand_query_hebrew(query: str) -> str:
    """Add morphological prefix variants for each token (from rag._expand_query)."""
    tokens = query.split()
    expanded = list(tokens)
    for tok in tokens:
        for prefix in HEB_PREFIXES:
            if not tok.startswith(prefix):
                expanded.append(prefix + tok)
    return " ".join(expanded)


def dense_search(q_vec: np.ndarray, index: faiss.IndexFlatIP,
                  chunks: list[dict], top_k: int) -> list[dict]:
    q = q_vec.reshape(1, -1).astype("float32").copy()
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx >= 0:
            r = dict(chunks[idx])
            r["score"] = float(score)
            results.append(r)
    return results


def rrf_fuse(dense_hits: list[dict], bm25_hits: list[dict],
             k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion. Returns (chunk_id, rrf_score) sorted descending."""
    scores: dict[int, float] = {}
    for rank, item in enumerate(dense_hits):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, item in enumerate(bm25_hits):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


# ═══════════════════════════════════════════════════════════════════════════════
# 7. PIPELINE RETRIEVE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def retrieve_p1(query: str, st: dict, top_k: int = TOP_K_FETCH) -> list[dict]:
    """P1: MiniLM + fixed-size chunks, pure cosine search."""
    q_vec = st["model_minilm"].encode([query], normalize_embeddings=True)[0]
    return dense_search(q_vec, st["idx_p1"], st["chunks_p1"], top_k)


def retrieve_p2(query: str, st: dict, top_k: int = TOP_K_FETCH) -> list[dict]:
    """P2: gemini-embedding-001 + semantic chunks + Hebrew prefix expansion."""
    import google.generativeai as genai
    expanded = expand_query_hebrew(query)
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=expanded,
        task_type="retrieval_query",
    )
    q_vec = np.array(result["embedding"], dtype="float32")
    time.sleep(0.07)   # rate limit guard
    return dense_search(q_vec, st["idx_p2"], st["chunks_sem"], top_k)


def retrieve_p3(query: str, st: dict, top_k: int = TOP_K_FETCH) -> list[dict]:
    """P3: E5-large + BM25 RRF fusion + 5-digit course-code entity boost."""
    q_vec = st["model_e5"].encode(
        ["query: " + query], normalize_embeddings=True)[0]

    chunks   = st["chunks_sem"]
    pool_k   = top_k * 3

    dense_hits = dense_search(q_vec, st["idx_p3"], chunks, pool_k)

    bm25_scores = st["bm25"].get_scores(query.split())
    bm25_hits = sorted(
        [{"chunk_id": i, "score": float(bm25_scores[i])} for i in range(len(chunks))],
        key=lambda x: -x["score"],
    )[:pool_k]

    fused = rrf_fuse(dense_hits, bm25_hits)

    # Entity boost: if query contains a 5-digit course code, uprank exact matches
    codes = re.findall(r"\b\d{5}\b", query)
    if codes:
        fused = [
            (cid, score * 1.3 if any(code in chunks[cid]["text"] for code in codes) else score)
            for cid, score in fused
        ]
        fused.sort(key=lambda x: -x[1])

    out = []
    for cid, score in fused[:top_k]:
        r = dict(chunks[cid])
        r["score"] = score
        out.append(r)
    return out


def retrieve_p4(query: str, st: dict,
                k_children: int = 40,
                k_parents: int = P4_K_PARENTS,
                top_k: int = P4_K_FETCH) -> list[dict]:
    """
    P4: Parent-Child retrieval + cross-encoder reranking (small-to-big).

    Stage 1 — search k_children CHILD chunks (~250 chars) via E5-large + BM25 RRF.
              k_children is independent — set large for better recall.
    Stage 2 — map top children → unique PARENT chunks (~900 chars).
              Capped at k_parents to keep the reranker pool clean.
    Stage 3 — cross-encoder reranks k_parents pairs, returns top_k.

    Key insight: k_children >> k_parents is optimal.
      More children = diverse parent coverage.
      Fewer parents = focused, high-precision reranker.
    """
    children   = st["children"]
    parents    = st["chunks_sem"]
    parent_map = {p["chunk_id"]: p for p in parents}

    # ── Stage 1: hybrid search over children ─────────────────────────────────
    expanded = expand_query_hebrew(query)   # also expand for E5 (fix #2)
    q_vec = st["model_e5"].encode(
        ["query: " + expanded], normalize_embeddings=True)[0]

    dense_hits = dense_search(q_vec, st["idx_children"], children, k_children)
    bm25_scores = st["bm25_children"].get_scores(expanded.split())
    bm25_hits = sorted(
        [{"chunk_id": i, "score": float(bm25_scores[i])} for i in range(len(children))],
        key=lambda x: -x["score"],
    )[:k_children]

    fused = rrf_fuse(dense_hits, bm25_hits)

    # Entity boost on children
    codes = re.findall(r"\b\d{5}\b", query)
    if codes:
        fused = [
            (cid, score * 1.3 if any(code in children[cid]["text"] for code in codes) else score)
            for cid, score in fused
        ]
        fused.sort(key=lambda x: -x[1])

    # ── Stage 2: map children → unique parents, capped at k_parents ──────────
    seen_parents: set[int] = set()
    candidate_parents: list[dict] = []
    for cid, _ in fused:
        pid = children[cid]["parent_id"]
        if pid not in seen_parents and pid in parent_map:
            seen_parents.add(pid)
            candidate_parents.append(parent_map[pid])
        if len(candidate_parents) >= k_parents:
            break

    if not candidate_parents:
        return []

    # ── Stage 3: rerank parents with cross-encoder ────────────────────────────
    pairs  = [[query, p["text"]] for p in candidate_parents]
    scores = st["reranker"].predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(candidate_parents, scores), key=lambda x: -float(x[1]))

    out = []
    for p, score in ranked[:top_k]:
        r = dict(p)
        r["score"] = float(score)
        out.append(r)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 8. GROUND TRUTH LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_ground_truth() -> list:
    """Load GT CSV → list of EvalSample. Handles column name mapping."""
    from evaluation_framework.schemas import EvalSample
    df = pd.read_csv(GT_CSV)
    samples = []
    for _, row in df.iterrows():
        sources = [s.strip() for s in str(row["source_urls"]).split(";") if s.strip()]
        samples.append(EvalSample(
            sample_id=str(row["id"]),
            question=str(row["question_he"]),
            ground_truth_answer=str(row["answer_he"]),
            ground_truth_sources=sources,
            category=str(row["category"]),
            source_type="single_turn",
        ))
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# 9. EVALUATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_pipeline(
    label: str,
    retrieve_fn,
    samples: list,
    state: dict,
) -> dict:
    """Run retrieve_fn on all samples; compute Recall@k and MRR."""
    from evaluation_framework.deterministic_retrieval import run_deterministic

    recall_sum = defaultdict(float)
    mrr_sum = 0.0
    cat_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    print(f"\n  [{label}] evaluating {len(samples)} questions...")
    for i, sample in enumerate(samples):
        # Work on a copy so GT fields are never overwritten
        s = copy.copy(sample)

        hits = retrieve_fn(sample.question, state)
        s.retrieved_sources = [h["url"] for h in hits]

        rows = run_deterministic(s, k_values=K_VALUES)
        for row in rows:
            m  = row["metric"]
            sc = row["score_norm"]
            if m.startswith("recall@"):
                recall_sum[m] += sc
                cat_scores[s.category][m].append(sc)
            elif m == "mrr":
                mrr_sum += sc
                cat_scores[s.category]["mrr"].append(sc)

        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(samples)}")

    n = len(samples)
    metrics: dict = {m: v / n for m, v in recall_sum.items()}
    metrics["mrr"] = mrr_sum / n
    metrics["per_category"] = {
        cat: {m: sum(v) / len(v) for m, v in ms.items() if v}
        for cat, ms in cat_scores.items()
    }
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 10. GEMINI LLM JUDGE (optional, --judge flag)
# ═══════════════════════════════════════════════════════════════════════════════

class GeminiJudge:
    name = "gemini-3.0-flash"

    def __init__(self):
        import google.generativeai as genai
        self._model = genai.GenerativeModel("gemini-3.0-flash")

    def score(self, prompt: str):
        from evaluation_framework.schemas import JudgeVerdict
        try:
            resp = self._model.generate_content(prompt)
            text = resp.text
            sm = re.search(r"SCORE:\s*([1-5])", text)
            em = re.search(r"EXPLANATION:\s*(.+)", text, re.DOTALL)
            s = int(sm.group(1)) if sm else 3
            e = em.group(1).strip()[:300] if em else text[:150]
        except Exception as exc:
            s, e = 3, f"Error: {exc}"
        return JudgeVerdict(score=s, explanation=e, judge_name=self.name, metric="")


def run_judge_eval(all_retrieve_fns: dict, samples: list, state: dict) -> dict:
    """Evaluate context_relevance for each pipeline using GeminiJudge."""
    from evaluation_framework import runner

    judge = GeminiJudge()
    cr_scores: dict[str, float] = {}

    for label, retrieve_fn in all_retrieve_fns.items():
        print(f"\n  [Judge/{label}] context_relevance on {len(samples)} questions...")
        scores = []
        for i, sample in enumerate(samples):
            s = copy.copy(sample)
            hits = retrieve_fn(sample.question, state)
            s.retrieved_sources = [h["url"] for h in hits]
            df = runner.run(
                [s], judge=judge,
                metrics=["context_relevance"],
                run_deterministic_metrics=False,
            )
            if len(df):
                scores.append(float(df["score_norm"].mean()))
            time.sleep(1.2)   # 50 req/min safety margin
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(samples)}")
        cr_scores[label] = sum(scores) / len(scores) if scores else 0.0

    return cr_scores


# ═══════════════════════════════════════════════════════════════════════════════
# 11. REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def print_comparison_table(all_results: dict[str, dict]):
    cols = [f"recall@{k}" for k in K_VALUES] + ["mrr"]
    col_labels = ["R@1", "R@3", "R@5", "R@10", "MRR"]

    print("\n" + "═" * 78)
    print("  RETRIEVAL EVALUATION  —  100 GT Questions  (URL-level match)")
    print("═" * 78)
    header = f"  {'Pipeline':<42}" + "".join(f" {h:>6}" for h in col_labels)
    print(header)
    print("  " + "─" * 76)
    for label, metrics in all_results.items():
        row = f"  {label:<42}"
        for col in cols:
            row += f"  {metrics.get(col, 0.0):.3f}"
        print(row)
    print("═" * 78)

    # Per-category MRR table
    all_cats = sorted({cat for m in all_results.values() for cat in m["per_category"]})
    p_labels = list(all_results.keys())
    print("\n  Per-Category  MRR")
    cat_header = f"  {'Category':<32}" + "".join(f"  P{i+1:>4}" for i in range(len(p_labels)))
    print(cat_header)
    print("  " + "─" * (32 + 8 * len(p_labels)))
    for cat in all_cats:
        row = f"  {cat:<32}"
        for metrics in all_results.values():
            v = metrics["per_category"].get(cat, {}).get("mrr", 0.0)
            row += f"  {v:.3f}"
        print(row)


def print_judge_table(cr_scores: dict[str, float]):
    print("\n" + "═" * 60)
    print("  LLM JUDGE  —  Context Relevance  (Gemini-2.5-Flash, 0-1)")
    print("═" * 60)
    for label, score in cr_scores.items():
        bar = "█" * int(score * 20)
        print(f"  {label:<42}  {score:.3f}  {bar}")
    print("═" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_charts(all_results: dict[str, dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLORS  = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    MARKERS = ["o", "s", "^", "D"]

    # ── Chart 1: Overall Recall@k curve ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (label, metrics) in enumerate(all_results.items()):
        ys = [metrics.get(f"recall@{k}", 0.0) for k in K_VALUES]
        ax.plot(K_VALUES, ys,
                color=COLORS[i], marker=MARKERS[i],
                linewidth=2.2, markersize=9,
                label=f"P{i+1}: {label.split(':', 1)[-1].strip()}")
    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel("Recall@k", fontsize=12)
    ax.set_title("Recall@k — MBA Hebrew RAG Retrieval Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.set_xticks(K_VALUES)
    ax.set_ylim(0.0, 1.05)
    path1 = RESULTS_DIR / "recall_at_k.png"
    fig.savefig(str(path1), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Chart 1 saved → {path1}")

    # ── Chart 2: Per-category Recall@k (2×4 subplots) ───────────────────────
    all_cats = sorted({cat for m in all_results.values() for cat in m["per_category"]})
    ncols = 4
    nrows = (len(all_cats) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.5 * nrows), sharey=True)
    axes_flat = axes.flatten() if nrows > 1 else list(axes)

    for ax_i, cat in enumerate(all_cats):
        ax = axes_flat[ax_i]
        for i, (label, metrics) in enumerate(all_results.items()):
            ys = [metrics["per_category"].get(cat, {}).get(f"recall@{k}", 0.0)
                  for k in K_VALUES]
            ax.plot(K_VALUES, ys,
                    color=COLORS[i], marker=MARKERS[i],
                    linewidth=1.8, markersize=7,
                    label=f"P{i+1}")
        ax.set_title(cat.replace("_", " ").title(), fontsize=9)
        ax.set_xticks(K_VALUES)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3, linestyle="--")

    for ax_i in range(len(all_cats), len(axes_flat)):
        axes_flat[ax_i].set_visible(False)

    handles, lbls = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper right", fontsize=10)
    fig.suptitle("Recall@k by Category — MBA Hebrew RAG", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.93, 0.97])
    path2 = RESULTS_DIR / "recall_by_category.png"
    fig.savefig(str(path2), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart 2 saved → {path2}")


# ═══════════════════════════════════════════════════════════════════════════════
# 13. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MBA RAG Retrieval Experiment")
    parser.add_argument("--skip-p2", action="store_true",
                        help="Skip Pipeline 2 (no Gemini API calls — saves cost/time)")
    parser.add_argument("--judge", action="store_true",
                        help="Run Gemini LLM-as-judge context_relevance after main eval")
    args = parser.parse_args()

    print("\n" + "═" * 62)
    print("  MBA HEBREW RAG — RETRIEVAL PIPELINE EXPERIMENT")
    print("═" * 62)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n[DATA]")
    samples    = load_ground_truth()
    chunks_sem = load_semantic_chunks()
    print(f"  GT questions          : {len(samples)}")
    print(f"  Semantic chunks (P2-4): {len(chunks_sem)}")
    print(f"  Chunking strategy     : header-aware semantic split, 900 chars, context prefix on continuations")

    # ── Entity extraction ──────────────────────────────────────────────────────
    entities = extract_entities(chunks_sem)
    print_entity_report(entities)

    # ── Load models ────────────────────────────────────────────────────────────
    print("\n[MODELS]")
    from sentence_transformers import SentenceTransformer, CrossEncoder

    print("  Loading intfloat/multilingual-e5-large ...")
    model_e5 = SentenceTransformer("intfloat/multilingual-e5-large")

    print("  Loading BAAI/bge-reranker-base ...")
    reranker = CrossEncoder("BAAI/bge-reranker-base")

    # ── Build P3 index (E5-large dense + BM25 on semantic chunks) ────────────
    print("\n[PIPELINE 3] E5-Large + BM25 Hybrid (semantic chunks, 900 chars)")
    fp3 = _fingerprint(chunks_sem)
    vecs_p3 = _load_or_embed(f"p3_e5_{fp3}", chunks_sem,
                              lambda c: embed_with_e5(c, model_e5))
    idx_p3  = build_faiss_index(vecs_p3)
    bm25    = build_bm25_index(chunks_sem)

    # ── Build P4 parent-child index ───────────────────────────────────────────
    print("\n[PIPELINE 4] Parent-Child: search children (~250 chars) → rerank parents (~900 chars)")
    _, children = build_parent_child_chunks(chunks_sem, child_size=250)
    print(f"  Parent chunks : {len(chunks_sem)}")
    print(f"  Child chunks  : {len(children)}")
    fpc = _fingerprint(children)
    vecs_children = _load_or_embed(f"p4_children_e5_{fpc}", children,
                                    lambda c: embed_with_e5(c, model_e5))
    idx_children  = build_faiss_index(vecs_children)
    bm25_children = build_bm25_index(children)

    # ── Shared state object ────────────────────────────────────────────────────
    state = {
        "model_e5":     model_e5,
        "reranker":     reranker,
        "chunks_sem":   chunks_sem,
        "idx_p3":       idx_p3,
        "bm25":         bm25,
        "children":     children,
        "idx_children": idx_children,
        "bm25_children":bm25_children,
    }

    # ── Evaluations ────────────────────────────────────────────────────────────
    print("\n[EVALUATION]")
    all_results: dict[str, dict] = {}

    if not args.skip_p2:
        print("\n[PIPELINE 2] Gemini-embedding-001 + Semantic Chunks")
        from dotenv import load_dotenv
        import google.generativeai as genai
        load_dotenv()
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

        fp2 = _fingerprint(chunks_sem)
        vecs_p2 = _load_or_embed(f"p2_gemini_{fp2}", chunks_sem, embed_with_gemini)
        idx_p2  = build_faiss_index(vecs_p2)
        state["idx_p2"] = idx_p2

        all_results["2: Gemini-001 + Semantic (prod)"] = evaluate_pipeline(
            "P2", retrieve_p2, samples, state)
    else:
        print("\n  [Skipping P2 — --skip-p2 flag set]")

    all_results["3: E5-Large + BM25 Hybrid"] = evaluate_pipeline(
        "P3", retrieve_p3, samples, state)

    all_results["4: Parent-Child + bge-reranker"] = evaluate_pipeline(
        "P4", retrieve_p4, samples, state)

    # ── Print table ────────────────────────────────────────────────────────────
    print_comparison_table(all_results)

    # ── Charts ────────────────────────────────────────────────────────────────
    plot_charts(all_results)

    # ── Optional LLM judge ─────────────────────────────────────────────────────
    if args.judge:
        retrieve_fns = {
            "3: E5-Large + BM25 Hybrid": retrieve_p3,
            "4: Hybrid + bge-reranker":  retrieve_p4,
        }
        if not args.skip_p2 and "idx_p2" in state:
            retrieve_fns["2: Gemini-001 + Semantic (prod)"] = retrieve_p2

        cr_scores = run_judge_eval(retrieve_fns, samples, state)
        print_judge_table(cr_scores)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
