"""
retrievers.py — retrieval methods for the Stage B ablation (and reused by C/K).

Methods (one per ArmConfig.retrieval value):
  dense          : cosine over FAISS IndexFlatIP (L2-normalized vectors).
  bm25_rrf       : dense + BM25 lexical, fused with Reciprocal Rank Fusion.
  sparse_rrf     : dense + bge-m3 learned-sparse, fused with RRF.
  rerank_ce      : bm25_rrf candidates re-scored by a cross-encoder.
  rerank_colbert : bm25_rrf candidates re-scored by ColBERT-v2 late interaction.
  rerank_ce_qe   : rerank_ce with Hebrew morphological query expansion.

The harness builds a RetrievalContext once per arm (indexes, models) and then
calls ``run_retrieval`` per query. Heavy deps (faiss, sentence-transformers,
FlagEmbedding, ragatouille/pylate) are imported lazily; ColBERT is optional and
its absence raises a clear, skippable error.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

HEB_PREFIXES = ("ב", "ל", "ה", "ש", "ו", "מ", "כ")
_COURSE_CODE = re.compile(r"\b\d{5}\b")


# ── Query utilities ─────────────────────────────────────────────────────────

def expand_query_hebrew(query: str) -> str:
    """Append morphological prefix variants for each token (mirrors rag._expand_query)."""
    tokens = query.split()
    expanded = list(tokens)
    for tok in tokens:
        if len(tok) >= 3:
            for p in HEB_PREFIXES:
                variant = p + tok
                if variant not in expanded:
                    expanded.append(variant)
    return " ".join(expanded)


# ── Index builders ──────────────────────────────────────────────────────────

def build_faiss_index(vectors: np.ndarray):
    import faiss
    vecs = vectors.copy().astype("float32")
    faiss.normalize_L2(vecs)
    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)
    return idx


def build_bm25(chunks: list[dict]):
    from rank_bm25 import BM25Okapi
    return BM25Okapi([c["text"].split() for c in chunks])


# ── Primitive searches ──────────────────────────────────────────────────────

def dense_search(q_vec: np.ndarray, index, top_k: int) -> list[tuple[int, float]]:
    import faiss
    q = q_vec.reshape(1, -1).astype("float32").copy()
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, top_k)
    return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]


def bm25_search(bm25, query: str, n_docs: int, top_k: int) -> list[tuple[int, float]]:
    scores = bm25.get_scores(query.split())
    ranked = sorted(range(n_docs), key=lambda i: -scores[i])[:top_k]
    return [(i, float(scores[i])) for i in ranked]


def sparse_search(query_weights: dict, doc_weights: list[dict], top_k: int) -> list[tuple[int, float]]:
    """Score docs by dot product of bge-m3 lexical weights (token_id -> weight)."""
    scored = []
    for i, dw in enumerate(doc_weights):
        s = sum(w * dw.get(tok, 0.0) for tok, w in query_weights.items())
        if s > 0:
            scored.append((i, float(s)))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


def rrf_fuse(ranked_lists: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion over several ranked lists of (chunk_index, score)."""
    fused: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (idx, _score) in enumerate(ranked):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: -x[1])


def _course_code_boost(query: str, fused: list[tuple[int, float]], chunks: list[dict],
                       factor: float = 1.3) -> list[tuple[int, float]]:
    """Uprank chunks containing an exact 5-digit course code present in the query."""
    codes = _COURSE_CODE.findall(query)
    if not codes:
        return fused
    boosted = [
        (idx, score * factor if any(c in chunks[idx]["text"] for c in codes) else score)
        for idx, score in fused
    ]
    boosted.sort(key=lambda x: -x[1])
    return boosted


# ── Context object built once per arm ───────────────────────────────────────

@dataclass
class RetrievalContext:
    chunks: list[dict]
    embedder: object                 # embedders.BaseEmbedder
    faiss_index: object              # dense index over chunks
    bm25: object | None = None
    doc_sparse_weights: list[dict] | None = None  # bge-m3 lexical weights per chunk
    reranker: object | None = None   # cross-encoder
    colbert: object | None = None    # ColBERT-v2 index/model
    sparse_embedder: object | None = None  # bge-m3 embedder for query sparse weights


# ── Rerankers ───────────────────────────────────────────────────────────────

def load_cross_encoder(model_id: str = "BAAI/bge-reranker-v2-m3"):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_id)


def cross_encoder_rerank(reranker, query: str, candidates: list[dict], top_k: int) -> list[dict]:
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(candidates, scores), key=lambda x: -float(x[1]))
    out = []
    for c, s in ranked[:top_k]:
        r = dict(c)
        r["score"] = float(s)
        out.append(r)
    return out


def colbert_rerank(colbert, query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """ColBERT-v2 late-interaction rerank. Optional dependency (ragatouille/pylate)."""
    if colbert is None:
        raise RuntimeError(
            "ColBERT-v2 reranker not available. Install 'ragatouille' (or 'pylate') "
            "to run arm B2b, or skip it — the rest of Stage B runs without it."
        )
    return colbert.rerank(query, candidates, top_k)  # adapter defined in harness


# ── Main dispatcher ─────────────────────────────────────────────────────────

def run_retrieval(method: str, query: str, ctx: RetrievalContext,
                  pool_k: int, rerank_k: int, context_k: int) -> list[dict]:
    """Run the configured retrieval method and return up to context_k chunk dicts."""
    use_qe = method.endswith("_qe")
    search_query = expand_query_hebrew(query) if use_qe else query
    chunks = ctx.chunks
    n = len(chunks)

    # ── First-stage candidate generation ────────────────────────────────────
    q_vec = ctx.embedder.embed_query(search_query)
    dense_hits = dense_search(q_vec, ctx.faiss_index, pool_k)

    if method == "dense":
        fused = dense_hits
    elif method == "bm25_rrf":
        bm = bm25_search(ctx.bm25, search_query, n, pool_k)
        fused = rrf_fuse([dense_hits, bm])
        fused = _course_code_boost(search_query, fused, chunks)
    elif method == "sparse_rrf":
        qw = ctx.sparse_embedder.encode_sparse([search_query])[0]
        sp = sparse_search(qw, ctx.doc_sparse_weights, pool_k)
        fused = rrf_fuse([dense_hits, sp])
        fused = _course_code_boost(search_query, fused, chunks)
    elif method in ("rerank_ce", "rerank_colbert", "rerank_ce_qe"):
        bm = bm25_search(ctx.bm25, search_query, n, pool_k)
        fused = rrf_fuse([dense_hits, bm])
        fused = _course_code_boost(search_query, fused, chunks)
    else:
        raise ValueError(f"unknown retrieval method '{method}'")

    candidates = [dict(chunks[idx], score=score) for idx, score in fused[:rerank_k]]

    # ── Optional rerank stage ────────────────────────────────────────────────
    if method in ("rerank_ce", "rerank_ce_qe"):
        return cross_encoder_rerank(ctx.reranker, query, candidates, context_k)
    if method == "rerank_colbert":
        return colbert_rerank(ctx.colbert, query, candidates, context_k)

    return candidates[:context_k]
