"""
harness.py — the shared OVAT engine.

``run_arm(config)`` builds (or loads from cache) the index for one ArmConfig,
runs every in-corpus ground-truth query, and returns aggregated metrics
(Recall@k, MRR, nDCG@k, latency p50/p95). Per-sample metric rows are appended to
``experiments/results/runs.csv`` so all reporting is generated from data.

This module orchestrates only; the actual model work lives in chunkers.py,
embedders.py and retrievers.py (all lazy-importing their heavy deps). Running an
arm performs real embedding/retrieval — so it is executed by the user/friend,
never at build time.
"""
from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

import numpy as np

from experiments.v2 import chunkers, embedders, retrievers
from experiments.v2.configs import ArmConfig

ROOT = Path(__file__).resolve().parent.parent.parent
GT_CSV = ROOT / "evaluation_framework" / "ground_truth_mba_qa.csv"
CHUNK_GT_CSV = ROOT / "evaluation_framework" / "chunk_ground_truth.csv"
RESULTS_DIR = ROOT / "experiments" / "results"
RUNS_CSV = RESULTS_DIR / "runs.csv"

K_VALUES = [1, 3, 5, 10]

_RUN_COLUMNS = [
    "run_id", "arm_id", "chunking", "embedding", "retrieval",
    "pool_k", "rerank_k", "context_k", "enrich",
    "sample_id", "metric", "raw_score", "score_norm", "category", "timestamp",
]


# ── Ground truth ────────────────────────────────────────────────────────────

def load_in_corpus_samples() -> list:
    """Load GT, attach chunk-level labels, return only the in-corpus subset."""
    from evaluation_framework import io_ground_truth
    samples = io_ground_truth.load_qa(GT_CSV)
    io_ground_truth.attach_chunk_gt(samples, CHUNK_GT_CSV)
    in_corpus, secretariat_only = io_ground_truth.segment_samples(samples)
    print(f"  [gt] {len(in_corpus)} in-corpus, {len(secretariat_only)} secretariat-only (excluded)")
    return in_corpus


# ── Context construction ────────────────────────────────────────────────────

def build_context(config: ArmConfig, chunks: list[dict]) -> retrievers.RetrievalContext:
    """Embed chunks (cached) and build every index the arm's method needs."""
    embedder = embedders.get_embedder(config.embedding)
    texts = [c["text"] for c in chunks]

    print(f"  [embed] {config.embedding} over {len(texts)} chunks")
    doc_vecs = embedder.embed_documents(texts, use_cache=True)
    faiss_index = retrievers.build_faiss_index(doc_vecs)

    ctx = retrievers.RetrievalContext(
        chunks=chunks, embedder=embedder, faiss_index=faiss_index,
    )

    method = config.retrieval
    if method in ("bm25_rrf", "rerank_ce", "rerank_colbert", "rerank_ce_qe"):
        ctx.bm25 = retrievers.build_bm25(chunks)
    if method == "sparse_rrf":
        sparse = embedders.get_embedder("bge-m3")
        ctx.sparse_embedder = sparse
        print("  [sparse] computing bge-m3 lexical weights for all chunks")
        ctx.doc_sparse_weights = sparse.encode_sparse(texts)
    if method in ("rerank_ce", "rerank_ce_qe"):
        print("  [rerank] loading BAAI/bge-reranker-v2-m3")
        ctx.reranker = retrievers.load_cross_encoder()
    if method == "rerank_colbert":
        ctx.colbert = _try_load_colbert(chunks)

    return ctx


def _try_load_colbert(chunks: list[dict]):
    """Best-effort ColBERT-v2 loader; returns None if the optional dep is absent."""
    try:
        from experiments.v2._colbert_adapter import ColbertReranker  # optional helper
        return ColbertReranker(chunks)
    except Exception as exc:  # noqa: BLE001 - optional path, degrade gracefully
        print(f"  [rerank] ColBERT-v2 unavailable ({exc}); arm B2b will be skipped")
        return None


# ── Metrics ─────────────────────────────────────────────────────────────────

def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1))))
    return s[idx]


def run_arm(config: ArmConfig, write: bool = True) -> dict:
    """Build the arm, evaluate all in-corpus queries, return aggregated metrics."""
    from datetime import datetime, timezone
    from evaluation_framework import deterministic_retrieval

    config.validate()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    print(f"\n=== run_arm {config.arm_id} :: {config.pipeline_signature()} ===")

    chunks = chunkers.build_chunks(config.chunking, enrich_meta=config.enrich)
    print(f"  [chunks] {len(chunks)} retrieval units ({config.chunking})")
    ctx = build_context(config, chunks)
    samples = load_in_corpus_samples()

    per_metric: dict[str, list[float]] = {}
    latencies: list[float] = []
    all_rows: list[dict] = []

    for i, sample in enumerate(samples):
        t0 = time.perf_counter()
        hits = retrievers.run_retrieval(
            config.retrieval, sample.question, ctx,
            pool_k=config.pool_k, rerank_k=config.rerank_k, context_k=config.context_k,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        sample.retrieved_chunk_ids = [h["chunk_id"] for h in hits]
        sample.retrieved_sources = [h.get("url", "") for h in hits]

        rows = deterministic_retrieval.run_deterministic(
            sample, k_values=K_VALUES, run_id=run_id, latency_ms=latency_ms,
        )
        for r in rows:
            per_metric.setdefault(r["metric"], []).append(r["score_norm"])
            all_rows.append(_results_row(config, run_id, r))

        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{len(samples)} queries")

    metrics = {m: (sum(v) / len(v) if v else 0.0) for m, v in per_metric.items()}
    metrics["latency_p50_ms"] = _percentile(latencies, 50)
    metrics["latency_p95_ms"] = _percentile(latencies, 95)
    metrics["n_samples"] = len(samples)

    if write:
        _append_runs(all_rows)
        print(f"  [write] {len(all_rows)} rows -> {RUNS_CSV}")

    _print_summary(config, metrics)
    return metrics


def _results_row(config: ArmConfig, run_id: str, metric_row: dict) -> dict:
    return {
        "run_id": run_id,
        "arm_id": config.arm_id,
        "chunking": config.chunking,
        "embedding": config.embedding,
        "retrieval": config.retrieval,
        "pool_k": config.pool_k,
        "rerank_k": config.rerank_k,
        "context_k": config.context_k,
        "enrich": config.enrich,
        "sample_id": metric_row["sample_id"],
        "metric": metric_row["metric"],
        "raw_score": metric_row["raw_score"],
        "score_norm": metric_row["score_norm"],
        "category": metric_row.get("category"),
        "timestamp": metric_row.get("timestamp"),
    }


def _append_runs(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    exists = RUNS_CSV.exists()
    with open(RUNS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_RUN_COLUMNS)
        if not exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _print_summary(config: ArmConfig, metrics: dict) -> None:
    keys = [f"recall@{k}" for k in K_VALUES] + ["mrr", f"ndcg@5", "latency_p50_ms", "latency_p95_ms"]
    parts = [f"{k}={metrics.get(k, 0.0):.3f}" for k in keys]
    print(f"  [metrics] {config.arm_id}: " + "  ".join(parts))


if __name__ == "__main__":  # pragma: no cover - manual smoke entrypoint
    print("Use run_stage.py to execute a stage; harness.run_arm(config) runs one arm.")
