"""
Deterministic retrieval metrics — no LLM calls required.

Metrics:
  recall_at_k  — 0/1 any-hit within top-k retrieved items
  mrr          — mean reciprocal rank of the first ground-truth hit
  ndcg_at_k    — normalized discounted cumulative gain at k (binary relevance);
                 rewards ranking ground-truth items higher, and handles the case
                 where a question has several relevant chunks.

Latency:
  run_deterministic accepts an optional latency_ms; when provided it emits a
  separate "latency_ms" row (raw milliseconds, not normalized to 0-1).

Match priority:
  If both sample.retrieved_chunk_ids and sample.ground_truth_chunk_ids are non-empty
  → chunk_id-level match.
  Otherwise → URL-level match (normalized: lowercase, no trailing slash, no fragment).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from evaluation_framework.schemas import EvalSample


def normalize_url(url: str) -> str:
    url = url.lower().strip()
    if "#" in url:
        url = url[: url.index("#")]
    return url.rstrip("/")


def _use_chunk_ids(sample: EvalSample) -> bool:
    return bool(sample.retrieved_chunk_ids) and bool(sample.ground_truth_chunk_ids)


def _ground_truth_set(sample: EvalSample) -> set[str]:
    if _use_chunk_ids(sample):
        return set(sample.ground_truth_chunk_ids)
    return {normalize_url(u) for u in sample.ground_truth_sources}


def _retrieved_list(sample: EvalSample) -> list[str]:
    if _use_chunk_ids(sample):
        return list(sample.retrieved_chunk_ids or [])
    return [normalize_url(u) for u in (sample.retrieved_sources or [])]


def recall_at_k(sample: EvalSample, k: int) -> float:
    gt = _ground_truth_set(sample)
    top_k = _retrieved_list(sample)[:k]
    return 1.0 if any(item in gt for item in top_k) else 0.0


def mrr(sample: EvalSample) -> float:
    gt = _ground_truth_set(sample)
    for rank, item in enumerate(_retrieved_list(sample), start=1):
        if item in gt:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(sample: EvalSample, k: int) -> float:
    """Normalized DCG@k with binary relevance.

    gain(rank) = 1 if the retrieved item is a ground-truth item else 0,
    discounted by 1/log2(rank+1). Normalized by the ideal DCG (all relevant
    items ranked first), so the score is in [0, 1]. Returns 0.0 when there are
    no ground-truth items.
    """
    gt = _ground_truth_set(sample)
    if not gt:
        return 0.0
    top_k = _retrieved_list(sample)[:k]

    dcg = 0.0
    for rank, item in enumerate(top_k, start=1):
        if item in gt:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(gt), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def run_deterministic(
    sample: EvalSample,
    k_values: list[int] = None,
    run_id: str = "",
    latency_ms: float | None = None,
) -> list[dict]:
    """Return a list of result-row dicts (one per metric) ready for the results table.

    Emits recall@k, ndcg@k (for each k in ``k_values``) and a single ``mrr`` row.
    If ``latency_ms`` is provided, also emits a ``latency_ms`` row whose
    ``raw_score`` is the measured milliseconds (``score_norm`` is left equal to the
    raw value and is intentionally NOT in [0,1] — latency is a cost, not a quality
    score; downstream reporting treats it separately).
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    if not sample.retrieved_sources and not sample.retrieved_chunk_ids:
        return []

    timestamp = datetime.now(timezone.utc).isoformat()

    def _row(metric: str, raw: float, norm: float) -> dict:
        return {
            "sample_id": sample.sample_id,
            "metric": metric,
            "raw_score": raw,
            "score_norm": norm,
            "explanation": "",
            "judge_name": "deterministic",
            "run_id": run_id,
            "timestamp": timestamp,
            "category": sample.category,
            "archetype": sample.archetype,
            "source_type": sample.source_type,
        }

    rows = []
    for k in k_values:
        score = recall_at_k(sample, k)
        rows.append(_row(f"recall@{k}", score, score))
    for k in k_values:
        nd = ndcg_at_k(sample, k)
        rows.append(_row(f"ndcg@{k}", nd, nd))

    mrr_score = mrr(sample)
    rows.append(_row("mrr", mrr_score, mrr_score))

    if latency_ms is not None:
        rows.append(_row("latency_ms", float(latency_ms), float(latency_ms)))

    return rows
