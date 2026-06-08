"""
Deterministic retrieval metrics — no LLM calls required.

Metrics:
  recall_at_k  — 0/1 any-hit within top-k retrieved items
  mrr          — mean reciprocal rank of the first ground-truth hit

Match priority:
  If both sample.retrieved_chunk_ids and sample.ground_truth_chunk_ids are non-empty
  → chunk_id-level match.
  Otherwise → URL-level match (normalized: lowercase, no trailing slash, no fragment).
"""
from __future__ import annotations

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


def run_deterministic(
    sample: EvalSample,
    k_values: list[int] = None,
    run_id: str = "",
) -> list[dict]:
    """Return a list of result-row dicts (one per metric) ready for the results parquet."""
    if k_values is None:
        k_values = [1, 3, 5, 10]

    if not sample.retrieved_sources and not sample.retrieved_chunk_ids:
        return []

    timestamp = datetime.now(timezone.utc).isoformat()
    rows = []

    for k in k_values:
        score = recall_at_k(sample, k)
        rows.append({
            "sample_id": sample.sample_id,
            "metric": f"recall@{k}",
            "raw_score": score,
            "score_norm": score,
            "explanation": "",
            "judge_name": "deterministic",
            "run_id": run_id,
            "timestamp": timestamp,
            "category": sample.category,
            "archetype": sample.archetype,
            "source_type": sample.source_type,
        })

    mrr_score = mrr(sample)
    rows.append({
        "sample_id": sample.sample_id,
        "metric": "mrr",
        "raw_score": mrr_score,
        "score_norm": mrr_score,
        "explanation": "",
        "judge_name": "deterministic",
        "run_id": run_id,
        "timestamp": timestamp,
        "category": sample.category,
        "archetype": sample.archetype,
        "source_type": sample.source_type,
    })

    return rows
