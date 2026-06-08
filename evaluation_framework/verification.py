"""Verification functions. All raise ValueError on failure."""
from __future__ import annotations

import pandas as pd

from evaluation_framework.schemas import EvalSample


def verify_ground_truth_schema(samples: list[EvalSample]) -> None:
    for s in samples:
        if not isinstance(s.sample_id, str) or not s.sample_id:
            raise ValueError(f"sample_id must be a non-empty str: {s}")
        if not isinstance(s.question, str) or not s.question:
            raise ValueError(f"question must be a non-empty str: {s.sample_id}")
        if not isinstance(s.ground_truth_answer, str) or not s.ground_truth_answer:
            raise ValueError(f"ground_truth_answer must be a non-empty str: {s.sample_id}")
        if not isinstance(s.ground_truth_sources, list):
            raise ValueError(f"ground_truth_sources must be a list: {s.sample_id}")
        if s.source_type not in ("single_turn", "conversation"):
            raise ValueError(f"source_type must be 'single_turn' or 'conversation': {s.sample_id}")


def verify_no_duplicate_sample_ids(samples: list[EvalSample]) -> None:
    ids = [s.sample_id for s in samples]
    seen: set[str] = set()
    dups = [i for i in ids if i in seen or seen.add(i)]  # type: ignore[func-returns-value]
    if dups:
        raise ValueError(f"Duplicate sample_ids: {dups}")


def verify_min_sample_size(samples: list[EvalSample], min_n: int = 30) -> None:
    if len(samples) < min_n:
        raise ValueError(
            f"Need at least {min_n} samples for reliable evaluation; got {len(samples)}."
        )


def verify_no_null_scores(results_df: pd.DataFrame) -> None:
    null_mask = results_df["score_norm"].isna()
    if null_mask.any():
        bad = results_df.loc[null_mask, "sample_id"].tolist()
        raise ValueError(f"Null score_norm for sample_ids: {bad}")


def verify_retrieval_inputs_present(samples: list[EvalSample]) -> None:
    missing = [
        s.sample_id
        for s in samples
        if s.retrieved_sources is None and s.retrieved_chunk_ids is None
    ]
    if missing:
        raise ValueError(
            f"Deterministic metrics requested but retrieved_sources is None for: {missing}"
        )


def verify_ranked_order(samples: list[EvalSample]) -> None:
    for s in samples:
        if s.retrieved_sources is not None and len(s.retrieved_sources) == 0:
            raise ValueError(
                f"retrieved_sources is an empty list for {s.sample_id}. "
                "Use None instead of [] when no sources were retrieved."
            )
        if s.retrieved_chunk_ids is not None and len(s.retrieved_chunk_ids) == 0:
            raise ValueError(
                f"retrieved_chunk_ids is an empty list for {s.sample_id}. "
                "Use None instead of [] when no chunk IDs were retrieved."
            )
