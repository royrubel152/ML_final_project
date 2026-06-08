"""
runner.py — orchestrates judge + deterministic metric evaluation over a list of EvalSamples.

Usage:
    from evaluation_framework.runner import run
    from evaluation_framework.judges.base import DummyJudge
    results_df = run(samples, judge=DummyJudge(), metrics=["faithfulness", "correctness"])
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pandas as pd

from evaluation_framework import deterministic_retrieval
from evaluation_framework import judges as _judges_pkg
from evaluation_framework.schemas import EvalSample

if TYPE_CHECKING:
    from evaluation_framework.judges.base import LLMJudge

_METRIC_MODULES = {
    "context_relevance": "evaluation_framework.judges.context_relevance",
    "faithfulness":      "evaluation_framework.judges.faithfulness",
    "correctness":       "evaluation_framework.judges.correctness",
    "answer_relevance":  "evaluation_framework.judges.answer_relevance",
    "completeness":      "evaluation_framework.judges.completeness",
}

ALL_METRICS = list(_METRIC_MODULES.keys())


def run(
    samples: list[EvalSample],
    judge: "LLMJudge",
    metrics: list[str] | None = None,
    run_deterministic_metrics: bool = True,
    k_values: list[int] | None = None,
    run_id: str | None = None,
) -> pd.DataFrame:
    """
    Evaluate all samples with the given judge and metrics.

    Returns a long-format DataFrame with columns:
      sample_id, metric, raw_score, score_norm, explanation,
      judge_name, run_id, timestamp, category, archetype, source_type
    """
    if metrics is None:
        metrics = ALL_METRICS
    if k_values is None:
        k_values = [1, 3, 5, 10]
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    import importlib

    rows: list[dict] = []

    for sample in samples:
        timestamp = datetime.now(timezone.utc).isoformat()

        # LLM-as-Judge metrics
        for metric in metrics:
            if metric not in _METRIC_MODULES:
                raise ValueError(f"Unknown metric '{metric}'. Choose from {ALL_METRICS}.")
            mod = importlib.import_module(_METRIC_MODULES[metric])
            prompt = mod.build_prompt(sample)
            verdict = judge.score(prompt)
            verdict.metric = metric

            raw = verdict.score
            rows.append({
                "sample_id":   sample.sample_id,
                "metric":      metric,
                "raw_score":   raw,
                "score_norm":  (raw - 1) / 4,
                "explanation": verdict.explanation,
                "judge_name":  judge.name,
                "run_id":      run_id,
                "timestamp":   timestamp,
                "category":    sample.category,
                "archetype":   sample.archetype,
                "source_type": sample.source_type,
            })

        # Deterministic retrieval metrics
        if run_deterministic_metrics:
            det_rows = deterministic_retrieval.run_deterministic(sample, k_values, run_id)
            rows.extend(det_rows)

    return pd.DataFrame(rows)
