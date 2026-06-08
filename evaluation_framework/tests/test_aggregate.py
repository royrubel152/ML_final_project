import pandas as pd
import pytest
from evaluation_framework.aggregate import summarize


def _make_results() -> pd.DataFrame:
    rows = []
    for sample_id in ["s1", "s2", "s3"]:
        for metric in ["faithfulness", "correctness", "recall@1", "mrr"]:
            rows.append({
                "sample_id":   sample_id,
                "metric":      metric,
                "raw_score":   3,
                "score_norm":  0.5,
                "explanation": "",
                "judge_name":  "dummy",
                "run_id":      "test_run",
                "timestamp":   "2026-01-01T00:00:00",
                "category":    "admissions" if sample_id == "s1" else "program_structure",
                "archetype":   "follow_up",
                "source_type": "single_turn",
            })
    return pd.DataFrame(rows)


def test_summarize_keys():
    df = _make_results()
    summary = summarize(df)
    assert "per_metric" in summary
    assert "per_category" in summary
    assert "per_archetype" in summary
    assert "per_source_type" in summary


def test_summarize_per_metric_values():
    df = _make_results()
    summary = summarize(df)
    for metric, mean_score in summary["per_metric"].items():
        assert 0.0 <= mean_score <= 1.0, f"{metric} mean out of range"


def test_summarize_per_category():
    df = _make_results()
    summary = summarize(df)
    cats = set(summary["per_category"].keys())
    assert "admissions" in cats
    assert "program_structure" in cats


def test_summarize_per_source_type():
    df = _make_results()
    summary = summarize(df)
    assert "single_turn" in summary["per_source_type"]
