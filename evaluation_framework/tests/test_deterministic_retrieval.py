import pytest
from evaluation_framework.schemas import EvalSample
from evaluation_framework.deterministic_retrieval import (
    normalize_url, recall_at_k, mrr, run_deterministic
)


def _sample(retrieved=None, gt_sources=None, retrieved_chunk_ids=None, gt_chunk_ids=None):
    return EvalSample(
        sample_id="test",
        question="q",
        ground_truth_answer="a",
        ground_truth_sources=gt_sources or [],
        retrieved_sources=retrieved,
        retrieved_chunk_ids=retrieved_chunk_ids,
        ground_truth_chunk_ids=gt_chunk_ids or [],
    )


# ── normalize_url ──────────────────────────────────────────────────

def test_normalize_url_lowercase():
    assert normalize_url("https://Example.COM/Page") == "https://example.com/page"


def test_normalize_url_trailing_slash():
    assert normalize_url("https://example.com/page/") == "https://example.com/page"


def test_normalize_url_fragment():
    assert normalize_url("https://example.com/page#section") == "https://example.com/page"


def test_normalize_url_combined():
    u = "HTTPS://Example.COM/Page/#Section"
    assert normalize_url(u) == "https://example.com/page"  # fragment stripped, then trailing slash stripped


# ── recall_at_k ────────────────────────────────────────────────────

def test_recall_at_k_exact_match():
    s = _sample(
        retrieved=["https://example.com/a", "https://example.com/b"],
        gt_sources=["https://example.com/a"],
    )
    assert recall_at_k(s, k=1) == 1.0


def test_recall_at_k_no_match():
    s = _sample(
        retrieved=["https://example.com/x"],
        gt_sources=["https://example.com/a"],
    )
    assert recall_at_k(s, k=5) == 0.0


def test_recall_at_k_match_beyond_k():
    s = _sample(
        retrieved=["https://example.com/x", "https://example.com/y", "https://example.com/a"],
        gt_sources=["https://example.com/a"],
    )
    assert recall_at_k(s, k=2) == 0.0
    assert recall_at_k(s, k=3) == 1.0


def test_recall_at_k_url_normalization():
    s = _sample(
        retrieved=["https://Example.COM/page/"],
        gt_sources=["https://example.com/page"],
    )
    assert recall_at_k(s, k=1) == 1.0


# ── mrr ────────────────────────────────────────────────────────────

def test_mrr_first_hit():
    s = _sample(
        retrieved=["https://example.com/a"],
        gt_sources=["https://example.com/a"],
    )
    assert mrr(s) == pytest.approx(1.0)


def test_mrr_second_hit():
    s = _sample(
        retrieved=["https://example.com/x", "https://example.com/a"],
        gt_sources=["https://example.com/a"],
    )
    assert mrr(s) == pytest.approx(0.5)


def test_mrr_no_hit():
    s = _sample(
        retrieved=["https://example.com/x"],
        gt_sources=["https://example.com/a"],
    )
    assert mrr(s) == 0.0


# ── chunk_id path ──────────────────────────────────────────────────

def test_chunk_id_match_preferred():
    s = _sample(
        retrieved=["https://example.com/wrong"],
        gt_sources=["https://example.com/wrong"],   # URL would match, but chunk path takes over
        retrieved_chunk_ids=["chunk_99"],
        gt_chunk_ids=["chunk_42"],
    )
    # chunk_ids don't match → 0
    assert recall_at_k(s, k=1) == 0.0


def test_chunk_id_match_hit():
    s = _sample(
        retrieved=["https://example.com/x"],
        gt_sources=["https://example.com/y"],
        retrieved_chunk_ids=["chunk_42"],
        gt_chunk_ids=["chunk_42"],
    )
    assert recall_at_k(s, k=1) == 1.0


# ── run_deterministic ──────────────────────────────────────────────

def test_run_deterministic_returns_rows():
    s = _sample(
        retrieved=["https://example.com/a"],
        gt_sources=["https://example.com/a"],
    )
    rows = run_deterministic(s, k_values=[1, 3])
    metrics = {r["metric"] for r in rows}
    assert "recall@1" in metrics
    assert "recall@3" in metrics
    assert "mrr" in metrics
    for r in rows:
        assert 0.0 <= r["score_norm"] <= 1.0
        assert r["judge_name"] == "deterministic"


def test_run_deterministic_no_sources():
    s = _sample()
    rows = run_deterministic(s)
    assert rows == []
