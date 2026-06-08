# EVAL_FRAMEWORK.md — MBA Advisor Evaluation Framework

## I. Executive Summary

This document describes the measurement scaffolding for the Hebrew University MBA Academic Advisor chatbot. The framework exists to provide reproducible, quantitative evaluation of the full RAG pipeline — before provider-specific choices are locked in and before large-scale ground truth collection begins.

**Why scaffolding first?** An untested pipeline produces results that cannot be compared across runs. By building the measurement layer early, every future change to chunker, embedder, retriever, or generator can be evaluated against a fixed benchmark.

**What it covers:**

| Layer | Metric | Type |
|---|---|---|
| Retrieval | context_relevance | LLM-as-Judge (1–5) |
| Generation (grounding) | faithfulness | LLM-as-Judge (1–5) |
| Generation (vs GT) | correctness | LLM-as-Judge (1–5) |
| Answer quality | answer_relevance | LLM-as-Judge (1–5) |
| Answer quality | completeness | LLM-as-Judge (1–5) |
| Retrieval | recall@k (k=1,3,5,10) | Deterministic (0/1) |
| Retrieval | MRR | Deterministic (0–1) |

**Ground truth:** Two CSV files with schema-ready rows. Domain expert (secretary) tagging (`secretary_tag`) is pending; the framework filters to approved-only once tags arrive.

---

## II. Architecture and Key Concepts

### 2.1 EvalSample Schema

| Field | Type | Notes |
|---|---|---|
| `sample_id` | `str` | Unique identifier |
| `question` | `str` | The user's question (Hebrew) |
| `ground_truth_answer` | `str` | Reference answer (Hebrew) |
| `ground_truth_sources` | `list[str]` | URLs from `source_urls` column (semicolon-separated in CSV) |
| `ground_truth_chunk_ids` | `list[str]` | Optional; for chunk-level retrieval eval |
| `chat_history` | `list[dict]` | `[{role, content}]`; `[]` for single-turn |
| `retrieved_sources` | `list[str] \| None` | Ordered rank-1 first; filled by the RAG pipeline |
| `retrieved_chunk_ids` | `list[str] \| None` | Optional; chunk-level match |
| `answer` | `str \| None` | Bot's answer; filled by the RAG pipeline |
| `category` | `str \| None` | Topic area (admissions, program_structure, exemptions, …) |
| `archetype` | `str \| None` | Conversation pattern: `clarification_loop`, `follow_up`, `mixed` |
| `source_type` | `str` | `single_turn` or `conversation` |
| `secretary_tag` | `str \| None` | `approved`, `needs_fix`, `too_robotic`; `None` until expert fills |

### 2.2 LLM-as-Judge Metrics

All five judges use the same interface: `score(prompt: str) -> JudgeVerdict(score: int 1–5, explanation: str)`.

**Input fields per metric:**

| Metric | question | chat_history | retrieved_sources | answer | ground_truth_answer |
|---|---|---|---|---|---|
| context_relevance | ✓ | ✓ | ✓ | | |
| faithfulness | ✓ | ✓ | ✓ | ✓ | |
| correctness | ✓ | ✓ | | ✓ | ✓ |
| answer_relevance | ✓ | ✓ | | ✓ | ✓ |
| completeness | ✓ | ✓ | | ✓ | ✓ |

**Normalization:** Raw Likert (1–5) is stored in `raw_score` for auditability. `score_norm = (raw_score - 1) / 4` maps to [0.0, 1.0]. Normalization happens in `aggregate.py`, not inside the judge.

**Provider-agnostic interface:**
```python
class LLMJudge(Protocol):
    name: str
    def score(self, prompt: str) -> JudgeVerdict: ...
```
Wire OpenAI, Claude, or a local model by implementing this Protocol. `DummyJudge` (score=3 always) keeps tests and CI green with no API key.

**Prompt templates** live in `judges/prompts/*.txt`. English instructions; Hebrew content via `str.format(**fields)`. When `chat_history` is empty, `chat_history_block` is an empty string and the section is omitted naturally.

### 2.3 Deterministic Retrieval Metrics

No LLM calls. Runs on the same `EvalSample` and lands in the same results parquet.

**recall@k** — 0/1 (any-hit): was at least one ground-truth source returned within the top-k retrieved items?

**MRR** — 1/rank of the first ground-truth hit; 0.0 if no hit.

**URL normalization rules:**
- Lowercase
- Strip trailing `/`
- Strip URL fragment (`#section`)

**Match priority:**
- If both `retrieved_chunk_ids` and `ground_truth_chunk_ids` are non-empty → chunk-id match.
- Otherwise → URL match (v1 default; chunk-level GT data deferred to roadmap P3).

**k values:** Configurable list; default `[1, 3, 5, 10]`. Recall is reported as `recall@1`, `recall@3`, etc.

**Results format:** `judge_name = "deterministic"`, `score_norm = raw_score` (already 0–1), `explanation = ""`.

### 2.4 Results Parquet Schema

One Parquet file per experiment at `evaluation_framework/results/<run_id>.parquet`:

| Column | Type | Notes |
|---|---|---|
| `sample_id` | str | |
| `metric` | str | e.g. `faithfulness`, `recall@3`, `mrr` |
| `raw_score` | float | 1–5 for LLM judges; 0–1 for deterministic |
| `score_norm` | float | 0.0–1.0 always |
| `explanation` | str | Empty for deterministic |
| `judge_name` | str | e.g. `claude`, `dummy`, `deterministic` |
| `run_id` | str | Timestamp or user-supplied identifier |
| `timestamp` | str | ISO-8601 |
| `category` | str | |
| `archetype` | str | |
| `source_type` | str | `single_turn` or `conversation` |

### 2.5 Recall@k Curve Charts

`aggregate.plot_recall_at_k()` produces two PNG files in `results/`:

- **`recall_at_k_overall.png`** — one line per `run_id`, x=k, y=avg recall. Overlay multiple experiment runs (different chunker/embedder combos) to see which configuration reaches saturation fastest.
- **`recall_at_k_per_category.png`** — one line per category within the target run. Shows which topic areas need higher k to reach good recall.

### 2.6 Verification Functions (`verification.py`)

| Function | Raises on |
|---|---|
| `verify_ground_truth_schema` | Missing/wrong-typed required fields |
| `verify_no_duplicate_sample_ids` | Repeated `sample_id` |
| `verify_min_sample_size(min_n=30)` | Fewer than 30 samples |
| `verify_no_null_scores` | `NaN` in `score_norm` |
| `verify_retrieval_inputs_present` | `retrieved_sources is None` when deterministic metrics run |
| `verify_ranked_order` | `retrieved_sources = []` (use `None` instead of empty list) |

---

## III. Results

**Not enough data available — pipeline runs pending.**

This section will be updated after each experiment. Waiting for:
1. LLM provider wired to `LLMJudge` Protocol (e.g. `ClaudeJudge`, `OpenAIJudge`)
2. Ground truth CSVs populated with 100 QA rows and 80 conversations; `secretary_tag = approved` set by domain expert
3. RAG pipeline filling `retrieved_sources` and `answer` on `EvalSample` before calling `runner.run()`

Expected artifacts once runs complete:
- `evaluation_framework/results/<run_id>.parquet`
- `evaluation_framework/results/recall_at_k_overall.png`
- `evaluation_framework/results/recall_at_k_per_category.png`
- Summary table: per-metric mean, per-category breakdown, per-archetype breakdown

---

## IV. Recommendations and Roadmap

### P1 — Calibrate LLM judges on ~10 hand-scored rows
Before trusting automated scores, score 10 samples by hand and compare to judge output. If inter-rater agreement (human vs judge) is below ~70%, revise the rubric anchors in the `.txt` prompt templates.

### P2 — Add Context Precision (LLM judge) when a reranker is introduced
Context Precision measures what fraction of retrieved passages are actually relevant (precision vs recall). It requires a reranker that assigns per-chunk relevance scores. Flag for when the reranker module is built.

### P3 — Add chunk-level ground truth for finer retrieval evaluation
Current `ground_truth_chunk_ids` defaults to `[]` — the schema is ready but the data does not exist yet. Once the RAG pipeline is stable, annotate a subset of ground truth with specific chunk IDs. This enables chunk-level MRR/recall, which is more sensitive than URL-level.
