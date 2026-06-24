# RAG Pipeline Improvement (experiments/v2)

A controlled, **one-variable-at-a-time (OVAT)** experiment to improve the Hebrew
MBA RAG pipeline, and fix the measurement that made the previous P1-P4 run
untrustworthy (confounded design + R@1=0 from URL-level matching).

Each stage changes **exactly one** component and locks the winner before the next
stage. The `configs.check_ovat` guard refuses to run a stage whose arms differ in
more than the intended dimension, so a stage can never silently become a
confounded comparison again.

> Status: this package is scaffolding. Nothing here has been executed. Run the
> stages below in order on a machine with the dependencies installed and a
> `GEMINI_API_KEY` in the environment.

## Prerequisites

```bash
pip install -r ../../requirements.txt          # torch + FlagEmbedding may be large
export GEMINI_API_KEY=...                       # for gemini-embedding-2 + judges
# Optional, only for Stage B arm B2b (ColBERT-v2):  pip install ragatouille
```

Inputs expected at the repo root / eval folder:
- `scraped_content.json` (raw corpus, key `content`) — produced by the scraper.
- `evaluation_framework/ground_truth_mba_qa.csv` — the 100-question gold set.

## Run order

| Stage | Command | Produces |
|-------|---------|----------|
| 0 | `python -m experiments.v2.build_chunk_gt` | `evaluation_framework/chunk_ground_truth.csv` (draft; **review** the `needs_review=TRUE` rows, then set them to FALSE) |
| A | `python -m experiments.v2.run_stage --stage A` | chunking comparison + recommended winner |
| B | `python -m experiments.v2.run_stage --stage B --chunking <A_winner>` | retrieval comparison + winner |
| C | `python -m experiments.v2.run_stage --stage C --chunking <A_winner> --retrieval <B_winner>` | embedding comparison + winner |
| K | `python -m experiments.v2.select_k --chunking <A> --retrieval <B> --embedding <C> --latency-budget-ms 1000` | `stage_k_knee.png` + recommended `(pool_k, rerank_k, context_k)` |
| P | `python -m experiments.v2.promote --chunking <A> --retrieval <B> --embedding <C> --pool-k .. --rerank-k .. --context-k ..` | `winning_config.json` + the exact `rag.py` constants to change (manual) |
| report | `python -m experiments.v2.report` | `IMPROVEMENT_REPORT.md` (+ charts) from `runs.csv` |

All per-query metric rows are appended to `experiments/results/runs.csv`. Read
the results in **`RESULTS.ipynb`** (narrative) or **`IMPROVEMENT_REPORT.md`**
(PR-friendly) — both are generated from that one CSV.

## What each module is

- `configs.py` — `ArmConfig` + per-stage arm builders + the OVAT diff guard.
- `embedders.py` — `gemini-embedding-2` (inline task), `neodictabert`, `bge-m3`; disk-cached.
- `chunkers.py` — baseline / breadcrumb / per_group / parent_child + metadata enrichment + URL-preserving dedup.
- `retrievers.py` — dense, BM25-RRF, sparse-RRF, cross-encoder & ColBERT-v2 rerank, Hebrew query expansion.
- `harness.py` — `run_arm(config)`: build/cache index, run in-corpus GT, write `runs.csv`, return metrics (Recall@k, MRR, nDCG@k, latency p50/p95).
- `build_chunk_gt.py` / `run_stage.py` / `select_k.py` / `promote.py` — the stage drivers.
- `report.py` / `RESULTS.ipynb` — the consolidated results in one place.

## Models

- Fixed constant for Stages A/B/K: **`gemini-embedding-2`** (latest Gemini embedder; 8,192-token input; no `task_type`, so query/doc intent is an inline prefix; vector space differs from `gemini-embedding-001`).
- Stage C challengers: **`dicta-il/neodictabert-bilingual-embed`** (Hebrew-specialized, hypothesized winner) and **`BAAI/bge-m3`**.
- Reranker: **`BAAI/bge-reranker-v2-m3`**; optional **ColBERT-v2** for the latency-friendly arm.
- Generation (unchanged, Stage P eval only): `gemini-3.0-flash` @ temp 0.

## Notes / safety

- Local models need `trust_remote_code=True` and benefit from a GPU.
- Do **not** commit `.env` / API keys. `runs.csv` and charts are generated artifacts.
- The old confounded `experiments/retrieval_experiment.py` (P1-P4) is kept for reference only; do not use it for decisions.
- `rag.py` is intentionally **unchanged** here — promotion happens after results exist, via `promote.py` guidance.
