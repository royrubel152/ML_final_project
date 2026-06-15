# Plan: Hebrew Retrieval Experimentation Framework

## Context
The MBA chatbot (rag.py) uses one fixed pipeline: semantic chunking + gemini-embedding-001 + FAISS cosine. This experiment compares 3 distinct pipelines that vary **chunking strategy**, **embedding model**, and **retrieval method** simultaneously. Each pipeline is evaluated with both deterministic metrics (Recall@k, MRR) and the full LLM-as-judge suite from `evaluation_framework/`. Ground truth: 100 single-turn Q&A from `evaluation_framework/ground_truth_mba_qa.csv` — conversations file excluded.

---

## Data Facts

- **Raw content:** `scraped_content.json` — `{"content": "=== source === \nURL: ...\n[body]..."}` format; parsed by `rag._parse_sections()`
- **Pre-chunked corpus:** `data/chunks.json` — 658 chunks (900-char semantic, with header context prefix)
- **Ground truth CSV:** `evaluation_framework/ground_truth_mba_qa.csv` — 100 rows
  - **Actual column names:** `id, category, question_he, answer_he, answer_he_chatbot, source_urls, needs_secretary_review, secretary_tag, notes`
  - **NOTE:** `io_ground_truth.load_qa()` expects `sample_id / question / ground_truth_answer` — **cannot use directly**; need pandas rename wrapper
- **Already installed:** `sentence-transformers==5.1.0`, `faiss-cpu==1.14.2`, `numpy`, `pandas`
- **Needs install:** `rank_bm25`

---

## Phase 1: RAG Preparation & Entity Extraction

Run before building any index. Print results as a markdown report.

### 1A — Hebrew Text Preprocessing (applied to all raw text before chunking)
```python
import unicodedata, re

def preprocess_hebrew(text: str) -> str:
    text = unicodedata.normalize("NFC", text)           # canonical Unicode form
    text = re.sub(r"[֑-ׇ]", "", text)                  # strip nikud/diacritics
    text = re.sub(r"\s{2,}", " ", text)                 # collapse whitespace
    return text.strip()
```
Applied to each chunk's `text` field before embedding and before BM25 tokenization.

### 1B — Entity Extraction (static analysis over `data/chunks.json`)

| Entity Category | Pattern | Example |
|---|---|---|
| Course Codes | `\b\d{5}\b` | 55844, 55810 |
| Credit Values | `\d+(?:\.\d+)?\s*נ["״]ז` | 3 נ"ז |
| Lecturer Names | text after `מרצה:\|מלמד:` | ד"ר כהן |
| Specialization Names | 9-item hardcoded list | פינטק, שיווק |
| Deadlines | `עד\s+\d{1,2}\.\d{2}\.\d{4}` | עד 16.08.2026 |
| Prerequisites | after `תנאי קדם:\|ידע קודם:` | מתמטיקה א' |
| Grade Thresholds | `ציון.*?\d{2,3}` | ציון עובר 60 |

Entity leverage proposals:
- **Course codes** → BM25 exact match + 1.3× entity boost in P3/P4
- **Specializations** → score boost already applied in production (rag.py)
- **Deadlines** → flag as time-sensitive; exclude from long-lived cache
- **Prerequisites** → prerequisite graph for course-planning Q&A

### 1C — Ground Truth Loading Wrapper
```python
def load_ground_truth() -> list[EvalSample]:
    df = pd.read_csv(GT_CSV)
    samples = []
    for _, row in df.iterrows():
        sources = [s.strip() for s in str(row["source_urls"]).split(";") if s.strip()]
        samples.append(EvalSample(
            sample_id=str(row["id"]),
            question=str(row["question_he"]),
            ground_truth_answer=str(row["answer_he"]),
            ground_truth_sources=sources,
            category=str(row["category"]),
            source_type="single_turn",
        ))
    return samples
```

---

## Phase 2: 4 Pipelines (Different Chunking + Embedding + Retrieval)

### Pipeline 1 — Baseline: Fixed-Size Chunking + MiniLM
- **Chunking:** Re-chunk raw text from `scraped_content.json` using `rag._parse_sections()`, then apply simple fixed-size splitter (500 chars, 50 overlap, no header awareness, no context prefix)
- **Preprocessing:** `preprocess_hebrew()` on each chunk text
- **Embedding model:** `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, sentence-transformers)
- **Index:** FAISS `IndexFlatIP` + L2 normalization (cosine)
- **Retrieval:** pure vector search, top-10 candidates, no query expansion

Why this is baseline: naive chunking breaks section structure; MiniLM is fast but weaker than E5-large.

### Pipeline 2 — Production Replica: Semantic Chunking + Gemini-Embedding-001 + Query Expansion
- **Chunking:** Use pre-built `data/chunks.json` (900-char semantic chunks with header context prefix — the current production chunking)
- **Preprocessing:** `preprocess_hebrew()` before embedding
- **Embedding model:** `gemini-embedding-001` via Google Generative AI API (3072-dim)
  - Uses `task_type="retrieval_document"` for chunks, `task_type="retrieval_query"` for queries
- **Index:** FAISS `IndexFlatIP` + L2 normalization
- **Retrieval:** vector search + Hebrew morphological expansion (`_expand_query`)
  - Prefix variants ב,ל,ה,ש,ו,מ,כ applied to each query token

Why this matters: this is the current production system as a clean benchmark. Use `--skip-p2` to avoid API cost.

**Caching:** embeddings saved to `experiments/cache/p2_gemini_*.npy` — reused across runs.

### Pipeline 3 — Hybrid: Semantic Chunking + E5-Large + BM25 + Entity Boost
- **Chunking:** same `data/chunks.json` as P2
- **Embedding model:** `intfloat/multilingual-e5-large` (1024-dim, local inference)
  - Queries prefixed with `"query: "`, passages prefixed with `"passage: "` (E5 instruction format)
- **Sparse index:** `rank_bm25.BM25Okapi` on whitespace-tokenized chunk texts
- **Retrieval:** Reciprocal Rank Fusion
  ```
  RRF(q, chunk_i) = 1/(60 + rank_dense_i) + 1/(60 + rank_bm25_i)
  ```
- **Entity boost:** if query contains `\b\d{5}\b` (course code) → multiply RRF score of matching chunks by 1.3

Why this is the contender: BM25 catches exact Hebrew terms and course codes; E5-large is MTEB SOTA for multilingual retrieval; RRF fusion is robust.

### Pipeline 4 — Ultimate: Hybrid + Cross-Encoder Reranker (Post-Processing)
- **First stage:** identical to Pipeline 3 — fetch top-**20** candidates via RRF(BM25 + E5-large)
- **Reranker:** `BAAI/bge-reranker-v2-m3` cross-encoder (~570MB, multilingual, supports Hebrew)
  - Input: list of `[query, chunk_text]` pairs → joint encoding → relevance scores
  - Re-sort 20 candidates by cross-encoder score → keep top-10
- **Entity boost:** applied before reranking to keep entity-matched chunks in top-20 pool

**Two-stage flow:**
```
query
  │
  ├─ E5-large dense  →  top-20 candidates  ─┐
  ├─ BM25 sparse     →  top-20 candidates   ├─ RRF fusion → top-20
  │                                         ─┘       │
  └─ bge-reranker-v2-m3 cross-encoder (20 pairs) → re-sort → top-10
```

Why reranking works: bi-encoders embed query and chunk independently → miss fine-grained interactions. Cross-encoders attend across both jointly → much higher precision at rank 1.

---

## Phase 3: Evaluation — Deterministic Metrics Only

(LLM-as-judge excluded per design decision.)

### Deterministic Metrics (always runs, no API cost)
Uses `evaluation_framework.deterministic_retrieval.run_deterministic()`.

For each pipeline, set `EvalSample.retrieved_sources` = URLs of top-10 retrieved chunks (ordered), then call `run_deterministic(sample, k_values=[1,3,5,10])`.

**Metrics computed:**
| Metric | Definition |
|---|---|
| Recall@1 | 1 if correct URL in top-1 retrieved, else 0 |
| Recall@3 | 1 if correct URL in top-3 retrieved, else 0 |
| Recall@5 | 1 if correct URL in top-5 retrieved, else 0 |
| Recall@10 | 1 if correct URL in top-10 retrieved, else 0 |
| MRR | 1 / rank of first correct URL hit |

URL matching: lowercase, strip fragment `#...`, strip trailing `/`.

---

## Phase 4: Output

### Console — Retrieval Metrics Table
```
| Pipeline                              | R@1  | R@3  | R@5  | R@10 | MRR  |
|---------------------------------------|------|------|------|------|------|
| 1: MiniLM + Fixed Chunks              | 0.xx | 0.xx | 0.xx | 0.xx | 0.xx |
| 2: Gemini-001 + Semantic (prod)       | 0.xx | 0.xx | 0.xx | 0.xx | 0.xx |
| 3: E5-Large + BM25 Hybrid             | 0.xx | 0.xx | 0.xx | 0.xx | 0.xx |
| 4: Hybrid + bge-reranker (2-stage)    | 0.xx | 0.xx | 0.xx | 0.xx | 0.xx |
```

### Charts (saved to `experiments/results/`)
1. **`recall_at_k.png`** — 4 pipeline curves on a single chart (x = k, y = Recall@k)
2. **`recall_by_category.png`** — 2×4 subplot grid, one per category, each with 4 pipeline curves

---

## Files Created

| File | Notes |
|---|---|
| `experiments/retrieval_experiment.py` | Main script (~500 lines) |
| `experiments/EXPERIMENT_PLAN.md` | This document |
| `experiments/__init__.py` | Empty package marker |
| `experiments/cache/` | Cached `.npy` embedding arrays (gitignored) |
| `experiments/results/` | PNG charts + run logs |

---

## Usage

```bash
# Full run (P1 + P2 + P3 + P4 — P2 requires Gemini API)
python experiments/retrieval_experiment.py

# Skip P2 to save API cost (P1, P3, P4 only)
python experiments/retrieval_experiment.py --skip-p2

# Second run is fast — loads cached embeddings
python experiments/retrieval_experiment.py --skip-p2
```

---

## Caching Strategy
- First run: computes embeddings, saves `experiments/cache/{model}_{fingerprint}.npy`
- Subsequent runs: loads from cache file if present
- Cache fingerprint: MD5 of first 30 chunks' text (detects corpus changes)

---

## Sanity Checks
- P3 Recall@5 ≥ P1 Recall@5 (hybrid must beat fixed-size baseline)
- P4 R@1 ≥ P3 R@1 (reranker lifts precision at rank 1)
- Charts saved and openable after run completes
