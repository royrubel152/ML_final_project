"""
experiments/v2 — Controlled (OVAT) RAG improvement experiment.

One Variable At A Time: hold a fixed, reasonable embedder constant while
ablating chunking and retrieval, then test the embedder as its own variable,
then choose k from a recall-vs-latency curve, then promote the winner.

This package is import-safe without the heavy ML dependencies installed: model
backends (faiss, sentence-transformers, FlagEmbedding, google-generativeai) are
imported lazily inside the functions that need them, so `import experiments.v2`
never fails just because an optional dependency is missing.

Run order (see README.md):
    Stage 0  -> build_chunk_gt.py        (trustworthy measurement)
    Stage A  -> run_stage.py --stage A   (chunking)
    Stage B  -> run_stage.py --stage B   (retrieval)
    Stage C  -> run_stage.py --stage C   (embedding)
    Stage K  -> select_k.py              (k selection)
    Stage P  -> promote.py + report.py   (promote + report)
"""

__all__ = ["configs", "embedders", "chunkers", "retrievers", "harness"]
