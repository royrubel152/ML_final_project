"""
Single-pipeline runner — called as a subprocess by orchestrate.py.
Each pipeline runs in its own process so models are fully unloaded between runs.

Usage:
  python experiments/run_pipeline.py --pipeline P2 --out experiments/results/p2.json
  python experiments/run_pipeline.py --pipeline P3 --out experiments/results/p3.json
  python experiments/run_pipeline.py --pipeline P4 --out experiments/results/p4.json
"""
import os
os.environ["USE_TF"]  = "0"
os.environ["USE_JAX"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from experiments.retrieval_experiment import (
    load_ground_truth, load_semantic_chunks, build_parent_child_chunks,
    build_fixed_size_chunks,
    extract_entities, print_entity_report,
    embed_with_e5, embed_with_gemini, embed_with_minilm,
    build_faiss_index, build_bm25_index,
    _fingerprint, _load_or_embed,
    retrieve_p1, retrieve_p2, retrieve_p3, retrieve_p4,
    evaluate_pipeline,
)

K_VALUES = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def run_p1(samples):
    from sentence_transformers import SentenceTransformer
    print("  Loading MiniLM...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    chunks_p1 = build_fixed_size_chunks(size=500, overlap=50)
    print(f"  Fixed-size chunks: {len(chunks_p1)}")

    fp   = _fingerprint(chunks_p1)
    vecs = _load_or_embed(f"p1_minilm_{fp}", chunks_p1, lambda c: embed_with_minilm(c, model))
    idx  = build_faiss_index(vecs)

    state = {"model_minilm": model, "chunks_p1": chunks_p1, "idx_p1": idx}
    return evaluate_pipeline("P1", retrieve_p1, samples, state)


def run_p2(samples, chunks_sem):
    from dotenv import load_dotenv
    import google.generativeai as genai
    load_dotenv()
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

    fp = _fingerprint(chunks_sem)
    vecs = _load_or_embed(f"p2_gemini_{fp}", chunks_sem, embed_with_gemini)
    idx  = build_faiss_index(vecs)

    state = {"chunks_sem": chunks_sem, "idx_p2": idx}
    return evaluate_pipeline("P2", retrieve_p2, samples, state)


def run_p3(samples, chunks_sem):
    from sentence_transformers import SentenceTransformer
    print("  Loading E5-large...")
    model_e5 = SentenceTransformer("intfloat/multilingual-e5-large")

    fp   = _fingerprint(chunks_sem)
    vecs = _load_or_embed(f"p3_e5_{fp}", chunks_sem, lambda c: embed_with_e5(c, model_e5))
    idx  = build_faiss_index(vecs)
    bm25 = build_bm25_index(chunks_sem)

    state = {"model_e5": model_e5, "chunks_sem": chunks_sem, "idx_p3": idx, "bm25": bm25}
    return evaluate_pipeline("P3", retrieve_p3, samples, state)


def run_p4(samples, chunks_sem):
    from sentence_transformers import SentenceTransformer, CrossEncoder
    print("  Loading E5-large...")
    model_e5 = SentenceTransformer("intfloat/multilingual-e5-large")
    print("  Loading bge-reranker-base...")
    reranker = CrossEncoder("BAAI/bge-reranker-base")

    _, children = build_parent_child_chunks(chunks_sem, child_size=250)
    print(f"  Parent chunks: {len(chunks_sem)} | Child chunks: {len(children)}")

    fpc          = _fingerprint(children)
    vecs_children = _load_or_embed(f"p4_children_e5_{fpc}", children,
                                    lambda c: embed_with_e5(c, model_e5))
    idx_children  = build_faiss_index(vecs_children)
    bm25_children = build_bm25_index(children)

    state = {
        "model_e5":      model_e5,
        "reranker":      reranker,
        "chunks_sem":    chunks_sem,
        "children":      children,
        "idx_children":  idx_children,
        "bm25_children": bm25_children,
    }
    return evaluate_pipeline("P4", retrieve_p4, samples, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True, choices=["P1", "P2", "P3", "P4"])
    parser.add_argument("--out",      required=True)
    args = parser.parse_args()

    print(f"\n[{args.pipeline}] Loading data...")
    samples    = load_ground_truth()
    chunks_sem = load_semantic_chunks()

    if args.pipeline == "P1":
        results = run_p1(samples)
    elif args.pipeline == "P2":
        results = run_p2(samples, chunks_sem)
    elif args.pipeline == "P3":
        results = run_p3(samples, chunks_sem)
    else:
        results = run_p4(samples, chunks_sem)

    # Strip non-serialisable per_category nested dicts for JSON
    out = {k: v for k, v in results.items() if k != "per_category"}
    out["per_category"] = {
        cat: {m: float(v) for m, v in ms.items()}
        for cat, ms in results["per_category"].items()
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  [{args.pipeline}] Results saved → {args.out}")


if __name__ == "__main__":
    main()
