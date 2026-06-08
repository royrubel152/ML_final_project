"""
eda.py — Pipeline entry point for the Hebrew University MBA corpus EDA.

Pipeline:
  1. Load source list from sources.txt
  2. Fetch all 30 URLs (cached to data/raw/)
  3. Extract text + structure from each raw file (saved to data/clean/)
  4. Compute all EDA features per document
  5. Build corpus.parquet (one row per source, ~25 feature columns)
  6. Run verify_corpus() guardrails

Usage:
  python eda.py                  # full run
  python eda.py --skip-fetch     # skip fetching (use existing data/raw/)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# ── Project paths ──────────────────────────────────────────────────────────────
RAG_EDA_DIR = Path(__file__).parent
SOURCES_FILE = RAG_EDA_DIR / "sources.txt"
DATA_DIR = RAG_EDA_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
CORPUS_PARQUET = DATA_DIR / "corpus.parquet"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eda")


def run_pipeline(skip_fetch: bool = False) -> pd.DataFrame:
    """Execute the full EDA pipeline and return the corpus DataFrame."""

    # ── Imports ──────────────────────────────────────────────────────────────
    from src.scraper import fetch_all, load_sources
    from src.extractor import extract_html, extract_json, extract_pdf
    from src.eda_utils import (
        compute_language_ratios,
        compute_length_stats,
        count_structural_signals,
        compute_numeric_density,
        extract_domain_entities,
        count_paragraph_language_switches,
        compute_niqqud_ratio,
        measure_prefix_attachment,
        find_boilerplate_ngrams,
        compute_pairwise_jaccard,
        audit_unicode_normalization,
        sample_sentence_segmentation,
        inspect_pdf_direction,
        verify_corpus,
    )

    # ── Step 1: Load sources ──────────────────────────────────────────────────
    logger.info("Loading sources from %s", SOURCES_FILE)
    sources = load_sources(SOURCES_FILE)
    logger.info("Loaded %d sources", len(sources))

    # ── Step 2: Fetch ─────────────────────────────────────────────────────────
    if skip_fetch:
        logger.info("--skip-fetch: loading existing manifest")
        manifest_path = RAW_DIR / "manifest.json"
        if not manifest_path.exists():
            logger.error("No manifest found at %s; run without --skip-fetch first", manifest_path)
            sys.exit(1)
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    else:
        logger.info("Fetching %d URLs → %s", len(sources), RAW_DIR)
        manifest = fetch_all(sources, RAW_DIR)

    # ── Step 3: Extract + compute per-doc features ────────────────────────────
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for url, meta in manifest.items():
        if meta.get("status") == "error" or not meta.get("local_path"):
            logger.warning("Skipping failed fetch: %s", url)
            rows.append(_error_row(url, meta))
            continue

        local_path = Path(meta["local_path"])
        if not local_path.exists():
            logger.warning("Raw file missing for %s; skipping", url)
            rows.append(_error_row(url, meta))
            continue

        logger.info("Extracting: %s", local_path.name)

        # Extract text + structure
        try:
            ext = meta.get("extension", "")
            if ext == ".pdf":
                extracted = extract_pdf(local_path)
            elif ext == ".json":
                extracted = extract_json(local_path)
            else:
                extracted = extract_html(local_path)
        except Exception as exc:
            logger.error("Extraction failed for %s: %s", url, exc)
            rows.append(_error_row(url, meta))
            continue

        text = extracted["text"]

        # Save clean text to data/clean/
        clean_txt_path = CLEAN_DIR / f"{local_path.stem}.txt"
        clean_txt_path.write_text(text, encoding="utf-8")

        # Save metadata JSON alongside
        clean_meta_path = CLEAN_DIR / f"{local_path.stem}.json"
        doc_meta = {
            "url": url,
            "group": meta["group"],
            "title": extracted["title"],
            "headings": extracted["headings"][:20],  # cap list length for storage
            "tables_count": extracted["tables_count"],
            "list_items_count": extracted["list_items_count"],
            "last_modified": meta.get("last_modified"),
            "content_type": meta.get("content_type"),
            "fetched_at": meta.get("fetched_at"),
            "size_bytes": meta.get("size_bytes"),
            "extraction_warnings": extracted["extraction_warnings"],
        }
        with open(clean_meta_path, "w", encoding="utf-8") as fh:
            json.dump(doc_meta, fh, ensure_ascii=False, indent=2)

        # ── Compute all EDA features ──────────────────────────────────────────
        length_stats = compute_length_stats(text)
        structural = count_structural_signals(text)
        numeric_density = compute_numeric_density(text)
        entities = extract_domain_entities(text)
        lang_ratios = compute_language_ratios(text)
        lang_switches = count_paragraph_language_switches(text)
        niqqud = compute_niqqud_ratio(text)
        prefix = measure_prefix_attachment(text)
        normalization = audit_unicode_normalization(text)

        # Sentence segmentation sample (run on first 5000 chars to keep it fast)
        seg_sample = sample_sentence_segmentation(text[:5000], n=20)

        # PDF direction check
        if meta["extension"] == ".pdf":
            pdf_direction = inspect_pdf_direction(text[:2000])
            direction_ok = pdf_direction["direction_likely_correct"]
            direction_msg = pdf_direction["message"]
        else:
            direction_ok = None
            direction_msg = None

        # ── Assemble row ──────────────────────────────────────────────────────
        row = {
            # Identity
            "url": url,
            "group": meta["group"],
            "title": extracted["title"],
            "extension": meta["extension"],
            "last_modified": meta.get("last_modified"),
            "fetched_at": meta.get("fetched_at"),
            "raw_size_bytes": meta.get("size_bytes"),
            # Text length
            **length_stats,
            # Structure
            **structural,
            # Numeric density
            **numeric_density,
            # Domain entities
            **entities,
            # Language
            **lang_ratios,
            **lang_switches,
            # Hebrew-specific
            **niqqud,
            "prefix_variant_pair_count": prefix["prefix_variant_pair_count"],
            # Data quality
            "needs_nfc_normalization": normalization["needs_normalization"],
            "nfc_changed_char_count": normalization["changed_char_count"],
            "sentence_false_split_rate": seg_sample["estimated_false_split_rate"],
            "total_sentences_found": seg_sample["total_sentences_found"],
            # PDF-specific
            "pdf_direction_ok": direction_ok,
            "pdf_direction_message": direction_msg,
            # Warnings
            "extraction_warnings": "; ".join(extracted["extraction_warnings"]),
            # Keep text for corpus-level analysis below
            "text": text,
        }
        rows.append(row)
        logger.info(
            "  ✓ %s | %d words | hebrew=%.1f%% | warnings: %d",
            meta["group"],
            length_stats["word_count"],
            lang_ratios["hebrew_ratio"] * 100,
            len(extracted["extraction_warnings"]),
        )

    corpus_df = pd.DataFrame(rows)

    # ── Step 4: Corpus-level analyses (require all docs together) ─────────────
    texts = corpus_df["text"].tolist()

    logger.info("Computing boilerplate n-grams across all docs...")
    boilerplate_df = find_boilerplate_ngrams(texts, n=8, min_docs=5)
    boilerplate_path = DATA_DIR / "boilerplate_ngrams.parquet"
    boilerplate_df.to_parquet(boilerplate_path, index=False)
    logger.info("Saved boilerplate n-grams → %s (%d rows)", boilerplate_path.name, len(boilerplate_df))

    logger.info("Computing pairwise Jaccard similarity matrix...")
    import numpy as np
    jaccard_matrix = compute_pairwise_jaccard(texts, shingle_size=5)
    np.save(DATA_DIR / "jaccard_matrix.npy", jaccard_matrix)
    logger.info("Saved Jaccard matrix → jaccard_matrix.npy")

    # Save boilerplate_ngrams count as a column on corpus_df
    # (number of boilerplate n-grams that appear in each doc)
    if not boilerplate_df.empty:
        boilerplate_set = set(boilerplate_df["ngram"].tolist())

        def _count_boilerplate_hits(text: str) -> int:
            return sum(1 for ng in boilerplate_set if ng in text)

        corpus_df["boilerplate_ngram_hits"] = corpus_df["text"].apply(_count_boilerplate_hits)
    else:
        corpus_df["boilerplate_ngram_hits"] = 0

    # Drop raw text from parquet (keep clean .txt files on disk instead)
    corpus_df = corpus_df.drop(columns=["text"])

    # ── Step 5: Save corpus.parquet ───────────────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    corpus_df.to_parquet(CORPUS_PARQUET, index=False)
    logger.info("Saved corpus → %s (%d rows, %d cols)", CORPUS_PARQUET, len(corpus_df), len(corpus_df.columns))

    # ── Step 6: Verify corpus ─────────────────────────────────────────────────
    logger.info("Running corpus verification...")
    try:
        verify_corpus(corpus_df)
    except ValueError as exc:
        logger.warning("Corpus verification issues:\n%s", exc)

    return corpus_df


def _error_row(url: str, meta: dict) -> dict:
    """Return a skeleton row for a source that failed to fetch/extract."""
    return {
        "url": url,
        "group": meta.get("group", "unknown"),
        "title": None,
        "extension": meta.get("extension"),
        "last_modified": None,
        "fetched_at": meta.get("fetched_at"),
        "raw_size_bytes": 0,
        "char_count": 0,
        "word_count": 0,
        "paragraph_count": 0,
        "avg_paragraph_length_words": 0.0,
        "h1_count": 0,
        "h2_count": 0,
        "h3_count": 0,
        "list_item_count": 0,
        "table_row_count": 0,
        "digit_char_ratio": 0.0,
        "numeric_line_ratio": 0.0,
        "course_code_count": 0,
        "credit_mention_count": 0,
        "semester_mention_count": 0,
        "gpa_mention_count": 0,
        "year_mention_count": 0,
        "hebrew_ratio": 0.0,
        "english_ratio": 0.0,
        "digit_ratio": 0.0,
        "other_ratio": 0.0,
        "mean_lang_switches_per_para": 0.0,
        "max_lang_switches_in_para": 0,
        "niqqud_char_count": 0,
        "niqqud_ratio": 0.0,
        "prefix_variant_pair_count": 0,
        "needs_nfc_normalization": False,
        "nfc_changed_char_count": 0,
        "sentence_false_split_rate": 0.0,
        "total_sentences_found": 0,
        "pdf_direction_ok": None,
        "pdf_direction_message": None,
        "extraction_warnings": "fetch_failed",
        "text": "",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Hebrew University MBA corpus EDA pipeline.")
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip fetching URLs; use existing files in data/raw/",
    )
    args = parser.parse_args()

    corpus_df = run_pipeline(skip_fetch=args.skip_fetch)

    print("\n── Corpus summary ──────────────────────────────────────────────")
    print(corpus_df[["group", "url", "word_count", "hebrew_ratio", "table_row_count"]].to_string(index=False))
    print(f"\nCorpus saved to: {CORPUS_PARQUET}")
    print("Run 'jupyter notebook eda.ipynb' to view the full analysis.")
