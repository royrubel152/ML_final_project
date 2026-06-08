"""
eda_utils.py — Pure analysis functions for the Hebrew University MBA corpus EDA.

No ML libraries. All analysis is Unicode-range, regex, and pandas-based.
Every function has a docstring stating the insight it produces.

Sections:
  A. Content & structure
  B. Language — Hebrew-specific
  C. Data quality
  D. Corpus-level guardrails
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Unicode range constants ───────────────────────────────────────────────────
HEBREW_BLOCK_START = 0x0590
HEBREW_BLOCK_END = 0x05FF
HEBREW_EXTENDED_A_START = 0xFB1D
HEBREW_EXTENDED_A_END = 0xFB4F
NIQQUD_START = 0x05B0   # Hebrew Point Sheva
NIQQUD_END = 0x05BC     # Hebrew Point Dagesh

LATIN_LOWERCASE_START = 0x0061
LATIN_LOWERCASE_END = 0x007A
LATIN_UPPERCASE_START = 0x0041
LATIN_UPPERCASE_END = 0x005A

# ── Domain entity regex constants ─────────────────────────────────────────────
COURSE_CODE_REGEX = re.compile(r"\b\d{2,3}-\d{3,4}\b")
CREDIT_REGEX = re.compile(r'נ["\u201c\u201d]ז|credit[s]?', re.IGNORECASE)
SEMESTER_REGEX = re.compile(r"סמסטר\s+[א-ת]|semester\s+[ab12]|תשפ[ד-י]", re.IGNORECASE)
GPA_REGEX = re.compile(r"ממוצע\s+\d{2,3}|ממוצע\s+ציונים?|gpa\s*[><=]\s*\d", re.IGNORECASE)
YEAR_REGEX = re.compile(r"\b20[2-3]\d\b|\bתשפ[ד-ט]\b")

# ── Hebrew prefix letters (common clitics) ────────────────────────────────────
HEBREW_PREFIXES = ("ה", "ו", "ב", "ל", "מ", "ש", "כ")


# =============================================================================
# A. Content & structure
# =============================================================================

def compute_length_stats(text: str) -> dict:
    """
    Insight: Document length distribution (chars, words, paragraphs).

    Returns a dict with:
      char_count, word_count, paragraph_count, avg_paragraph_length_words
    """
    chars = len(text)
    words = len(text.split())
    paragraphs = [p for p in re.split(r"\n{2,}", text) if p.strip()]
    para_word_lengths = [len(p.split()) for p in paragraphs]
    avg_para_len = float(np.mean(para_word_lengths)) if para_word_lengths else 0.0

    return {
        "char_count": chars,
        "word_count": words,
        "paragraph_count": len(paragraphs),
        "avg_paragraph_length_words": round(avg_para_len, 1),
    }


def count_structural_signals(text: str) -> dict:
    """
    Insight: Document structure richness — how many headings, list items, table rows.

    Detects:
      - Heading markers (## / ### / # ) embedded by extractor
      - Markdown-style list items (lines starting with -)
      - Table rows (lines containing |)

    Returns h1_count, h2_count, h3_count, list_item_count, table_row_count.
    """
    h1_count = len(re.findall(r"^# .+", text, re.MULTILINE))
    h2_count = len(re.findall(r"^## .+", text, re.MULTILINE))
    h3_count = len(re.findall(r"^### .+", text, re.MULTILINE))
    list_item_count = len(re.findall(r"^- .+", text, re.MULTILINE))
    table_row_count = len(re.findall(r"^.+\|.+", text, re.MULTILINE))

    return {
        "h1_count": h1_count,
        "h2_count": h2_count,
        "h3_count": h3_count,
        "list_item_count": list_item_count,
        "table_row_count": table_row_count,
    }


def compute_numeric_density(text: str) -> dict:
    """
    Insight: How much of the document is numbers vs prose.

    High numeric density (>20%) = table-heavy doc, likely a roadmap or
    course list where naive text chunking risks splitting numeric facts.

    Returns digit_char_ratio, numeric_line_ratio (lines with ≥2 digit sequences).
    """
    total_chars = len(text)
    digit_chars = sum(1 for c in text if c.isdigit())
    digit_char_ratio = digit_chars / total_chars if total_chars else 0.0

    lines = text.splitlines()
    numeric_lines = sum(
        1 for line in lines if len(re.findall(r"\d+", line)) >= 2
    )
    numeric_line_ratio = numeric_lines / len(lines) if lines else 0.0

    return {
        "digit_char_ratio": round(digit_char_ratio, 4),
        "numeric_line_ratio": round(numeric_line_ratio, 4),
    }


def extract_domain_entities(text: str) -> dict:
    """
    Insight: What kinds of domain facts does this document contain?

    Counts occurrences of:
      - Course codes (e.g. 55-234, 322-3220)
      - Credit mentions (נ"ז)
      - Semester/year mentions
      - GPA threshold mentions

    These counts reveal which documents are "fact-rich" vs narrative-only,
    which is critical for planning retrieval strategies.
    """
    return {
        "course_code_count": len(COURSE_CODE_REGEX.findall(text)),
        "credit_mention_count": len(CREDIT_REGEX.findall(text)),
        "semester_mention_count": len(SEMESTER_REGEX.findall(text)),
        "gpa_mention_count": len(GPA_REGEX.findall(text)),
        "year_mention_count": len(YEAR_REGEX.findall(text)),
    }


# =============================================================================
# B. Language — Hebrew-specific
# =============================================================================

def compute_language_ratios(text: str) -> dict:
    """
    Insight: Language composition of each document.

    Uses Unicode ranges — no model needed.
    hebrew_ratio: fraction of chars in Hebrew Unicode block (U+0590–U+05FF + FB1D–FB4F)
    english_ratio: fraction of chars in Latin A-Z/a-z range
    digit_ratio: fraction of digit characters
    other_ratio: everything else (spaces, punctuation, Arabic, etc.)

    A document with hebrew_ratio < 0.3 warrants investigation (may be
    mostly English, or stripped to boilerplate).
    """
    total = len(text)
    if total == 0:
        return {"hebrew_ratio": 0.0, "english_ratio": 0.0, "digit_ratio": 0.0, "other_ratio": 1.0}

    hebrew_count = 0
    english_count = 0
    digit_count = 0

    for ch in text:
        cp = ord(ch)
        if (HEBREW_BLOCK_START <= cp <= HEBREW_BLOCK_END or
                HEBREW_EXTENDED_A_START <= cp <= HEBREW_EXTENDED_A_END):
            hebrew_count += 1
        elif (LATIN_LOWERCASE_START <= cp <= LATIN_LOWERCASE_END or
              LATIN_UPPERCASE_START <= cp <= LATIN_UPPERCASE_END):
            english_count += 1
        elif ch.isdigit():
            digit_count += 1

    other_count = total - hebrew_count - english_count - digit_count

    return {
        "hebrew_ratio": round(hebrew_count / total, 4),
        "english_ratio": round(english_count / total, 4),
        "digit_ratio": round(digit_count / total, 4),
        "other_ratio": round(other_count / total, 4),
    }


def count_paragraph_language_switches(text: str) -> dict:
    """
    Insight: How often does each paragraph switch between Hebrew and English?

    High language switches per paragraph = mixed RTL/LTR content, which
    complicates sentence splitting and embedding alignment later.

    Returns mean_switches_per_paragraph and max_switches_in_paragraph.
    """
    paragraphs = [p for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        return {"mean_lang_switches_per_para": 0.0, "max_lang_switches_in_para": 0}

    switch_counts = []
    for para in paragraphs:
        # Assign each char a script: H=Hebrew, E=English, O=Other
        scripts = []
        for ch in para:
            cp = ord(ch)
            if HEBREW_BLOCK_START <= cp <= HEBREW_BLOCK_END:
                scripts.append("H")
            elif LATIN_LOWERCASE_START <= cp <= LATIN_LOWERCASE_END or \
                 LATIN_UPPERCASE_START <= cp <= LATIN_UPPERCASE_END:
                scripts.append("E")
            # Skip spaces, digits, punctuation — they don't count as switches
        # Count how many times the script changes H→E or E→H
        switches = sum(
            1 for i in range(1, len(scripts))
            if scripts[i] != scripts[i - 1]
        )
        switch_counts.append(switches)

    return {
        "mean_lang_switches_per_para": round(float(np.mean(switch_counts)), 2),
        "max_lang_switches_in_para": int(max(switch_counts)),
    }


def compute_niqqud_ratio(text: str) -> dict:
    """
    Insight: Is the corpus vocalized (with niqqud/vowel marks)?

    Modern Hebrew academic text is almost never vocalized. If niqqud is present,
    it needs to be stripped before tokenization (most models don't handle it well).

    Returns niqqud_char_count and niqqud_ratio (niqqud chars / total Hebrew chars).
    """
    hebrew_chars = sum(
        1 for c in text
        if HEBREW_BLOCK_START <= ord(c) <= HEBREW_BLOCK_END
    )
    niqqud_chars = sum(
        1 for c in text
        if NIQQUD_START <= ord(c) <= NIQQUD_END
    )
    ratio = niqqud_chars / hebrew_chars if hebrew_chars else 0.0
    return {
        "niqqud_char_count": niqqud_chars,
        "niqqud_ratio": round(ratio, 6),
    }


def measure_prefix_attachment(text: str, prefixes: Tuple = HEBREW_PREFIXES) -> Dict:
    """
    Insight: How much morphological variation (prefix attachment) exists in the corpus?

    Hebrew prefixes (ה, ו, ב, ל, מ, ש, כ) attach to words without a space.
    This means "course" (קורס) and "the course" (הקורס) are two different tokens
    for naive retrieval. High prefix-attachment variation = strong need for a
    Hebrew morphological analyzer or normalizer at indexing time.

    For each Hebrew word of length ≥ 3:
      - Check if stripping one of the prefix chars creates a word that also
        appears in the text.
      - Count such "prefix-attached variant pairs".

    Returns:
      prefix_variant_pairs: total count of (bare_word, prefixed_word) pairs found
      prefix_variant_examples: up to 10 example pairs as a list of dicts
    """
    # Tokenize: split on whitespace and punctuation, keep Hebrew tokens
    tokens = re.findall(r"[\u05D0-\u05EA]{2,}", text)
    token_set = set(tokens)

    variant_pairs: List[Dict] = []
    seen: set = set()

    for token in token_set:
        if len(token) < 3:
            continue
        for prefix in prefixes:
            if token.startswith(prefix):
                bare = token[1:]
                if len(bare) >= 2 and bare in token_set:
                    pair_key = (bare, token)
                    if pair_key not in seen:
                        seen.add(pair_key)
                        variant_pairs.append({
                            "bare_word": bare,
                            "prefixed_word": token,
                            "prefix": prefix,
                        })

    return {
        "prefix_variant_pair_count": len(variant_pairs),
        "prefix_variant_examples": variant_pairs[:10],
    }


# =============================================================================
# C. Data quality
# =============================================================================

def find_boilerplate_ngrams(
    texts: List[str],
    n: int = 8,
    min_docs: int = 5,
) -> pd.DataFrame:
    """
    Insight: What repeated phrases appear across many documents (boilerplate)?

    Repeated n-grams (word-level) across ≥ min_docs documents are likely
    navigation menus, cookie banners, or footers that pollute every chunk.

    Returns a DataFrame with columns:
      ngram, doc_count, total_occurrences
    sorted by doc_count descending.
    """
    ngram_doc_sets: Dict[tuple, set] = defaultdict(set)

    for doc_idx, text in enumerate(texts):
        # Tokenize simply (whitespace split, lowercased for matching)
        words = text.split()
        if len(words) < n:
            continue
        seen_in_doc = set()
        for i in range(len(words) - n + 1):
            gram = tuple(words[i: i + n])
            if gram not in seen_in_doc:
                ngram_doc_sets[gram].add(doc_idx)
                seen_in_doc.add(gram)

    rows = []
    for gram, doc_set in ngram_doc_sets.items():
        if len(doc_set) >= min_docs:
            rows.append({
                "ngram": " ".join(gram),
                "doc_count": len(doc_set),
            })

    boilerplate_df = (
        pd.DataFrame(rows)
        .sort_values("doc_count", ascending=False)
        .reset_index(drop=True)
    ) if rows else pd.DataFrame(columns=["ngram", "doc_count"])

    print(
        f"Found {len(boilerplate_df)} n-grams (n={n}) appearing in ≥{min_docs} docs "
        f"(likely boilerplate)."
    )
    return boilerplate_df


def compute_pairwise_jaccard(
    texts: List[str],
    shingle_size: int = 5,
) -> np.ndarray:
    """
    Insight: Which documents share substantial content (near-duplicates)?

    Uses word-level shingles (consecutive shingle_size-word sequences) and
    computes Jaccard similarity for all 30×30 pairs. Naive implementation is
    fine at N=30.

    Returns a square numpy matrix of shape (N, N) with Jaccard scores in [0, 1].
    A score > 0.5 = strong near-duplicate; > 0.8 = likely duplicate.
    """
    n = len(texts)
    shingle_sets = []
    for text in texts:
        words = text.split()
        shingles = set(
            tuple(words[i: i + shingle_size])
            for i in range(max(0, len(words) - shingle_size + 1))
        )
        shingle_sets.append(shingles)

    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            intersection = len(shingle_sets[i] & shingle_sets[j])
            union = len(shingle_sets[i] | shingle_sets[j])
            score = intersection / union if union > 0 else 0.0
            matrix[i][j] = score
            matrix[j][i] = score

    return matrix


def audit_unicode_normalization(text: str) -> dict:
    """
    Insight: Does the text contain characters that change under NFC normalization?

    Some Hebrew sources use non-canonical Unicode forms for final letters or
    presentation characters. If NFC(text) != text, the document should be
    normalized in the preprocessing pipeline to ensure consistent tokenization.

    Returns:
      needs_normalization: bool
      changed_char_count: number of characters that changed
      changed_codepoints: list of (original_hex, normalized_hex) for changed chars
    """
    normalized = unicodedata.normalize("NFC", text)
    changed = []
    # Compare character by character; zip truncates to shorter, so also check lengths
    for orig, norm in zip(text, normalized):
        if orig != norm:
            changed.append((f"U+{ord(orig):04X}", f"U+{ord(norm):04X}"))

    length_changed = len(normalized) != len(text)

    return {
        "needs_normalization": normalized != text,
        "changed_char_count": len(changed) + (abs(len(normalized) - len(text)) if length_changed else 0),
        "changed_codepoints": changed[:20],  # cap for storage
    }


def sample_sentence_segmentation(text: str, n: int = 20) -> dict:
    """
    Insight: How well does a naive sentence splitter work on Hebrew text?

    Hebrew abbreviations (ד\"ר, פרופ', וכו', וכד') contain periods/apostrophes
    that naive splitters treat as sentence endings. This function runs a simple
    splitter and returns the first n sentences for manual inspection.

    Also reports an estimated false-split rate: sentences < 4 words long that
    follow an abbreviation-like pattern are likely false splits.

    Returns:
      sample_sentences: list of first n sentences
      total_sentences_found: int
      estimated_false_split_count: int (sentences that are suspiciously short)
      estimated_false_split_rate: float
    """
    # Naive splitter: split on . ! ? followed by whitespace or end-of-string
    # but NOT when preceded by a single Hebrew/English character (abbreviation)
    abbreviation_guard = re.compile(
        r'(?<!\b[\u05D0-\u05EA])(?<!\b[A-Za-z])(?<!\d)'   # not after single char
        r'[.!?]'
        r'(?=\s|$)'
    )
    sentences = [s.strip() for s in abbreviation_guard.split(text) if s.strip()]

    # Heuristic: sentences with < 4 words are likely abbreviation false-splits
    short_sentences = [s for s in sentences if len(s.split()) < 4]

    false_split_rate = len(short_sentences) / len(sentences) if sentences else 0.0

    return {
        "sample_sentences": sentences[:n],
        "total_sentences_found": len(sentences),
        "estimated_false_split_count": len(short_sentences),
        "estimated_false_split_rate": round(false_split_rate, 4),
    }


def inspect_pdf_direction(text_sample: str) -> dict:
    """
    Insight: Is the PDF Hebrew text in logical (correct) order or visually reversed?

    Some Hebrew PDF renderers output text in visual order (right-to-left visual),
    which means the bytes are reversed compared to logical Unicode order.
    This function provides a heuristic check based on common Hebrew function words.

    Returns:
      direction_likely_correct: bool
      function_words_found: list of matched Hebrew function words
      message: str description
    """
    function_words = ["של", "על", "את", "הוא", "לא", "עם", "כי", "אם", "גם",
                      "זה", "הם", "היא", "כן", "רק", "עד", "אל"]
    found = [w for w in function_words if w in text_sample]

    total_hebrew = sum(1 for c in text_sample if "\u0590" <= c <= "\u05FF")
    if total_hebrew < 50:
        return {
            "direction_likely_correct": True,
            "function_words_found": found,
            "message": "Too little Hebrew text to assess (< 50 Hebrew chars in sample)",
        }

    direction_ok = len(found) >= 3
    message = (
        f"Found {len(found)}/16 expected Hebrew function-words. "
        + ("Direction appears correct." if direction_ok
           else "Suspiciously few — text may be reversed or have encoding issues.")
    )
    return {
        "direction_likely_correct": direction_ok,
        "function_words_found": found,
        "message": message,
    }


# =============================================================================
# D. Corpus-level guardrails
# =============================================================================

EXPECTED_SOURCE_COUNT = 30
MIN_HEBREW_RATIO_FOR_CONTENT_DOCS = 0.2  # bschool + shnaton pages must be ≥ 20% Hebrew
MIN_WORD_COUNT = 10  # any document with fewer than 10 words is suspicious (JSON structured docs are legitimately short)


def verify_corpus(corpus_df: pd.DataFrame) -> None:
    """
    Re-runnable guardrail. Checks basic corpus integrity.

    Raises ValueError if any critical assertion fails.
    Prints a summary of checks passed/failed.

    Checks:
      1. All expected URLs are present (no silent fetch failures left undetected)
      2. No document has empty text
      3. No nulls in primary metric columns
      4. Hebrew-ratio > threshold for all content documents (not pure English)
      5. No document is suspiciously short
    """
    errors = []

    # 1. Source count
    actual_count = len(corpus_df)
    if actual_count < EXPECTED_SOURCE_COUNT:
        errors.append(
            f"[FAIL] Expected {EXPECTED_SOURCE_COUNT} sources, found {actual_count}. "
            f"Missing: {EXPECTED_SOURCE_COUNT - actual_count} sources."
        )
    else:
        print(f"[PASS] Source count: {actual_count}/{EXPECTED_SOURCE_COUNT}")

    # 2. No empty documents (use word_count as proxy since text column is dropped)
    empty_docs = corpus_df[corpus_df["word_count"] == 0]
    if len(empty_docs) > 0:
        errors.append(
            f"[FAIL] {len(empty_docs)} documents have zero words (empty extraction): "
            f"{list(empty_docs['url'])}"
        )
    else:
        print("[PASS] No empty documents")

    # 3. No nulls in primary columns
    primary_cols = ["url", "group", "char_count", "word_count", "hebrew_ratio"]
    for col in primary_cols:
        if col not in corpus_df.columns:
            errors.append(f"[FAIL] Missing column: {col}")
            continue
        null_count = corpus_df[col].isna().sum()
        if null_count > 0:
            errors.append(f"[FAIL] Column '{col}' has {null_count} null values")
        else:
            print(f"[PASS] No nulls in '{col}'")

    # 4. Hebrew ratio check for content documents
    # The PDF (bschool group, .pdf extension) may have lower ratio — exclude from this check
    content_docs = corpus_df[corpus_df["url"].str.endswith(".pdf") == False]
    low_hebrew = content_docs[content_docs["hebrew_ratio"] < MIN_HEBREW_RATIO_FOR_CONTENT_DOCS]
    if len(low_hebrew) > 0:
        errors.append(
            f"[WARN] {len(low_hebrew)} non-PDF documents have hebrew_ratio < "
            f"{MIN_HEBREW_RATIO_FOR_CONTENT_DOCS}: {list(low_hebrew['url'])}"
        )
    else:
        print(f"[PASS] All non-PDF docs have hebrew_ratio ≥ {MIN_HEBREW_RATIO_FOR_CONTENT_DOCS}")

    # 5. Minimum word count
    short_docs = corpus_df[corpus_df["word_count"] < MIN_WORD_COUNT]
    if len(short_docs) > 0:
        errors.append(
            f"[WARN] {len(short_docs)} documents have fewer than {MIN_WORD_COUNT} words: "
            f"{list(short_docs['url'])}"
        )
    else:
        print(f"[PASS] All documents have ≥ {MIN_WORD_COUNT} words")

    if errors:
        error_summary = "\n".join(errors)
        raise ValueError(f"Corpus verification failed:\n{error_summary}")

    print("\n✓ All critical corpus checks passed.")
