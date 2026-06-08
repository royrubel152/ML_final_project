# EDA Summary — Hebrew MBA Corpus

**Corpus:** 30 sources from the Hebrew University MBA program  
**Run date:** June 7, 2026 (final — complete specialization course data)  
**Pipeline:** `eda.py` → `corpus.parquet` (30 rows × 40 columns) → `eda.ipynb`

> **Data correction history:** Two fixes were applied to the Shnaton scraper before this summary was produced. (1) The first fix added the `/yearly-roadmaps/{id}/thresholds?year=YEAR` endpoint call to fetch course lists for roadmaps and specializations, which required browser-like headers and the `year` parameter. (2) The second fix discovered — by reading the React app's JS bundle — that specializations must be queried via `/specializations/{user_code}/thresholds?activeYear=YEAR&thresholdYear=YEAR&include=courses`, using the user-facing code (e.g. `3111`) rather than the internal database id (`45`). Using the db id returned the wrong entity (a generic humanities roadmap). All numbers in this summary reflect the fully corrected corpus.

* * *

## 1. Corpus Overview

| Group | Sources |
|---|---|
| `shnaton_specialization` | 17 |
| `shnaton_roadmap` | 7 |
| `bschool` | 6 |
| **Total** | **30** |

**Fetch errors: 0.** All 30 sources were successfully acquired and all 9 data-quality guardrails pass.

The corpus spans two structurally different systems. `bschool.huji.ac.il` serves Drupal HTML pages and one PDF — narrative text about admissions, program structure, and regulations. `shnaton.huji.ac.il` serves structured academic catalog data (course groups, credit requirements, semester assignments) via a JSON API.

* * *

## 2. Document Length

| Group | Min words | Median words | Max words |
|---|---|---|---|
| `bschool` | 416 | 1,302 | 20,021 |
| `shnaton_roadmap` | 68 | 658 | 3,265 |
| `shnaton_specialization` | 501 | 1,118 | 2,060 |

**Conclusion:** All three groups now have substantial content. The largest document is the bschool exemptions page (20,021 words — dense course tables). Shnaton specializations range from 501 to 2,060 words — all containing real course catalog data with full course groups, credit requirements, and semester assignments. The two smallest roadmaps (68 and 152 words) are thin tracks (supplementary masters, PhD) with very few courses — this reflects actual program scope, not an extraction gap. For chunking, the bschool exemptions page needs aggressive splitting; Shnaton records need boundary-aware chunking to keep course groups intact.

* * *

## 3. Language Composition

| Group | Hebrew | English | Digits | Other |
|---|---|---|---|---|
| `bschool` | 69.0% | 4.3% | 3.7% | 22.9% |
| `shnaton_roadmap` | 61.3% | 1.0% | 7.4% | 30.2% |
| `shnaton_specialization` | 62.7% | 1.0% | 7.2% | 29.2% |

**Conclusion:** The corpus is solidly Hebrew across all groups. Shnaton documents are slightly lower in Hebrew ratio (61–63%) compared to bschool (69%), which is expected — course catalog records contain more numeric content (course codes, credit counts) and punctuation. The "other" bucket is mostly punctuation and separators. No document falls below 20% Hebrew, confirming the corpus is clean for a Hebrew RAG system.

* * *

## 4. Niqqud (Hebrew Vowel Marks)

- **Documents with niqqud: 1 out of 30**
- The single case is `bschool/ptorim` with just 2 niqqud characters (ratio: ~0.00003) — effectively zero.

**Conclusion:** The corpus is niqqud-free. Modern academic and institutional Hebrew rarely uses vowel marks. No niqqud-stripping preprocessing is needed.

* * *

## 5. Unicode Normalization

- **Documents needing NFC normalization: 0 out of 30**

**Conclusion:** All 30 documents are already in NFC (Unicode Canonical Composition) form. No normalization fix is required before chunking. This is a positive signal — the source data is cleanly encoded.

* * *

## 6. RTL/LTR Language Mixing

| Group | Mean switches/paragraph | Max switches in any paragraph |
|---|---|---|
| `bschool` | 6.91 | 304 |
| `shnaton_roadmap` | 0.83 | 8 |
| `shnaton_specialization` | 0.48 | 8 |

**Conclusion:** Only bschool pages have significant RTL/LTR mixing (average 6.9 direction switches per paragraph, with one paragraph in the exemptions page switching 304 times). This is caused by dense course tables that interleave Hebrew course names with English abbreviations and numeric codes. Standard sentence splitters that assume a single text direction will fragment these paragraphs incorrectly. Shnaton records are relatively clean — course data follows a consistent `code | Hebrew name | credits | semester` structure with minimal prose mixing.

* * *

## 7. Hebrew Morphology — Prefix Variants

| Group | Total prefix-variant pairs |
|---|---|
| `bschool` | ~958 |
| `shnaton_roadmap` | ~275 |
| `shnaton_specialization` | ~460 |
| **All corpus** | **1,693** |

Sample prefix pairs found in the corpus:

| Prefix | Base | Prefixed form |
|---|---|---|
| ב | מנהל | במנהל |
| ל | ניתוח | לניתוח |
| ה | ספרים | הספרים |
| ש | חובה | שחובה |
| ו | מערכות | ומערכות |

**Conclusion:** 1,693 prefix-variant word pairs exist across the corpus. The same concept appears as multiple tokens: "ניהול", "בניהול", "לניהול" are all the same root. A naive BM25 or dense retriever without morphological normalization will miss these variants and return incomplete results for any query involving prefixed forms. This is the most significant linguistic challenge for the retrieval stage.

* * *

## 8. Document Structure

| Group | H1 | H2 | H3 | List items | Table rows |
|---|---|---|---|---|---|
| `bschool` | 0 | 25 | 48 | 61 | 36 |
| `shnaton_roadmap` | 7 | 9 | 31 | 465 | 465 |
| `shnaton_specialization` | 17 | 31 | 128 | 835 | 835 |

**Conclusion:** The full Shnaton course data is richly structured. Each specialization has H1 (program name), H2 (academic info / course section headers), and H3 (year/group sub-headings), with every course rendered as a list item (also counted as a table row in the extractor). The 835 list items across specializations are the individual course entries (`code | name | credits | semester | notes`). This structure is a major asset for RAG: course group headers are natural chunk boundaries, and each course row is an atomic, answerable fact.

* * *

## 9. Numeric Density

| Group | Digit char ratio | Numeric line ratio |
|---|---|---|
| `bschool` | 3.7% | 9.5% |
| `shnaton_roadmap` | 7.4% | 13.1% |
| `shnaton_specialization` | 7.2% | ~7% |

**Conclusion:** Shnaton documents have the highest digit density (~7%), concentrated in course codes (5-digit numbers like 55813), credit counts, and year markers. These values are high-value retrieval targets — questions like "כמה נקודות זכות דרושות בהתמחות X?" rely on exact numeric preservation. Chunks must not split across a `code | name | credits` row.

* * *

## 10. Domain Entity Inventory

| Entity type | bschool | shnaton_roadmap | shnaton_specialization |
|---|---|---|---|
| Course codes (regex) | 154 | 7 | 0 |
| Credit mentions | 224 | 488 | 1,013 |
| Semester mentions | 23 | 320 | 684 |
| GPA mentions | 12 | 0 | 0 |
| Year mentions | 9 | 0 | 0 |

**Note on course codes:** The regex pattern matches 5-digit codes embedded in prose. In Shnaton, course codes appear as structured fields inside the list items and are captured as part of the course row (`- 55813 | אקונומטריקה למימון | 3 נ"ז`) rather than standalone in text — so the regex count is low but the data is fully present in the list structure.

**Conclusion:** Shnaton is now the dominant source for credit and semester data (1,013 and 684 mentions in specializations alone). The bschool pages remain the primary source for GPA rules, admissions policy, and program-level regulations (12 GPA mentions, 154 course code references in prose context). Together, the two source types are complementary: Shnaton answers factual course-level questions; bschool answers policy and process questions.

* * *

## 11. Boilerplate Phrases

- **Total repeated 8-gram phrases: 940**
- Most repeated: `"### שנה 1 — חובה חובה ללמוד לפחות"` — appears in **18 documents**

Top repeated phrases are Shnaton structural headings and requirement phrasing (e.g., "חובה ללמוד לפחות X נ\"ז מקבוצה זו"), appearing identically across all specializations.

**Conclusion:** Two distinct types of boilerplate exist:
1. **bschool navigation noise** (site menus, sidebar HTML) — should be stripped before indexing.
2. **Shnaton structural phrases** (repeated group headings and eligibility instructions) — these are legitimate factual content that should be **retained**, as they answer real user questions about enrollment requirements. The 940 repeated phrases are concentrated in the Shnaton catalog structure and are not noise.

* * *

## 12. Near-Duplicate Detection (Jaccard Similarity)

- **Near-duplicate pairs (Jaccard ≥ 0.5): 8 pairs**
- Highest similarity: **0.969** between `roadmap/322-3220` and `roadmap/322-3222` (research vs. non-research MBA — almost identical course catalogs)

Top 8 most similar pairs:

| Jaccard | Document A | Document B |
|---|---|---|
| 0.969 | roadmap/322-3220 | roadmap/322-3222 |
| 0.801 | specialization/3551 | specialization/3552 |
| 0.764 | specialization/3111 | specialization/3113 |
| 0.763 | specialization/3441 | specialization/3443 |
| 0.693 | specialization/3121 | specialization/3123 |
| 0.677 | specialization/3331 | specialization/3333 |
| 0.649 | roadmap/826-3254 | roadmap/826-3255 |
| 0.567 | specialization/3661 | specialization/3662 |

**Conclusion:** 8 document pairs have substantial overlap. The pattern is consistent: every "primary" (ראשית) and "secondary" (משנית) variant of the same specialization shares a large portion of the same elective course pool. Indexing both without deduplication will inflate retrieval scores for shared courses — a question about course X may retrieve 2 nearly identical chunks. These pairs should be merged or carefully deduplicated before indexing.

* * *

## 13. PDF Extraction

- **Source:** Academic regulations document (`academic_regulations_for_masters_programs_01.pdf`)
- **Word count:** 7,596 | **Character count:** 49,221 | **Hebrew ratio:** 68.3%
- **Direction check:** PASS

Sample extracted text:
```
תקנון לימודים לתואר מוסמך במנהל עסקים – MBA
המסמך שלהלן מיועד לסטודנטים ולסטודנטיות המתחילים את לימודיהם בשנת הלימודים תשפ"ה.
```

**Conclusion:** The PDF extraction is clean and correct. This document contains the most GPA mentions (12) in the corpus and serves as the regulatory backbone of the MBA program — the primary source for policy questions about graduation requirements, grade thresholds, and academic regulations.

* * *

## 14. Sentence Segmentation

| Group | Mean false-split rate |
|---|---|
| `bschool` | ~2.1% |
| `shnaton_roadmap` | ~7.1% |
| `shnaton_specialization` | ~3.4% |

**Conclusion:** Bschool pages have a 2.1% false sentence-split rate caused by Hebrew abbreviations containing quotation marks (`נ"ז`, `תשפ"ה`). Shnaton roadmaps have a higher rate (7.1%) because of their dense numeric notation. Shnaton specializations are at 3.4%. None of these rates are catastrophic, but they confirm that a Hebrew-aware sentence boundary detector is needed — standard punctuation-based splitters will incorrectly break at the `"` in Hebrew abbreviations.

* * *

## Issues Found — Summary

| # | Issue | Severity | Affected Sources |
|---|---|---|---|
| 1 | **Near-duplicates** — 8 document pairs with Jaccard ≥ 0.5 | High | 6 specialization pairs + 2 roadmap pairs |
| 2 | **Hebrew morphology** — 1,693 prefix-variant pairs | High | Entire corpus |
| 3 | **RTL/LTR mixing** — bschool/ptorim has max 304 direction switches in one paragraph | Medium | bschool pages, primarily ptorim |
| 4 | **Boilerplate** — 940 repeated 8-gram phrases | Low-Medium | All Shnaton documents (structural, not noise) |
| 5 | **Sentence false-splits** — up to 7.1% in shnaton_roadmap | Low | All groups; highest in Shnaton roadmaps |

* * *

## Preprocessing Recommendations (for next stage)

| Priority | Recommendation | Rationale |
|---|---|---|
| P1 | **Deduplication** — merge or deduplicate the 8 near-duplicate pairs before indexing | Jaccard 0.57–0.97 overlap; identical course facts will appear in multiple chunks, inflating retrieval scores |
| P2 | **Morphological normalization at index time** — integrate HebMorph, Stanza Hebrew, or a sub-word model | 1,693 prefix-variant pairs; same concept stored as multiple tokens across the corpus |
| P3 | **Selective boilerplate stripping** — strip bschool nav/sidebar noise; retain Shnaton structural phrases | bschool navigation text will pollute chunk embeddings; Shnaton requirement phrases answer real user questions |
| P4 | **Hebrew-aware sentence splitting** — use paragraph-level boundaries for bschool; do not split on `"` within Hebrew abbreviations | 2–7% false-split rate across groups; 304 direction switches in one bschool paragraph |
| P5 | **Structure-aware chunking for Shnaton** — treat each course group (H3 section + its list items) as one chunk | Course group boundaries are the most natural retrieval units; splitting mid-group will break credit-requirement context |
