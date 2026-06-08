# RAG-EDA — Hebrew University MBA Corpus Exploration

Exploratory Data Analysis (EDA) for a Hebrew-language RAG system over 30 Hebrew University MBA sources.

**Goal:** Understand the corpus structure, language characteristics, and data-quality issues *before* any chunking or model decisions.

## Scope

This EDA answers two questions only:
1. **What does the corpus look like?** — content, language, structure, length, metadata.
2. **What data-quality issues exist?** — duplicates, boilerplate, extraction errors, Unicode issues, mixed scripts.

Chunking simulation, model selection, and retrieval evaluation are explicitly **out of scope** here.

## Folder structure

```
RAG-EDA/
  sources.txt          # 30 source URLs grouped by type
  requirements.txt
  src/
    scraper.py         # idempotent HTML/PDF fetcher
    extractor.py       # HTML → clean text + structure metadata; PDF → text
    eda_utils.py       # pure analysis functions (no ML libs)
  data/
    raw/               # cached HTML/PDF bytes as fetched
    clean/             # extracted text (.txt) + metadata (.json) per doc
    corpus.parquet     # one row per source, ~25 feature columns
  eda.py               # pipeline script: fetch → extract → build parquet → verify
  eda.ipynb            # narrated analysis notebook (read from corpus.parquet)
  figures/             # PNG plots saved by the notebook
```

## Quick start

```bash
pip install -r requirements.txt
python eda.py          # scrape, extract, build corpus.parquet
jupyter notebook eda.ipynb   # view analysis
```

## Sources

- **bschool** (6): Official HUJI MBA pages + academic regulations PDF
- **shnaton_roadmap** (7): Program roadmaps from the HUJI course catalogue
- **shnaton_specialization** (17): Specialization pages from the HUJI course catalogue
