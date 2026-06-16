"""Load ground-truth CSV files into EvalSample lists.

Stage 0 fix: ``load_qa`` now reads the *real* column names present in
``ground_truth_mba_qa.csv`` (``id``, ``question_he``, ``answer_he``,
``source_urls``, ``secretary_tag``). The previous version read
``sample_id``/``question``/``ground_truth_answer`` and silently failed against
the actual file, which is one cause of the R@1=0 artifact described in the plan.

Two new helpers support trustworthy measurement:
  * ``attach_chunk_gt`` joins chunk-level gold labels from
    ``chunk_ground_truth.csv`` so the harness can compute chunk-level Recall@k.
  * ``segment_samples`` splits the set into ``in_corpus`` (used for retrieval
    metrics) and ``secretariat_only`` (no source URL; reported separately).
"""
from __future__ import annotations

import csv
from pathlib import Path

from evaluation_framework.schemas import EvalSample


def _parse_sources(raw: str) -> list[str]:
    """Split semicolon-separated URL string; skip blanks."""
    if not raw or not raw.strip():
        return []
    return [u.strip() for u in raw.split(";") if u.strip()]


def _str_to_bool(s: str) -> bool:
    return s.strip().upper() in ("TRUE", "1", "YES")


def _first(row: dict, *keys: str) -> str:
    """Return the first present, non-empty value among ``keys`` (schema-tolerant)."""
    for key in keys:
        if key in row and row[key] is not None and str(row[key]).strip():
            return str(row[key]).strip()
    return ""


def load_qa(path: str | Path) -> list[EvalSample]:
    """Load ground_truth_mba_qa.csv into single-turn EvalSamples.

    Reads the real Hebrew schema (``id``/``question_he``/``answer_he``) while
    staying tolerant of the older ``sample_id``/``question``/``ground_truth_answer``
    names so any legacy CSV still loads.
    """
    samples = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = _first(row, "id", "sample_id")
            question = _first(row, "question_he", "question")
            answer = _first(row, "answer_he", "ground_truth_answer")
            if not sample_id or not question:
                continue
            samples.append(EvalSample(
                sample_id=sample_id,
                question=question,
                ground_truth_answer=answer,
                ground_truth_sources=_parse_sources(_first(row, "source_urls")),
                category=_first(row, "category") or None,
                secretary_tag=_first(row, "secretary_tag") or None,
                source_type="single_turn",
                chat_history=[],
            ))
    return samples


def attach_chunk_gt(
    samples: list[EvalSample],
    chunk_gt_path: str | Path,
) -> list[EvalSample]:
    """Join chunk-level gold labels onto samples (mutates and returns ``samples``).

    Expects a CSV with columns ``sample_id`` and ``chunk_ids`` (semicolon- or
    comma-separated chunk identifiers). Samples without an entry are left with an
    empty ``ground_truth_chunk_ids`` list. Safe to call when the file is absent
    (returns samples unchanged), so the pipeline degrades to URL-level matching.
    """
    path = Path(chunk_gt_path)
    if not path.exists():
        return samples

    by_id: dict[str, list[str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = _first(row, "sample_id", "id")
            raw = _first(row, "chunk_ids", "ground_truth_chunk_ids", "chunk_id")
            if not sid:
                continue
            ids = [c.strip() for c in raw.replace(",", ";").split(";") if c.strip()]
            if ids:
                by_id[sid] = ids

    for sample in samples:
        if sample.sample_id in by_id:
            sample.ground_truth_chunk_ids = by_id[sample.sample_id]
    return samples


def segment_samples(
    samples: list[EvalSample],
) -> tuple[list[EvalSample], list[EvalSample]]:
    """Split into (in_corpus, secretariat_only).

    ``secretariat_only`` = rows with no ground-truth source URL (the answer lives
    only with the secretariat and can never be matched by retrieval). These are
    excluded from retrieval metrics and reported separately, removing the
    unmatchable rows that drag Recall@k toward zero.
    """
    in_corpus, secretariat_only = [], []
    for sample in samples:
        if sample.ground_truth_sources:
            in_corpus.append(sample)
        else:
            secretariat_only.append(sample)
    return in_corpus, secretariat_only


def load_conversations(path: str | Path) -> list[EvalSample]:
    """
    Load ground_truth_mba_conversations.csv.
    Groups rows by conversation_id. Builds chat_history from all turns before
    the is_final_truth_answer=TRUE row. Ground-truth answer and sources come
    from the final-answer row only.
    """
    rows_by_conv: dict[str, list[dict]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["conversation_id"].strip()
            rows_by_conv.setdefault(cid, []).append(row)

    samples = []
    for cid, rows in rows_by_conv.items():
        rows_sorted = sorted(rows, key=lambda r: int(r["turn_index"]))

        final_row = next(
            (r for r in rows_sorted if _str_to_bool(r.get("is_final_truth_answer", ""))),
            None,
        )
        if final_row is None:
            continue

        history = []
        for r in rows_sorted:
            if r is final_row:
                break
            history.append({"role": r["role"].strip(), "content": r["content"].strip()})

        samples.append(EvalSample(
            sample_id=f"{cid}_final",
            question=history[-1]["content"] if history and history[-1]["role"] == "user" else "",
            ground_truth_answer=final_row.get("ground_truth_answer", "").strip(),
            ground_truth_sources=_parse_sources(final_row.get("source_urls", "")),
            chat_history=history,
            category=final_row.get("category", "").strip() or None,
            archetype=final_row.get("archetype", "").strip() or None,
            secretary_tag=final_row.get("secretary_tag", "").strip() or None,
            source_type="conversation",
        ))
    return samples


def load_all(qa_path: str | Path, conv_path: str | Path) -> list[EvalSample]:
    return load_qa(qa_path) + load_conversations(conv_path)
