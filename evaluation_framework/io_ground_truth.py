"""Load ground-truth CSV files into EvalSample lists."""
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


def load_qa(path: str | Path) -> list[EvalSample]:
    """Load ground_truth_mba_qa.csv. Returns single-turn EvalSamples."""
    samples = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(EvalSample(
                sample_id=row["sample_id"].strip(),
                question=row["question"].strip(),
                ground_truth_answer=row["ground_truth_answer"].strip(),
                ground_truth_sources=_parse_sources(row.get("source_urls", "")),
                category=row.get("category", "").strip() or None,
                secretary_tag=row.get("secretary_tag", "").strip() or None,
                source_type="single_turn",
                chat_history=[],
            ))
    return samples


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
