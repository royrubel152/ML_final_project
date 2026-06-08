from __future__ import annotations

from evaluation_framework.schemas import EvalSample, JudgeVerdict
from evaluation_framework.judges._shared import (
    load_template, format_chat_history, parse_verdict
)

METRIC = "completeness"


def build_prompt(sample: EvalSample) -> str:
    template = load_template(METRIC)
    return template.format(
        chat_history_block=format_chat_history(sample.chat_history),
        question=sample.question,
        answer=sample.answer or "(no answer provided)",
        ground_truth_answer=sample.ground_truth_answer,
    )


def score(sample: EvalSample, judge) -> JudgeVerdict:
    prompt = build_prompt(sample)
    verdict = judge.score(prompt)
    verdict.metric = METRIC
    return verdict


def parse_verdict_str(raw: str, judge_name: str) -> JudgeVerdict:
    return parse_verdict(raw, judge_name, METRIC)
