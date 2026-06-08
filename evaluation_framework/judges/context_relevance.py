from __future__ import annotations

from evaluation_framework.schemas import EvalSample, JudgeVerdict
from evaluation_framework.judges._shared import (
    load_template, format_chat_history, format_sources, parse_verdict
)

METRIC = "context_relevance"


def build_prompt(sample: EvalSample) -> str:
    template = load_template(METRIC)
    return template.format(
        chat_history_block=format_chat_history(sample.chat_history),
        question=sample.question,
        retrieved_sources_block=format_sources(sample.retrieved_sources),
    )


def score(sample: EvalSample, judge) -> JudgeVerdict:
    prompt = build_prompt(sample)
    verdict = judge.score(prompt)
    verdict.metric = METRIC
    return verdict


def parse_verdict_str(raw: str, judge_name: str) -> JudgeVerdict:
    return parse_verdict(raw, judge_name, METRIC)
