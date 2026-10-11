"""Summary-quality judges. Both return an integer 1-5."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from .providers import Provider, ProviderError, content_stems

JUDGE_SYSTEM_PROMPT = """You grade one-sentence summaries of customer support emails.
Compare the CANDIDATE summary to the REFERENCE summary for the given EMAIL.
Score 1-5:
5 = same key facts and customer intent, nothing wrong or missing
4 = minor omission or wording issue
3 = gets the gist but misses an important detail
2 = mostly wrong or vague
1 = unrelated, empty, or factually wrong
Reply with JSON only: {"score": <1-5>, "reason": "<short>"}"""


@dataclass(frozen=True)
class JudgeScore:
    score: int
    tokens: int = 0
    error: str | None = None


class Judge(Protocol):
    name: str

    async def score(self, email: str, reference: str, candidate: str) -> JudgeScore: ...


def token_f1(reference: str, candidate: str) -> float:
    ref, cand = content_stems(reference), content_stems(candidate)
    if not ref or not cand:
        return 0.0
    ref_counts: dict[str, int] = {}
    for t in ref:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    overlap = 0
    for t in cand:
        if ref_counts.get(t, 0) > 0:
            overlap += 1
            ref_counts[t] -= 1
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(cand), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def f1_to_score(f1: float) -> int:
    for floor, score in ((0.6, 5), (0.45, 4), (0.3, 3), (0.15, 2)):
        if f1 >= floor:
            return score
    return 1


class LexicalJudge:
    """Deterministic stand-in for an LLM judge: content-word F1 against the reference, bucketed to 1-5."""

    name = "lexical-f1"

    async def score(self, email: str, reference: str, candidate: str) -> JudgeScore:
        return JudgeScore(f1_to_score(token_f1(reference, candidate)))


class LLMJudge:
    name = "llm"

    def __init__(self, provider: Provider) -> None:
        self.provider = provider
        self.name = f"llm:{provider.model}"

    async def score(self, email: str, reference: str, candidate: str) -> JudgeScore:
        user = f"EMAIL:\n{email}\n\nREFERENCE: {reference}\nCANDIDATE: {candidate}"
        try:
            completion = await self.provider.complete(
                [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=200,
            )
        except ProviderError as exc:
            return JudgeScore(1, 0, f"judge call failed: {exc}")
        tokens = completion.prompt_tokens + completion.completion_tokens
        score = parse_judge_score(completion.text)
        if score is None:
            return JudgeScore(1, tokens, f"unparseable judge reply: {completion.text[:80]!r}")
        return JudgeScore(score, tokens)


def parse_judge_score(text: str) -> int | None:
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            value = json.loads(match.group(0)).get("score")
        except (ValueError, AttributeError):
            value = None
        if isinstance(value, (int, float)) and 1 <= value <= 5:
            return int(round(value))
    digit = re.search(r"\b([1-5])\b", text)
    return int(digit.group(1)) if digit else None
