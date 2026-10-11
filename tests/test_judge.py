import asyncio

import pytest

from regresscheck.judge import JUDGE_SYSTEM_PROMPT, LexicalJudge, LLMJudge, f1_to_score, parse_judge_score, token_f1
from regresscheck.providers import Completion, Message, ProviderError


def test_token_f1_identical_and_disjoint() -> None:
    assert token_f1("Customer was charged twice.", "Customer was charged twice.") == 1.0
    assert token_f1("refund request", "crash report") == 0.0
    assert token_f1("", "anything") == 0.0
    assert token_f1("anything", "") == 0.0


def test_token_f1_partial_overlap_value() -> None:
    # content stems: ref [refund, duplic, charg], cand [refund, charg, today]; P=R=2/3
    assert token_f1("refund duplicate charge", "refund charge today") == pytest.approx(2 / 3)


def test_token_f1_counts_repeats_once_per_reference_occurrence() -> None:
    # ref [charg], cand [charg, charg]: overlap 1, P=1/2, R=1 -> 2/3
    assert token_f1("charge", "charge charge") == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("f1", "score"),
    [(1.0, 5), (0.6, 5), (0.59, 4), (0.45, 4), (0.44, 3), (0.3, 3), (0.29, 2), (0.15, 2), (0.14, 1), (0.0, 1)],
)
def test_f1_buckets(f1: float, score: int) -> None:
    assert f1_to_score(f1) == score


def test_lexical_judge() -> None:
    result = asyncio.run(LexicalJudge().score("e", "Customer wants a refund.", "Customer wants a refund."))
    assert (result.score, result.tokens, result.error) == (5, 0, None)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"score": 4, "reason": "ok"}', 4),
        ('```json\n{"score": 2}\n```', 2),
        ('{"score": 4.6}', 5),
        ('{"score": 9}', None),
        ('{"score": 0}', None),
        ('{"score": "3"} The answer is 3', 3),
        ("Score: 5", 5),
        ("no idea", None),
        ('{"score": 1}', 1),
        ('{"score": 5}', 5),
        ("[1, 2]", 1),
    ],
)
def test_parse_judge_score(text: str, expected: int | None) -> None:
    assert parse_judge_score(text) == expected


class FakeProvider:
    name = "fake"
    model = "judge-1"

    def __init__(self, text: str = "", fail: bool = False) -> None:
        self.text, self.fail = text, fail
        self.messages: list[Message] = []
        self.kwargs: dict[str, object] = {}

    async def complete(self, messages: list[Message], *, temperature: float, max_tokens: int) -> Completion:
        self.messages = messages
        self.kwargs = {"temperature": temperature, "max_tokens": max_tokens}
        if self.fail:
            raise ProviderError("down")
        return Completion(self.text, 30, 7, 5.0)


def test_llm_judge_success_sends_rubric_and_counts_tokens() -> None:
    provider = FakeProvider('{"score": 3}')
    judge = LLMJudge(provider)
    assert judge.name == "llm:judge-1"
    result = asyncio.run(judge.score("EMAIL", "REF", "CAND"))
    assert (result.score, result.tokens, result.error) == (3, 37, None)
    assert provider.messages[0] == {"role": "system", "content": JUDGE_SYSTEM_PROMPT}
    assert "EMAIL" in provider.messages[1]["content"]
    assert "REFERENCE: REF" in provider.messages[1]["content"]
    assert "CANDIDATE: CAND" in provider.messages[1]["content"]
    assert provider.kwargs == {"temperature": 0.0, "max_tokens": 200}


def test_llm_judge_unparseable_reply() -> None:
    result = asyncio.run(LLMJudge(FakeProvider("dunno")).score("e", "r", "c"))
    assert result.score == 1
    assert result.tokens == 37
    assert result.error is not None and "unparseable" in result.error


def test_llm_judge_provider_failure() -> None:
    result = asyncio.run(LLMJudge(FakeProvider(fail=True)).score("e", "r", "c"))
    assert (result.score, result.tokens) == (1, 0)
    assert result.error is not None and "down" in result.error


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"reason": "4 problems", "score": 1}', 1),
        ('{"reason": "2 nits", "score": 5}', 5),
        ('{"score": 5.5}', 5),
        ("{oops} score 4", 4),
    ],
)
def test_parse_judge_score_boundaries(text: str, expected: int) -> None:
    assert parse_judge_score(text) == expected


def test_token_f1_consumes_one_reference_occurrence_per_match() -> None:
    assert token_f1("charge charge", "charge charge") == 1.0
