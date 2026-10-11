import asyncio

import pytest

from regresscheck.contract import GoldenDataset, PromptConfig
from regresscheck.judge import JudgeScore, LexicalJudge
from regresscheck.providers import Completion, Message, MockProvider, ProviderError
from regresscheck.runner import prompt_sha, run_eval

PROMPT = PromptConfig.model_validate(
    {
        "version": "t1",
        "created": "2026-01-01T00:00:00Z",
        "system_prompt": "- billing: refunds\n- technical: crashes\nIf unsure, choose technical.",
    }
)
DATASET = GoldenDataset.model_validate(
    {
        "version": "d9",
        "updated": "x",
        "cases": [
            {
                "id": "bil-001",
                "input": "I want a refund",
                "expected_category": "billing",
                "ideal_summary": "I want a refund.",
                "expected_difficulty": "easy",
                "tags": ["short"],
                "notes": "n",
            },
            {
                "id": "tec-001",
                "input": "It crashes",
                "expected_category": "technical",
                "ideal_summary": "It crashes.",
                "expected_difficulty": "medium",
                "notes": "n",
            },
            {
                "id": "bil-002",
                "input": "the app crashes",
                "expected_category": "billing",
                "ideal_summary": "Unrelated words.",
                "expected_difficulty": "hard",
                "notes": "n",
            },
        ],
    }
)


def test_run_eval_with_mock_provider() -> None:
    record = asyncio.run(run_eval(PROMPT, DATASET, MockProvider(), LexicalJudge(), concurrency=2, track="main"))
    assert [c.case_id for c in record.cases] == ["bil-001", "tec-001", "bil-002"]
    assert [c.passed for c in record.cases] == [True, True, False]
    assert record.cases[2].predicted == "technical"
    assert record.cases[0].tags == ["short"]
    assert record.cases[0].difficulty == "easy"
    assert record.cases[0].summary == "I want a refund."
    assert record.cases[0].summary_score == 5
    assert record.cases[2].summary_score == 1
    assert record.cases[0].latency_ms > 0 and record.cases[0].prompt_tokens > 0
    assert record.summary.accuracy == pytest.approx(2 / 3)
    assert (record.track, record.prompt_version, record.dataset_version) == ("main", "t1", "d9")
    assert (record.provider, record.model, record.judge) == ("mock", "mock-classifier-v1", "lexical-f1")
    assert record.prompt_sha == prompt_sha(PROMPT)
    assert len(record.run_id) == 12


def test_prompt_sha_changes_with_content() -> None:
    other = PROMPT.model_copy(update={"system_prompt": "different"})
    assert prompt_sha(PROMPT) == prompt_sha(PROMPT.model_copy())
    assert prompt_sha(PROMPT) != prompt_sha(other)
    assert len(prompt_sha(PROMPT)) == 12


class Flaky:
    name = "flaky"
    model = "f"

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0

    async def complete(self, messages: list[Message], *, temperature: float, max_tokens: int) -> Completion:
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        email = messages[-1]["content"]
        if "refund" in email:
            raise ProviderError("rate limited")
        if email.split()[-1] == "crashes" and email.startswith("the"):
            return Completion("not json", 1, 1, 1.0)
        return Completion('{"category": "technical", "summary": "It crashes."}', 5, 2, 7.0)


class StubJudge:
    name = "stub"

    async def score(self, email: str, reference: str, candidate: str) -> JudgeScore:
        return JudgeScore(2, 9, "judge note")


def test_run_eval_records_provider_and_parse_errors_and_respects_concurrency() -> None:
    provider = Flaky()
    record = asyncio.run(run_eval(PROMPT, DATASET, provider, StubJudge(), concurrency=1))
    assert provider.peak == 1
    first, second, third = record.cases
    assert first.error == "ProviderError: rate limited"
    assert (first.predicted, first.passed, first.summary, first.summary_score) == (None, False, None, 1)
    assert (first.latency_ms, first.prompt_tokens, first.completion_tokens) == (0.0, 0, 0)
    assert third.error is not None and third.error.startswith("OutputParseError")
    assert second.passed and second.error == "judge note"
    assert (second.summary_score, second.judge_tokens, second.latency_ms) == (2, 9, 7.0)
    assert (second.prompt_tokens, second.completion_tokens) == (5, 2)
    assert record.summary.errors == 3


def test_run_eval_parallelism_and_validation() -> None:
    provider = Flaky()
    asyncio.run(run_eval(PROMPT, DATASET, provider, StubJudge(), concurrency=3))
    assert provider.peak == 3
    with pytest.raises(ValueError, match="concurrency"):
        asyncio.run(run_eval(PROMPT, DATASET, provider, StubJudge(), concurrency=0))


def test_default_concurrency_is_eight() -> None:
    import inspect

    assert inspect.signature(run_eval).parameters["concurrency"].default == 8
