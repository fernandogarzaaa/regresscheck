import asyncio
import json

import pytest

from regresscheck.classifier import OutputParseError, build_messages, classify, parse_output
from regresscheck.contract import FewShotExample, PromptConfig
from regresscheck.providers import Completion, Message

PROMPT = PromptConfig.model_validate(
    {
        "version": "t",
        "created": "2026-01-01T00:00:00Z",
        "system_prompt": "SYS",
        "temperature": 0.3,
        "max_tokens": 77,
        "few_shot": [{"email": "E1", "category": "billing", "summary": "S1"}],
    }
)


def test_build_messages_order_and_roles() -> None:
    msgs = build_messages(PROMPT, "EMAIL")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[0]["content"] == "SYS"
    assert msgs[1]["content"] == "E1"
    assert json.loads(msgs[2]["content"]) == {"category": "billing", "summary": "S1"}
    assert msgs[-1]["content"] == "EMAIL"


def test_build_messages_without_examples() -> None:
    prompt = PROMPT.model_copy(update={"few_shot": []})
    assert build_messages(prompt, "E") == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "E"}]


@pytest.mark.parametrize(
    "text",
    [
        '{"category": "technical", "summary": "App crashes."}',
        '```json\n{"category": "technical", "summary": "App crashes."}\n```',
        'Sure! Here you go: {"category": "TECHNICAL", "summary": "App crashes."} Hope that helps.',
    ],
)
def test_parse_output_accepts_common_shapes(text: str) -> None:
    out = parse_output(text)
    assert out.category.value == "technical"
    assert out.summary == "App crashes."


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("I think it is billing", "no JSON object"),
        ("} {", "no JSON object"),
        ('{"category": "billing", "summary": }', "invalid JSON"),
        ('{"category": "sales", "summary": "x"}', "violates contract"),
        ('{"category": "billing"}', "violates contract"),
    ],
)
def test_parse_output_rejects_contract_violations(text: str, message: str) -> None:
    with pytest.raises(OutputParseError, match=message):
        parse_output(text)


class Recorder:
    name = "rec"
    model = "rec-1"

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[tuple[list[Message], float, int]] = []

    async def complete(self, messages: list[Message], *, temperature: float, max_tokens: int) -> Completion:
        self.calls.append((messages, temperature, max_tokens))
        return Completion(self.text, 11, 3, 42.0)


def test_classify_passes_prompt_settings_and_returns_completion() -> None:
    provider = Recorder('{"category": "account", "summary": "Locked out."}')
    result = asyncio.run(classify("help", PROMPT, provider))
    assert result.output.category.value == "account"
    assert result.completion.latency_ms == 42.0
    messages, temperature, max_tokens = provider.calls[0]
    assert (temperature, max_tokens) == (0.3, 77)
    assert messages[-1] == {"role": "user", "content": "help"}


def test_fewshot_example_model_forbids_extra() -> None:
    with pytest.raises(ValueError):
        FewShotExample.model_validate({"email": "e", "category": "billing", "summary": "s", "x": 1})


def test_parse_output_ignores_trailing_text_right_after_the_object() -> None:
    assert parse_output('{"category": "general", "summary": "Hi."}!').category.value == "general"
