"""The LLM feature under test: support-email classifier (category + one-sentence summary)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from .contract import ClassifierOutput, PromptConfig
from .providers import Completion, Message, Provider

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


class OutputParseError(ValueError):
    pass


@dataclass(frozen=True)
class Classification:
    output: ClassifierOutput
    completion: Completion


def build_messages(prompt: PromptConfig, email: str) -> list[Message]:
    messages: list[Message] = [{"role": "system", "content": prompt.system_prompt}]
    for ex in prompt.few_shot:
        messages.append({"role": "user", "content": ex.email})
        messages.append(
            {"role": "assistant", "content": json.dumps({"category": ex.category.value, "summary": ex.summary})}
        )
    messages.append({"role": "user", "content": email})
    return messages


def parse_output(text: str) -> ClassifierOutput:
    """Accept bare JSON, fenced JSON, or JSON embedded in prose. Anything else is a contract violation."""
    cleaned = _FENCE_RE.sub("", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end < start:
        raise OutputParseError(f"no JSON object in output: {text[:120]!r}")
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise OutputParseError(f"invalid JSON: {exc.msg}") from exc
    try:
        return ClassifierOutput.model_validate(data)
    except ValidationError as exc:
        raise OutputParseError(f"output violates contract: {exc.errors()[0]['msg']}") from exc


async def classify(email: str, prompt: PromptConfig, provider: Provider) -> Classification:
    completion = await provider.complete(
        build_messages(prompt, email), temperature=prompt.temperature, max_tokens=prompt.max_tokens
    )
    return Classification(parse_output(completion.text), completion)
