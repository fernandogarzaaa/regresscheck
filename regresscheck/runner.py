"""Async eval runner: every golden case through the feature, then through the judge."""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from datetime import datetime, timezone

from .classifier import OutputParseError, classify
from .contract import GoldenCase, GoldenDataset, PromptConfig
from .judge import Judge
from .providers import Provider, ProviderError
from .records import CaseResult, RunRecord, summarise


def prompt_sha(prompt: PromptConfig) -> str:
    canonical = json.dumps(prompt.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


async def run_case(case: GoldenCase, prompt: PromptConfig, provider: Provider, judge: Judge) -> CaseResult:
    expected, difficulty, tags = case.expected_category.value, case.expected_difficulty.value, list(case.tags)
    try:
        result = await classify(case.input, prompt, provider)
    except (ProviderError, OutputParseError) as exc:
        return CaseResult(
            case_id=case.id,
            expected=expected,
            difficulty=difficulty,
            tags=tags,
            predicted=None,
            passed=False,
            summary=None,
            summary_score=1,
            latency_ms=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            error=f"{type(exc).__name__}: {exc}",
        )
    output, completion = result.output, result.completion
    verdict = await judge.score(case.input, case.ideal_summary, output.summary)
    return CaseResult(
        case_id=case.id,
        expected=expected,
        difficulty=difficulty,
        tags=tags,
        predicted=output.category.value,
        passed=output.category == case.expected_category,
        summary=output.summary,
        summary_score=verdict.score,
        latency_ms=completion.latency_ms,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        judge_tokens=verdict.tokens,
        error=verdict.error,
    )


async def run_eval(
    prompt: PromptConfig,
    dataset: GoldenDataset,
    provider: Provider,
    judge: Judge,
    *,
    concurrency: int = 8,
    track: str = "default",
) -> RunRecord:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    gate = asyncio.Semaphore(concurrency)

    async def bounded(case: GoldenCase) -> CaseResult:
        async with gate:
            return await run_case(case, prompt, provider, judge)

    cases = await asyncio.gather(*(bounded(c) for c in dataset.cases))
    return RunRecord(
        run_id=uuid.uuid4().hex[:12],
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        track=track,
        prompt_version=prompt.version,
        prompt_sha=prompt_sha(prompt),
        dataset_version=dataset.version,
        provider=provider.name,
        model=provider.model,
        judge=judge.name,
        cases=list(cases),
        summary=summarise(list(cases)),
    )
