from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from regresscheck.records import CaseResult, RunRecord, summarise

ROOT = Path(__file__).resolve().parent.parent


def make_case(
    case_id: str,
    expected: str = "billing",
    predicted: str | None = "billing",
    summary_score: int = 4,
    latency_ms: float = 100.0,
    difficulty: str = "easy",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        expected=expected,
        predicted=predicted,
        passed=predicted == expected,
        summary="s" if predicted else None,
        summary_score=summary_score,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        difficulty=difficulty,
    )


def make_run(cases: list[CaseResult], run_id: str = "run", prompt_version: str = "v1") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        created_at="2026-10-11T00:00:00+00:00",
        prompt_version=prompt_version,
        prompt_sha="abc",
        dataset_version="d1",
        provider="mock",
        model="m",
        judge="lexical-f1",
        cases=cases,
        summary=summarise(cases),
    )


@pytest.fixture
def case_factory() -> Callable[..., CaseResult]:
    return make_case


@pytest.fixture
def run_factory() -> Callable[..., RunRecord]:
    return make_run
