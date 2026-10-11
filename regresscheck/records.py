"""Run records: what one eval run produced, per case and in aggregate."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .contract import CATEGORIES


class CaseResult(BaseModel):
    case_id: str
    expected: str
    predicted: str | None
    passed: bool
    summary: str | None
    summary_score: int = Field(ge=1, le=5)
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    judge_tokens: int = 0
    difficulty: str
    tags: list[str] = Field(default_factory=list)
    error: str | None = None


class Bucket(BaseModel):
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


class RunSummary(BaseModel):
    n: int
    accuracy: float
    mean_summary_score: float
    per_category: dict[str, Bucket]
    per_difficulty: dict[str, Bucket]
    latency_p50_ms: float
    latency_p95_ms: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_judge_tokens: int
    errors: int


class RunRecord(BaseModel):
    run_id: str
    created_at: str
    track: str = "default"
    prompt_version: str
    prompt_sha: str
    dataset_version: str
    provider: str
    model: str
    judge: str
    cases: list[CaseResult]
    summary: RunSummary


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (pct in 0..100). Empty input -> 0.0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def summarise(cases: list[CaseResult]) -> RunSummary:
    per_category = {c: Bucket() for c in CATEGORIES}
    per_difficulty: dict[str, Bucket] = {}
    for case in cases:
        for bucket in (
            per_category.setdefault(case.expected, Bucket()),
            per_difficulty.setdefault(case.difficulty, Bucket()),
        ):
            bucket.total += 1
            bucket.correct += int(case.passed)
    n = len(cases)
    latencies = [c.latency_ms for c in cases]
    return RunSummary(
        n=n,
        accuracy=sum(c.passed for c in cases) / n if n else 0.0,
        mean_summary_score=sum(c.summary_score for c in cases) / n if n else 0.0,
        per_category=per_category,
        per_difficulty=dict(sorted(per_difficulty.items())),
        latency_p50_ms=round(percentile(latencies, 50), 1),
        latency_p95_ms=round(percentile(latencies, 95), 1),
        total_prompt_tokens=sum(c.prompt_tokens for c in cases),
        total_completion_tokens=sum(c.completion_tokens for c in cases),
        total_judge_tokens=sum(c.judge_tokens for c in cases),
        errors=sum(c.error is not None for c in cases),
    )
