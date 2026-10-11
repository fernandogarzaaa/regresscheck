"""Run-to-run comparison: the core value of the system."""

from __future__ import annotations

from enum import Enum
from math import comb

from pydantic import BaseModel

from .config import Thresholds
from .contract import CATEGORIES
from .records import CaseResult, RunRecord


class Status(str, Enum):
    PASS = "pass"
    WARN = "warn"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"pass": 0, "warn": 1, "critical": 2}[self.value]


def worst(*statuses: Status) -> Status:
    result = Status.PASS
    for status in statuses:
        if status.rank > result.rank:
            result = status
    return result


class CaseFlip(BaseModel):
    case_id: str
    expected: str
    difficulty: str
    tags: list[str]
    old_predicted: str | None
    new_predicted: str | None
    old_summary: str | None
    new_summary: str | None
    old_summary_score: int
    new_summary_score: int


class CategoryDelta(BaseModel):
    category: str
    before: float
    after: float
    delta: float
    total: int


class DiffResult(BaseModel):
    baseline_run_id: str
    candidate_run_id: str
    baseline_prompt: str
    candidate_prompt: str
    compared_cases: int
    added_cases: list[str]
    removed_cases: list[str]
    accuracy_before: float
    accuracy_after: float
    accuracy_delta: float
    summary_before: float
    summary_after: float
    summary_delta: float
    per_category: list[CategoryDelta]
    regressions: list[CaseFlip]
    improvements: list[CaseFlip]
    summary_drops: list[CaseFlip]
    mcnemar_p: float
    status: Status
    reasons: list[str]

    def headline(self) -> str:
        return (
            f"{len(self.regressions)} regression(s), {len(self.improvements)} improvement(s); "
            f"accuracy {self.accuracy_before:.1%} -> {self.accuracy_after:.1%} ({self.accuracy_delta:+.1%}), "
            f"summary {self.summary_before:.2f} -> {self.summary_after:.2f} ({self.summary_delta:+.2f})"
        )


def mcnemar_exact_p(regressions: int, improvements: int) -> float:
    """Two-sided exact McNemar (binomial) test on discordant pairs. Answers: is this flip pattern noise?"""
    n = regressions + improvements
    if n == 0:
        return 1.0
    k = min(regressions, improvements)
    tail: float = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def _flip(old: CaseResult, new: CaseResult) -> CaseFlip:
    return CaseFlip(
        case_id=new.case_id,
        expected=new.expected,
        difficulty=new.difficulty,
        tags=new.tags,
        old_predicted=old.predicted,
        new_predicted=new.predicted,
        old_summary=old.summary,
        new_summary=new.summary,
        old_summary_score=old.summary_score,
        new_summary_score=new.summary_score,
    )


def _acc(cases: list[CaseResult]) -> float:
    return sum(c.passed for c in cases) / len(cases) if cases else 0.0


def _mean_score(cases: list[CaseResult]) -> float:
    return sum(c.summary_score for c in cases) / len(cases) if cases else 0.0


def compare(baseline: RunRecord, candidate: RunRecord, thresholds: Thresholds) -> DiffResult:
    old = {c.case_id: c for c in baseline.cases}
    new = {c.case_id: c for c in candidate.cases}
    shared = [cid for cid in new if cid in old]
    old_cases = [old[cid] for cid in shared]
    new_cases = [new[cid] for cid in shared]

    regressions = [_flip(old[c], new[c]) for c in shared if old[c].passed and not new[c].passed]
    improvements = [_flip(old[c], new[c]) for c in shared if not old[c].passed and new[c].passed]
    summary_drops = [_flip(old[c], new[c]) for c in shared if old[c].summary_score - new[c].summary_score >= 2]

    acc_before, acc_after = _acc(old_cases), _acc(new_cases)
    sum_before, sum_after = _mean_score(old_cases), _mean_score(new_cases)

    per_category: list[CategoryDelta] = []
    for cat in CATEGORIES:
        o = [old[c] for c in shared if old[c].expected == cat]
        n = [new[c] for c in shared if new[c].expected == cat]
        if o:
            before, after = _acc(o), _acc(n)
            per_category.append(
                CategoryDelta(category=cat, before=before, after=after, delta=round(after - before, 6), total=len(o))
            )

    status = Status.PASS
    reasons: list[str] = []
    acc_drop = round(acc_before - acc_after, 6)
    if acc_drop > thresholds.critical:
        status = Status.CRITICAL
        reasons.append(f"accuracy dropped {acc_drop:.1%} (critical > {thresholds.critical:.0%})")
    elif acc_drop > thresholds.warn:
        status = Status.WARN
        reasons.append(f"accuracy dropped {acc_drop:.1%} (warn > {thresholds.warn:.0%})")

    sum_drop = round((sum_before - sum_after) / 4, 6)
    if sum_drop > thresholds.summary_critical:
        status = worst(status, Status.CRITICAL)
        reasons.append(f"summary score dropped {sum_drop:.1%} of scale (critical > {thresholds.summary_critical:.0%})")
    elif sum_drop > thresholds.summary_warn:
        status = worst(status, Status.WARN)
        reasons.append(f"summary score dropped {sum_drop:.1%} of scale (warn > {thresholds.summary_warn:.0%})")

    for cd in per_category:
        if -cd.delta > thresholds.category_warn:
            status = worst(status, Status.WARN)
            reasons.append(
                f"{cd.category} accuracy dropped {-cd.delta:.1%} (category warn > {thresholds.category_warn:.0%})"
            )

    return DiffResult(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        baseline_prompt=baseline.prompt_version,
        candidate_prompt=candidate.prompt_version,
        compared_cases=len(shared),
        added_cases=sorted(set(new) - set(old)),
        removed_cases=sorted(set(old) - set(new)),
        accuracy_before=acc_before,
        accuracy_after=acc_after,
        accuracy_delta=round(acc_after - acc_before, 6),
        summary_before=sum_before,
        summary_after=sum_after,
        summary_delta=round(sum_after - sum_before, 6),
        per_category=per_category,
        regressions=regressions,
        improvements=improvements,
        summary_drops=summary_drops,
        mcnemar_p=mcnemar_exact_p(len(regressions), len(improvements)),
        status=status,
        reasons=reasons,
    )
