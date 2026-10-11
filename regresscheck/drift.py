"""Slow-drift detection: a moving average that sags while no single run trips a threshold."""

from __future__ import annotations

from pydantic import BaseModel

from .config import Thresholds


class DriftResult(BaseModel):
    triggered: bool
    runs: int
    window: int
    moving_average: float | None
    peak: float | None
    reason: str | None
    series: list[float | None]


def moving_averages(values: list[float], window: int) -> list[float | None]:
    out: list[float | None] = []
    for i in range(len(values)):
        if i + 1 < window:
            out.append(None)
        else:
            chunk = values[i + 1 - window : i + 1]
            out.append(sum(chunk) / window)
    return out


def detect_drift(accuracies: list[float], thresholds: Thresholds) -> DriftResult:
    """accuracies: chronological, oldest first, current run last."""
    window = thresholds.drift_window
    series = moving_averages(accuracies, window)
    current = series[-1] if series else None
    previous = [v for v in series[:-1] if v is not None]
    peak = max(previous) if previous else None
    reason = None
    if current is not None:
        if peak is not None and round(peak - current, 6) > thresholds.drift_max_drop:
            reason = (
                f"{window}-run moving average {current:.1%} is {peak - current:.1%} below its peak "
                f"{peak:.1%} (drift > {thresholds.drift_max_drop:.0%})"
            )
        elif thresholds.drift_min_accuracy is not None and current < thresholds.drift_min_accuracy:
            reason = f"{window}-run moving average {current:.1%} is below the floor {thresholds.drift_min_accuracy:.0%}"
    return DriftResult(
        triggered=reason is not None,
        runs=len(accuracies),
        window=window,
        moving_average=current,
        peak=peak,
        reason=reason,
        series=series,
    )
