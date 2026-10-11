import pytest

from regresscheck.config import Thresholds
from regresscheck.drift import detect_drift, moving_averages

T = Thresholds(drift_window=3, drift_max_drop=0.03)


def test_moving_averages() -> None:
    assert moving_averages([], 3) == []
    assert moving_averages([1.0, 2.0], 3) == [None, None]
    assert moving_averages([1.0, 2.0, 3.0, 4.0], 3) == [None, None, 2.0, 3.0]
    assert moving_averages([1.0, 3.0], 2) == [None, 2.0]


def test_not_enough_history() -> None:
    r = detect_drift([0.9, 0.9], T)
    assert not r.triggered and r.moving_average is None and r.peak is None and r.reason is None
    assert r.runs == 2 and r.window == 3
    assert detect_drift([], T).moving_average is None


def test_first_full_window_has_no_peak() -> None:
    r = detect_drift([0.9, 0.9, 0.5], T)
    assert r.moving_average == pytest.approx(2.3 / 3)
    assert r.peak is None and not r.triggered


def test_slow_drift_triggers_when_ma_falls_below_peak() -> None:
    # each step drops 2 points (below a 3% per-run warn) but the MA sags 4 points from peak
    acc = [0.90, 0.90, 0.90, 0.88, 0.86, 0.84]
    r = detect_drift(acc, T)
    assert r.triggered
    assert r.peak == pytest.approx(0.90)
    assert r.moving_average == pytest.approx(0.86)
    assert r.reason is not None and "below its peak" in r.reason and "3-run" in r.reason
    assert r.series[-1] == pytest.approx(0.86)


def test_drop_exactly_at_threshold_does_not_trigger() -> None:
    acc = [0.90, 0.90, 0.90, 0.87, 0.87, 0.87]
    assert not detect_drift(acc, T).triggered


def test_recovery_clears_drift() -> None:
    acc = [0.90, 0.90, 0.90, 0.80, 0.90, 0.90, 0.90]
    assert not detect_drift(acc, T).triggered


def test_absolute_floor() -> None:
    t = Thresholds(drift_window=3, drift_max_drop=0.5, drift_min_accuracy=0.8)
    r = detect_drift([0.79, 0.79, 0.79], t)
    assert r.triggered and r.reason is not None and "below the floor 80%" in r.reason
    assert not detect_drift([0.8, 0.8, 0.8], t).triggered


def test_floor_is_strict() -> None:
    t = Thresholds(drift_window=3, drift_max_drop=0.5, drift_min_accuracy=0.75)
    assert not detect_drift([0.75, 0.75, 0.75], t).triggered
