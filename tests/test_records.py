import pytest

from regresscheck.records import Bucket, percentile, summarise

from .conftest import make_case


def test_percentile() -> None:
    assert percentile([], 50) == 0.0
    assert percentile([7.0], 95) == 7.0
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([4, 1, 3, 2], 0) == 1
    assert percentile([4, 1, 3, 2], 100) == 4
    assert percentile([0, 10], 95) == pytest.approx(9.5)
    assert percentile([0, 10, 20], 75) == pytest.approx(15.0)


def test_bucket_accuracy() -> None:
    assert Bucket().accuracy == 0.0
    assert Bucket(correct=3, total=4).accuracy == 0.75


def test_summarise_aggregates_every_dimension() -> None:
    cases = [
        make_case("bil-001", "billing", "billing", 5, 100, "easy", 10, 2),
        make_case("bil-002", "billing", "general", 3, 300, "hard", 20, 4),
        make_case("tec-001", "technical", "technical", 4, 200, "hard", 30, 6),
        make_case("acc-001", "account", None, 1, 0, "medium", 0, 0).model_copy(update={"error": "boom"}),
    ]
    s = summarise(cases)
    assert s.n == 4
    assert s.accuracy == 0.5
    assert s.mean_summary_score == pytest.approx(13 / 4)
    assert s.per_category["billing"].correct == 1 and s.per_category["billing"].total == 2
    assert s.per_category["technical"].accuracy == 1.0
    assert s.per_category["account"].total == 1 and s.per_category["account"].correct == 0
    assert s.per_category["general"].total == 0
    assert list(s.per_category) == ["billing", "technical", "account", "general"]
    assert list(s.per_difficulty) == ["easy", "hard", "medium"]
    assert s.per_difficulty["hard"].correct == 1 and s.per_difficulty["hard"].total == 2
    assert s.latency_p50_ms == 150.0
    assert s.latency_p95_ms == 285.0
    assert (s.total_prompt_tokens, s.total_completion_tokens, s.total_judge_tokens) == (60, 12, 0)
    assert s.errors == 1


def test_summarise_empty() -> None:
    s = summarise([])
    assert (s.n, s.accuracy, s.mean_summary_score, s.latency_p50_ms) == (0, 0.0, 0.0, 0.0)


def test_case_result_score_bounds_and_latency_rounding() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        make_case("bil-001", summary_score=6)
    with pytest.raises(ValidationError):
        make_case("bil-001", summary_score=0)
    s = summarise([make_case("bil-001", latency_ms=100.04), make_case("bil-002", latency_ms=100.06)])
    raw = percentile([100.04, 100.06], 50)
    assert s.latency_p50_ms == round(raw, 1) and s.latency_p50_ms != raw
    assert s.latency_p95_ms == round(percentile([100.04, 100.06], 95), 1)
