import pytest

from regresscheck.config import Thresholds
from regresscheck.diff import Status, compare, mcnemar_exact_p, worst

from .conftest import make_case, make_run

T = Thresholds()


def run_with(passes: list[bool], scores: list[int] | None = None, run_id: str = "r", version: str = "v") -> object:
    scores = scores or [4] * len(passes)
    cases = [
        make_case(f"bil-{i:03d}", "billing", "billing" if ok else "general", s)
        for i, (ok, s) in enumerate(zip(passes, scores))
    ]
    return make_run(cases, run_id, version)


def test_status_rank_and_worst() -> None:
    assert [s.rank for s in Status] == [0, 1, 2]
    assert worst() is Status.PASS
    assert worst(Status.PASS, Status.WARN) is Status.WARN
    assert worst(Status.CRITICAL, Status.WARN, Status.PASS) is Status.CRITICAL
    assert worst(Status.WARN, Status.PASS) is Status.WARN


@pytest.mark.parametrize(
    ("b", "c", "p"),
    [(0, 0, 1.0), (1, 0, 1.0), (5, 0, 0.0625), (6, 0, 0.03125), (3, 3, 1.0), (10, 2, 0.0386)],
)
def test_mcnemar(b: int, c: int, p: float) -> None:
    assert mcnemar_exact_p(b, c) == pytest.approx(p, abs=1e-4)
    assert mcnemar_exact_p(c, b) == pytest.approx(p, abs=1e-4)


def test_identical_runs_pass() -> None:
    a = run_with([True] * 10)
    d = compare(a, a, T)
    assert d.status is Status.PASS and d.reasons == []
    assert d.regressions == [] and d.improvements == [] and d.summary_drops == []
    assert d.accuracy_delta == 0 and d.mcnemar_p == 1.0
    assert d.compared_cases == 10


def _passes(n_fail: int, n: int = 100) -> list[bool]:
    return [i >= n_fail for i in range(n)]


@pytest.mark.parametrize(
    ("fails", "status"),
    [(0, Status.PASS), (3, Status.PASS), (4, Status.WARN), (8, Status.WARN), (9, Status.CRITICAL)],
)
def test_accuracy_thresholds_are_strictly_greater_than(fails: int, status: Status) -> None:
    d = compare(run_with(_passes(0)), run_with(_passes(fails)), Thresholds(category_warn=1.0))
    assert d.status is status
    assert len(d.regressions) == fails
    if status is not Status.PASS:
        assert d.reasons[0].startswith("accuracy dropped")
        assert ("critical" if status is Status.CRITICAL else "warn") in d.reasons[0]


def test_float_noise_does_not_trip_a_threshold() -> None:
    # 0.92 - 0.89 is 0.030000000000000027 in floating point; that must not count as > 3%.
    before = run_with(_passes(8))
    after = run_with(_passes(11))
    assert compare(before, after, Thresholds(category_warn=1.0)).status is Status.PASS


def test_improvement_is_not_a_regression() -> None:
    d = compare(run_with(_passes(20)), run_with(_passes(0)), T)
    assert d.status is Status.PASS
    assert len(d.improvements) == 20 and d.regressions == []
    assert d.accuracy_delta == pytest.approx(0.2)
    imp = d.improvements[0]
    assert (imp.old_predicted, imp.new_predicted, imp.expected) == ("general", "billing", "billing")


def test_summary_thresholds() -> None:
    base = run_with([True] * 4, [5, 5, 5, 5])
    warn = run_with([True] * 4, [5, 5, 5, 4])  # mean drop 0.25 -> 6.25% of scale
    crit = run_with([True] * 4, [5, 5, 4, 3])  # mean drop 0.75 -> 18.75%
    edge = run_with([True] * 4, [5, 5, 5, 5])
    assert compare(base, warn, T).status is Status.WARN
    assert "summary score dropped 6.2%" in compare(base, warn, T).reasons[0]
    d = compare(base, crit, T)
    assert d.status is Status.CRITICAL and "critical" in d.reasons[0]
    assert [f.case_id for f in d.summary_drops] == ["bil-003"]
    assert compare(base, edge, T).status is Status.PASS
    # exactly at the threshold is not "exceeds"
    assert compare(base, warn, Thresholds(summary_warn=0.0625, summary_critical=0.1)).status is Status.PASS


def test_summary_drop_of_exactly_two_points_is_listed_one_point_is_not() -> None:
    d = compare(run_with([True, True], [5, 5]), run_with([True, True], [3, 4]), T)
    assert [f.case_id for f in d.summary_drops] == ["bil-000"]


def test_per_category_deltas_and_category_warning() -> None:
    old = make_run(
        [make_case("tec-001", "technical", "technical"), make_case("tec-002", "technical", "technical")]
        + [make_case(f"bil-{i:03d}") for i in range(30)]
    )
    new = make_run(
        [make_case("tec-001", "technical", "technical"), make_case("tec-002", "technical", "general")]
        + [make_case(f"bil-{i:03d}") for i in range(30)]
        + [make_case("gen-001", "general", "general")]
    )
    d = compare(old, new, Thresholds(warn=0.5, critical=0.6))
    cats = {c.category: c for c in d.per_category}
    assert set(cats) == {"billing", "technical"}
    assert (cats["technical"].before, cats["technical"].after, cats["technical"].delta, cats["technical"].total) == (
        1.0,
        0.5,
        -0.5,
        2,
    )
    assert cats["billing"].delta == 0
    assert d.status is Status.WARN
    assert d.reasons == ["technical accuracy dropped 50.0% (category warn > 10%)"]
    assert d.added_cases == ["gen-001"] and d.removed_cases == []
    assert d.compared_cases == 32
    # exactly at the category threshold is not a warning
    assert compare(old, new, Thresholds(warn=0.5, critical=0.6, category_warn=0.5)).status is Status.PASS


def test_removed_cases_are_excluded_from_accuracy() -> None:
    old = make_run([make_case("bil-001"), make_case("bil-002", "billing", "general")], "a", "v1")
    new = make_run([make_case("bil-001")], "b", "v2")
    d = compare(old, new, T)
    assert d.removed_cases == ["bil-002"]
    assert (d.accuracy_before, d.accuracy_after) == (1.0, 1.0)
    assert (d.baseline_run_id, d.candidate_run_id, d.baseline_prompt, d.candidate_prompt) == ("a", "b", "v1", "v2")


def test_empty_runs() -> None:
    d = compare(make_run([]), make_run([]), T)
    assert d.status is Status.PASS and d.accuracy_before == 0.0 and d.summary_before == 0.0 and d.per_category == []


def test_flip_carries_both_outputs() -> None:
    old = make_run([make_case("acc-001", "account", "account", 5)])
    new = make_run([make_case("acc-001", "account", None, 1, difficulty="hard")])
    f = compare(old, new, T).regressions[0]
    assert (f.old_predicted, f.new_predicted, f.old_summary, f.new_summary) == ("account", None, "s", None)
    assert (f.old_summary_score, f.new_summary_score, f.difficulty) == (5, 1, "hard")


def test_headline() -> None:
    d = compare(run_with(_passes(0, 10)), run_with(_passes(2, 10), [3] * 10), T)
    assert d.headline() == (
        "2 regression(s), 0 improvement(s); accuracy 100.0% -> 80.0% (-20.0%), summary 4.00 -> 3.00 (-1.00)"
    )
    assert d.summary_delta == -1.0
    assert d.accuracy_delta == pytest.approx(-0.2)


def test_summary_drop_exactly_at_critical_is_only_a_warning() -> None:
    base = run_with([True] * 4, [5, 5, 5, 5])
    cand = run_with([True] * 4, [5, 5, 4, 4])  # drop 0.5 / 4 = 12.5% of scale
    t = Thresholds(summary_warn=0.1, summary_critical=0.125)
    assert compare(base, cand, t).status is Status.WARN
