from pathlib import Path

import pytest

from regresscheck.config import ConfigError, Settings, Thresholds, load_settings


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "rc.toml"
    p.write_text(text)
    return p


def test_defaults_when_file_missing(tmp_path: Path) -> None:
    s = load_settings(tmp_path / "nope.toml", env={})
    assert s == Settings()
    t = s.thresholds
    assert (t.warn, t.critical, t.summary_warn, t.summary_critical) == (0.03, 0.08, 0.03, 0.08)
    assert (t.category_warn, t.drift_window, t.drift_max_drop, t.drift_min_accuracy) == (0.10, 7, 0.03, None)
    assert load_settings(None, env={}) == Settings()


def test_file_values_and_env_overrides(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        '[regresscheck]\nactive_prompt = "prompts/v9.yaml"\nconcurrency = 2\n'
        "[regresscheck.thresholds]\nwarn = 0.05\ncritical = 0.1\ndrift_window = 5\n",
    )
    s = load_settings(path, env={})
    assert (s.active_prompt, s.concurrency) == ("prompts/v9.yaml", 2)
    assert (s.thresholds.warn, s.thresholds.critical, s.thresholds.drift_window) == (0.05, 0.1, 5)
    env = {
        "REGRESSCHECK_WARN_THRESHOLD": "0.01",
        "REGRESSCHECK_CRITICAL_THRESHOLD": "0.2",
        "REGRESSCHECK_SUMMARY_WARN_THRESHOLD": "0.04",
        "REGRESSCHECK_SUMMARY_CRITICAL_THRESHOLD": "0.09",
        "REGRESSCHECK_CATEGORY_WARN_THRESHOLD": "0.5",
        "REGRESSCHECK_DRIFT_WINDOW": "3",
        "REGRESSCHECK_DRIFT_MAX_DROP": "0.02",
        "REGRESSCHECK_DRIFT_MIN_ACCURACY": "0.8",
        "REGRESSCHECK_TRACK": "pr-12",
    }
    s = load_settings(path, env=env)
    t = s.thresholds
    assert (t.warn, t.critical, t.summary_warn, t.summary_critical) == (0.01, 0.2, 0.04, 0.09)
    assert (t.category_warn, t.drift_window, t.drift_max_drop, t.drift_min_accuracy) == (0.5, 3, 0.02, 0.8)
    assert s.track == "pr-12"
    assert isinstance(t.drift_window, int)


def test_empty_env_values_are_ignored(tmp_path: Path) -> None:
    s = load_settings(None, env={"REGRESSCHECK_WARN_THRESHOLD": "", "REGRESSCHECK_TRACK": ""})
    assert s.thresholds.warn == 0.03 and s.track == "default"


def test_reads_os_environ_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REGRESSCHECK_CRITICAL_THRESHOLD", "0.15")
    assert load_settings(None).thresholds.critical == 0.15


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[regresscheck.thresholds]\nwarn = 0.2\ncritical = 0.1\n", "must not exceed critical"),
        ("[regresscheck.thresholds]\nsummary_warn = 0.2\nsummary_critical = 0.1\n", "summary_warn"),
        ("[regresscheck.thresholds]\nwarn = -0.1\n", "warn must be between 0 and 1"),
        ("[regresscheck.thresholds]\ncritical = 1.5\n", "critical must be between"),
        ("[regresscheck.thresholds]\ndrift_window = 1\n", "drift_window"),
        ("[regresscheck.thresholds]\ndrift_min_accuracy = 2\n", "drift_min_accuracy"),
        ("[regresscheck.thresholds]\nwarn = 'high'\n", "expected a number"),
        ("[regresscheck.thresholds]\nbogus = 1\n", "unknown threshold keys"),
        ("[regresscheck]\nbogus = 1\n", "unknown settings keys"),
        ("[regresscheck]\nconcurrency = 0\n", "concurrency"),
    ],
)
def test_invalid_config(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_settings(write(tmp_path, text), env={})


def test_invalid_env_value() -> None:
    with pytest.raises(ConfigError, match="drift_window"):
        load_settings(None, env={"REGRESSCHECK_DRIFT_WINDOW": "seven"})


def test_threshold_boundaries_are_valid() -> None:
    Thresholds(
        warn=0,
        critical=0,
        summary_warn=1,
        summary_critical=1,
        category_warn=0,
        drift_max_drop=1,
        drift_window=2,
        drift_min_accuracy=0,
    ).validate()
    Thresholds(warn=0.08, critical=0.08, drift_min_accuracy=1.0).validate()
    for field in ("summary_warn", "summary_critical", "category_warn", "drift_max_drop"):
        with pytest.raises(ConfigError, match=field):
            Thresholds(**{field: 1.01}).validate()  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        Thresholds(drift_min_accuracy=-0.01).validate()


def test_settings_defaults_and_minimum_concurrency(tmp_path: Path) -> None:
    s = Settings()
    assert (s.dataset, s.prompts_dir, s.active_prompt) == ("golden/dataset.json", "prompts", "prompts/v2.yaml")
    assert (s.db_path, s.baseline_dir, s.report_dir, s.concurrency, s.track) == (
        ".regresscheck/history.db",
        "baselines",
        "reports",
        8,
        "default",
    )
    assert load_settings(write(tmp_path, "[regresscheck]\nconcurrency = 1\n"), env={}).concurrency == 1
