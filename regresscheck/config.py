"""Thresholds and paths, from regresscheck.toml with environment-variable overrides."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Thresholds:
    # Absolute drops in accuracy (0.03 == 3 percentage points). "Exceeds" means strictly greater.
    warn: float = 0.03
    critical: float = 0.08
    # Drop in mean judge score, normalised to 0-1 by dividing by the 4-point scale width.
    summary_warn: float = 0.03
    summary_critical: float = 0.08
    # Any single category losing more than this much accuracy is at least a warning.
    category_warn: float = 0.10
    # Slow drift: N-run moving average of accuracy vs its own peak.
    drift_window: int = 7
    drift_max_drop: float = 0.03
    drift_min_accuracy: float | None = None

    def validate(self) -> Thresholds:
        for name in ("warn", "critical", "summary_warn", "summary_critical", "category_warn", "drift_max_drop"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ConfigError(f"{name} must be between 0 and 1, got {value}")
        if self.warn > self.critical:
            raise ConfigError(f"warn ({self.warn}) must not exceed critical ({self.critical})")
        if self.summary_warn > self.summary_critical:
            raise ConfigError("summary_warn must not exceed summary_critical")
        if self.drift_window < 2:
            raise ConfigError("drift_window must be >= 2")
        if self.drift_min_accuracy is not None and not 0 <= self.drift_min_accuracy <= 1:
            raise ConfigError("drift_min_accuracy must be between 0 and 1")
        return self


@dataclass(frozen=True)
class Settings:
    thresholds: Thresholds = field(default_factory=Thresholds)
    dataset: str = "golden/dataset.json"
    prompts_dir: str = "prompts"
    active_prompt: str = "prompts/v2.yaml"
    db_path: str = ".regresscheck/history.db"
    baseline_dir: str = "baselines"
    report_dir: str = "reports"
    concurrency: int = 8
    track: str = "default"


ENV_THRESHOLDS = {
    "REGRESSCHECK_WARN_THRESHOLD": "warn",
    "REGRESSCHECK_CRITICAL_THRESHOLD": "critical",
    "REGRESSCHECK_SUMMARY_WARN_THRESHOLD": "summary_warn",
    "REGRESSCHECK_SUMMARY_CRITICAL_THRESHOLD": "summary_critical",
    "REGRESSCHECK_CATEGORY_WARN_THRESHOLD": "category_warn",
    "REGRESSCHECK_DRIFT_WINDOW": "drift_window",
    "REGRESSCHECK_DRIFT_MAX_DROP": "drift_max_drop",
    "REGRESSCHECK_DRIFT_MIN_ACCURACY": "drift_min_accuracy",
}


def _coerce(name: str, raw: Any) -> Any:
    try:
        if name == "drift_window":
            return int(raw)
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name}: expected a number, got {raw!r}") from exc


def load_settings(path: str | Path | None = "regresscheck.toml", env: dict[str, str] | None = None) -> Settings:
    env = dict(os.environ) if env is None else env
    data: dict[str, Any] = {}
    if path is not None and Path(path).is_file():
        with open(path, "rb") as fh:
            data = tomllib.load(fh).get("regresscheck", {})

    known_thresholds = {f.name for f in fields(Thresholds)}
    raw_thresholds = data.pop("thresholds", {})
    unknown = set(raw_thresholds) - known_thresholds
    if unknown:
        raise ConfigError(f"unknown threshold keys: {sorted(unknown)}")
    thresholds = Thresholds(**{k: _coerce(k, v) for k, v in raw_thresholds.items()})
    for var, name in ENV_THRESHOLDS.items():
        if env.get(var, "") != "":
            thresholds = replace(thresholds, **{name: _coerce(name, env[var])})

    known_settings = {f.name for f in fields(Settings)} - {"thresholds"}
    unknown = set(data) - known_settings
    if unknown:
        raise ConfigError(f"unknown settings keys: {sorted(unknown)}")
    settings = Settings(thresholds=thresholds.validate(), **data)
    if env.get("REGRESSCHECK_TRACK"):
        settings = replace(settings, track=env["REGRESSCHECK_TRACK"])
    if settings.concurrency < 1:
        raise ConfigError("concurrency must be >= 1")
    return settings
