"""Load prompts (YAML) and the golden dataset (JSON)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import ValidationError

from .contract import GoldenDataset, PromptConfig


class LoadError(ValueError):
    pass


def load_prompt(path: str | Path) -> PromptConfig:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise LoadError(f"{path}: {exc}") from exc
    try:
        return PromptConfig.model_validate(data)
    except ValidationError as exc:
        raise LoadError(f"{path}: invalid prompt config: {exc}") from exc


def load_dataset(path: str | Path) -> GoldenDataset:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LoadError(f"{path}: {exc}") from exc
    try:
        return GoldenDataset.model_validate(data)
    except ValidationError as exc:
        raise LoadError(f"{path}: invalid golden dataset: {exc}") from exc
