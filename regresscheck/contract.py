"""Typed interface contract between the feature under test and the eval pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"


CATEGORIES: tuple[str, ...] = tuple(c.value for c in Category)


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class FewShotExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str
    category: Category
    summary: str


class PromptConfig(BaseModel):
    """One versioned prompt, loaded from prompts/*.yaml. This is the 'code' CI runs against."""

    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1)
    created: datetime
    description: str = ""
    system_prompt: str = Field(min_length=1)
    few_shot: list[FewShotExample] = Field(default_factory=list)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=200, gt=0)


class ClassifierOutput(BaseModel):
    """What the feature must return for every email."""

    model_config = ConfigDict(extra="ignore")

    category: Category
    summary: str = Field(min_length=1)

    @field_validator("category", mode="before")
    @classmethod
    def _normalise_category(cls, v: object) -> object:
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("summary")
    @classmethod
    def _strip_summary(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("summary must not be blank")
        return v


class GoldenCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z]+-\d{3}$")
    input: str = Field(min_length=1)
    expected_category: Category
    ideal_summary: str = Field(min_length=1)
    expected_difficulty: Difficulty
    tags: list[str] = Field(default_factory=list)
    notes: str = Field(min_length=1)


class GoldenDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    updated: str
    description: str = ""
    cases: list[GoldenCase]

    @field_validator("cases")
    @classmethod
    def _unique_ids(cls, cases: list[GoldenCase]) -> list[GoldenCase]:
        seen: set[str] = set()
        for case in cases:
            if case.id in seen:
                raise ValueError(f"duplicate case id: {case.id}")
            seen.add(case.id)
        return cases
