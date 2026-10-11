import pytest
from pydantic import ValidationError

from regresscheck.contract import CATEGORIES, ClassifierOutput, GoldenCase, GoldenDataset, PromptConfig


def _case(case_id: str = "bil-001") -> dict[str, object]:
    return {
        "id": case_id,
        "input": "x",
        "expected_category": "billing",
        "ideal_summary": "y",
        "expected_difficulty": "easy",
        "notes": "n",
    }


def test_categories_are_the_four_routing_queues() -> None:
    assert CATEGORIES == ("billing", "technical", "account", "general")


def test_classifier_output_normalises_category_case_and_whitespace() -> None:
    out = ClassifierOutput.model_validate({"category": "  Billing ", "summary": "  Charged twice.  "})
    assert out.category.value == "billing"
    assert out.summary == "Charged twice."


@pytest.mark.parametrize("summary", ["", "   "])
def test_classifier_output_rejects_blank_summary(summary: str) -> None:
    with pytest.raises(ValidationError):
        ClassifierOutput.model_validate({"category": "billing", "summary": summary})


def test_classifier_output_rejects_unknown_category() -> None:
    with pytest.raises(ValidationError):
        ClassifierOutput.model_validate({"category": "sales", "summary": "x"})


def test_classifier_output_ignores_extra_keys_and_non_string_category_is_rejected() -> None:
    assert ClassifierOutput.model_validate({"category": "general", "summary": "x", "confidence": 0.9})
    with pytest.raises(ValidationError):
        ClassifierOutput.model_validate({"category": 3, "summary": "x"})


def test_golden_case_id_format_is_enforced() -> None:
    GoldenCase.model_validate(_case("tec-010"))
    with pytest.raises(ValidationError):
        GoldenCase.model_validate(_case("TEC-10"))


def test_golden_dataset_rejects_duplicate_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate case id: bil-001"):
        GoldenDataset.model_validate({"version": "1", "updated": "x", "cases": [_case(), _case()]})


def test_prompt_config_bounds() -> None:
    base = {"version": "v1", "created": "2026-01-01T00:00:00Z", "system_prompt": "s"}
    assert PromptConfig.model_validate(base).temperature == 0.0
    with pytest.raises(ValidationError):
        PromptConfig.model_validate({**base, "temperature": 3})
    with pytest.raises(ValidationError):
        PromptConfig.model_validate({**base, "max_tokens": 0})
    with pytest.raises(ValidationError):
        PromptConfig.model_validate({**base, "surprise": 1})
