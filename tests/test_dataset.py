"""Invariants of the committed golden dataset and prompts. These are the eval bar; guard them."""

from collections import Counter

from regresscheck.loader import load_dataset, load_prompt

from .conftest import ROOT

DATASET = load_dataset(ROOT / "golden" / "dataset.json")


def test_size_is_within_the_agreed_range() -> None:
    assert 60 <= len(DATASET.cases) <= 100


def test_categories_are_balanced() -> None:
    counts = Counter(c.expected_category.value for c in DATASET.cases)
    assert set(counts) == {"billing", "technical", "account", "general"}
    assert min(counts.values()) >= 15


def test_required_edge_case_tags_are_present() -> None:
    tags = Counter(t for c in DATASET.cases for t in c.tags)
    for tag in ("ambiguous", "short", "typos", "mixed_language", "sarcastic"):
        assert tags[tag] >= 3, tag


def test_every_difficulty_is_represented() -> None:
    assert {c.expected_difficulty.value for c in DATASET.cases} == {"easy", "medium", "hard"}


def test_id_prefix_matches_category() -> None:
    prefix = {"billing": "bil", "technical": "tec", "account": "acc", "general": "gen"}
    for case in DATASET.cases:
        assert case.id.startswith(prefix[case.expected_category.value] + "-"), case.id


def test_reference_summaries_are_one_sentence() -> None:
    for case in DATASET.cases:
        assert case.ideal_summary.rstrip().endswith("."), case.id
        assert len(case.ideal_summary.split()) <= 30, case.id


def test_all_prompts_load() -> None:
    versions = {load_prompt(p).version for p in (ROOT / "prompts").glob("*.yaml")}
    assert {"v1", "v2"} <= versions
    assert load_prompt(ROOT / "demo" / "prompts" / "v3-bad.yaml").version == "v3-bad"
