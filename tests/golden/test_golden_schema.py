"""
Schema validator for tests/golden/ask_questions.yaml.

Unlike test_ask_eval.py, this module does NOT touch the network. It loads the
golden YAML and enforces structural invariants so a bad edit fails CI
immediately rather than silently corrupting a costly eval run.

Invariants enforced:
  * File parses as a list of dicts.
  * IDs are unique and monotonically numbered Q001, Q002, ...
  * Each entry has every required field with the expected type.
  * `category` is drawn from an allow-list.
  * `expected_route` is drawn from an allow-list.
  * `session_parent_id`, when present, points at an ID that exists earlier in
    the file (a follow-up cannot reference its own seed forward).
  * must_mention / must_not_mention are lists of strings (not a stray scalar).

Runs under default pytest (no env gate).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

GOLDEN_PATH = Path(__file__).parent / "ask_questions.yaml"

ALLOWED_CATEGORIES = {
    # Sprint 0 buckets (Q001..Q050)
    "count", "lookup", "compare", "recommend", "taxonomy",
    "trend", "graph", "health", "synthesis",
    # Sprint 1 expansion (Q051..Q120)
    "adversarial", "ambiguous", "out_of_graph",
    "multi_hop", "follow_up", "empty_result",
}

ALLOWED_ROUTES = {
    # Existing router buckets carried through from Sprint 0.
    "count", "lookup", "compare", "recommend", "taxonomy",
    "trend", "graph", "health", "synthesis",
    # Sprint 1 additions — off_topic is the canonical refusal label mapped to
    # the prompt-injection / off-topic filter in app.routers.intelligence.
    "off_topic", "clarify", "multi_hop", "follow_up",
}

REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "id": str,
    "question": str,
    "category": str,
    "expected_route": str,
    "must_mention": list,
    "must_not_mention": list,
    "expected_source_ids": list,
    "source": str,
    "notes": str,
}

ID_RE = re.compile(r"^Q\d{3}$")


@pytest.fixture(scope="module")
def questions() -> list[dict[str, Any]]:
    with GOLDEN_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, list), f"{GOLDEN_PATH} must be a YAML list"
    return data


def test_all_entries_are_dicts(questions: list[dict[str, Any]]) -> None:
    for i, entry in enumerate(questions):
        assert isinstance(entry, dict), f"entry index {i} is {type(entry)}, expected dict"


def test_ids_unique_and_well_formed(questions: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for entry in questions:
        qid = entry.get("id")
        assert isinstance(qid, str) and ID_RE.match(qid), (
            f"bad id {qid!r}; expected pattern QNNN"
        )
        assert qid not in seen, f"duplicate id {qid}"
        seen.add(qid)


def test_ids_are_contiguous(questions: list[dict[str, Any]]) -> None:
    nums = [int(e["id"][1:]) for e in questions]
    assert nums == list(range(1, len(nums) + 1)), (
        f"IDs must be contiguous Q001..Q{len(nums):03d}; got {nums[:5]}..{nums[-5:]}"
    )


def test_required_fields_and_types(questions: list[dict[str, Any]]) -> None:
    for entry in questions:
        qid = entry.get("id", "?")
        for field, ftype in REQUIRED_FIELDS.items():
            assert field in entry, f"{qid}: missing required field {field!r}"
            value = entry[field]
            assert isinstance(value, ftype), (
                f"{qid}: field {field!r} is {type(value).__name__}, expected {ftype}"
            )


def test_categories_in_allow_list(questions: list[dict[str, Any]]) -> None:
    for entry in questions:
        cat = entry["category"]
        assert cat in ALLOWED_CATEGORIES, (
            f"{entry['id']}: category {cat!r} not in allow-list {sorted(ALLOWED_CATEGORIES)}"
        )


def test_routes_in_allow_list(questions: list[dict[str, Any]]) -> None:
    for entry in questions:
        route = entry["expected_route"]
        assert route in ALLOWED_ROUTES, (
            f"{entry['id']}: expected_route {route!r} not in allow-list {sorted(ALLOWED_ROUTES)}"
        )


def test_mention_lists_are_strings(questions: list[dict[str, Any]]) -> None:
    for entry in questions:
        for field in ("must_mention", "must_not_mention", "expected_source_ids"):
            for i, item in enumerate(entry[field]):
                assert isinstance(item, str), (
                    f"{entry['id']}: {field}[{i}] is {type(item).__name__}, expected str"
                )


def test_session_parent_id_resolves_backward(questions: list[dict[str, Any]]) -> None:
    """A follow-up entry's session_parent_id must name an earlier entry."""
    seen_ids: set[str] = set()
    for entry in questions:
        qid = entry["id"]
        parent = entry.get("session_parent_id")
        if parent is not None:
            assert isinstance(parent, str), (
                f"{qid}: session_parent_id must be str or absent, got {type(parent).__name__}"
            )
            assert parent in seen_ids, (
                f"{qid}: session_parent_id {parent!r} not found in prior entries"
            )
            assert parent != qid, f"{qid}: session_parent_id cannot be self"
        seen_ids.add(qid)


def test_follow_up_entries_declare_parent(questions: list[dict[str, Any]]) -> None:
    """Entries with expected_route == follow_up must carry a session_parent_id.

    The seed turn itself uses category=follow_up with session_parent_id: null
    (it IS the seed) but a different expected_route (lookup/compare/etc.), so
    the check keys off expected_route, not category.
    """
    for entry in questions:
        if entry["expected_route"] == "follow_up":
            parent = entry.get("session_parent_id")
            assert parent, (
                f"{entry['id']}: expected_route=follow_up requires session_parent_id"
            )


def test_questions_non_empty(questions: list[dict[str, Any]]) -> None:
    assert len(questions) >= 120, f"expected >=120 entries, got {len(questions)}"
