"""
PR4 (Ask UX): tests for the follow-up suggestion generator.

The generator is launched in parallel with the main answer stream and its
output is included in the `done` SSE event as `suggestions: [str]` when ready.
Frontend renders these as clickable chips below the answer.

Two layers under test here:
  1. ``_parse_followups`` — pure-function JSON extraction with bounds & cleanup.
  2. ``_generate_followups`` — async wrapper that calls Haiku, with the
     anthropic client mocked.
"""
from unittest.mock import MagicMock, patch

import pytest

from app.routers.intelligence import (
    _FOLLOWUPS_COUNT,
    _FOLLOWUPS_MAX_LEN,
    _generate_followups,
    _parse_followups,
)


# ---------------------------------------------------------------------------
# _parse_followups
# ---------------------------------------------------------------------------


def test_parse_clean_json_array():
    raw = '["What about LangSmith?", "How does it compare to Haystack?", "Is RAG built in?"]'
    out = _parse_followups(raw)
    assert out == [
        "What about LangSmith?",
        "How does it compare to Haystack?",
        "Is RAG built in?",
    ]


def test_parse_strips_surrounding_prose():
    raw = (
        "Here are 3 follow-ups:\n"
        '["A?", "B?", "C?"]\n'
        "Hope these help!"
    )
    assert _parse_followups(raw) == ["A?", "B?", "C?"]


def test_parse_caps_at_three_items():
    raw = '["a?", "b?", "c?", "d?", "e?"]'
    out = _parse_followups(raw)
    assert len(out) == _FOLLOWUPS_COUNT == 3
    assert out == ["a?", "b?", "c?"]


def test_parse_trims_overlong_strings_with_ellipsis():
    too_long = "x" * (_FOLLOWUPS_MAX_LEN + 50)
    raw = f'["{too_long}"]'
    out = _parse_followups(raw)
    assert len(out) == 1
    assert len(out[0]) == _FOLLOWUPS_MAX_LEN
    assert out[0].endswith("…")


def test_parse_drops_non_string_items():
    raw = '["A?", 42, null, {"q": "B?"}, "C?"]'
    assert _parse_followups(raw) == ["A?", "C?"]


def test_parse_drops_blank_items():
    raw = '["A?", "   ", "", "C?"]'
    assert _parse_followups(raw) == ["A?", "C?"]


def test_parse_returns_empty_on_invalid_json():
    assert _parse_followups("not json at all") == []
    assert _parse_followups("[unterminated") == []
    assert _parse_followups("") == []
    assert _parse_followups(None) == []  # type: ignore[arg-type]


def test_parse_returns_empty_when_top_level_is_not_array():
    assert _parse_followups('{"q": "A?"}') == []


# ---------------------------------------------------------------------------
# _generate_followups
# ---------------------------------------------------------------------------


def _mock_anthropic_response(text: str) -> MagicMock:
    """Build a fake Anthropic SDK response carrying a single text block."""
    block = MagicMock()
    block.text = text
    msg = MagicMock()
    msg.content = [block]
    return msg


def _sources(n: int = 2) -> list[dict]:
    return [
        {
            "name": f"repo{i}",
            "owner": "perditioinc",
            "forked_from": f"upstream{i}/repo{i}",
            "primary_category": "LLM Framework",
        }
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_generate_returns_empty_for_empty_inputs():
    assert await _generate_followups("", _sources()) == []
    assert await _generate_followups("hi", []) == []


@pytest.mark.asyncio
async def test_generate_uses_canonical_fork_names_in_prompt():
    """The compact source view fed to Haiku must use the upstream owner for
    forks (matches _build_sources_block hotfix). We capture the prompt and
    assert it contains the upstream — and never the perditioinc mirror — for
    forked repos."""
    captured: dict[str, str] = {}

    def _fake_create(**kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        return _mock_anthropic_response('["A?", "B?", "C?"]')

    fake_client = MagicMock()
    fake_client.messages.create.side_effect = _fake_create

    with patch("app.routers.intelligence._get_client", return_value=fake_client):
        out = await _generate_followups("how do I do RAG?", _sources(2))

    assert out == ["A?", "B?", "C?"]
    prompt = captured["prompt"]
    assert "upstream0/repo0" in prompt
    assert "upstream1/repo1" in prompt
    assert "perditioinc" not in prompt


@pytest.mark.asyncio
async def test_generate_returns_empty_when_client_raises():
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = RuntimeError("anthropic 500")

    with patch("app.routers.intelligence._get_client", return_value=fake_client):
        out = await _generate_followups("q?", _sources())

    assert out == []


@pytest.mark.asyncio
async def test_generate_returns_empty_when_response_unparseable():
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _mock_anthropic_response(
        "I cannot suggest follow-ups."
    )

    with patch("app.routers.intelligence._get_client", return_value=fake_client):
        out = await _generate_followups("q?", _sources())

    assert out == []
