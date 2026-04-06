"""
KAN-ask-source-compaction: Tests for compact numbered source format.

The sources block now uses a compact one-line-per-repo format instead of
XML <repo> tags, saving ~165 input tokens per request.

Format:  1. owner/repo (1.2k★, Category): Description text
"""
import re

from app.routers.intelligence import (
    _build_sources_block,
    _format_stars,
    _SOURCES_DESCRIPTION_MAX,
)


def _mock_repo(**overrides):
    base = {
        "name": "langchain",
        "owner": "langchain-ai",
        "primary_category": "LLM Framework",
        "stars": 1234,
        "description": "Build LLM applications with composable primitives.",
    }
    base.update(overrides)
    return base


# ---- _format_stars ----------------------------------------------------------

class TestFormatStars:
    def test_thousands(self):
        assert _format_stars(1234) == "1.2k"

    def test_exact_thousand(self):
        assert _format_stars(1000) == "1.0k"

    def test_ten_thousands(self):
        assert _format_stars(12345) == "12.3k"

    def test_hundred_thousands(self):
        assert _format_stars(95000) == "95.0k"

    def test_below_thousand(self):
        assert _format_stars(500) == "500"

    def test_zero(self):
        assert _format_stars(0) == "0"

    def test_none_returns_empty(self):
        assert _format_stars(None) == ""


# ---- compact format output --------------------------------------------------

class TestCompactFormat:
    def test_single_repo_matches_pattern(self):
        block = _build_sources_block([_mock_repo()])
        # Expected: 1. langchain-ai/langchain (1.2k★, LLM Framework): Build LLM...
        assert block.startswith("1. langchain-ai/langchain")
        assert "1.2k★" in block
        assert "LLM Framework" in block
        assert ": Build LLM applications" in block

    def test_no_xml_tags(self):
        block = _build_sources_block([_mock_repo()])
        assert "<repo" not in block
        assert "</repo>" not in block
        assert "<sources" not in block

    def test_multiple_repos_numbered_sequentially(self):
        repos = [
            _mock_repo(),
            _mock_repo(name="llamaindex", owner="run-llama", stars=500),
        ]
        block = _build_sources_block(repos)
        lines = block.strip().split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("1. ")
        assert lines[1].startswith("2. ")

    def test_owner_slash_name(self):
        block = _build_sources_block([_mock_repo()])
        assert "langchain-ai/langchain" in block

    def test_no_owner_omits_slash(self):
        block = _build_sources_block([_mock_repo(owner="")])
        assert re.search(r"1\. langchain", block)
        assert "/" not in block.split(":")[0]  # no slash in repo name portion

    def test_stars_none_omitted_from_meta(self):
        block = _build_sources_block([_mock_repo(stars=None)])
        assert "★" not in block
        # Category should still appear
        assert "LLM Framework" in block

    def test_category_none_omitted_from_meta(self):
        block = _build_sources_block([_mock_repo(primary_category=None)])
        assert "1.2k★" in block
        # No trailing comma or empty parens
        assert "(1.2k★)" in block

    def test_no_meta_no_parens(self):
        block = _build_sources_block([_mock_repo(stars=None, primary_category=None)])
        assert "(" not in block

    def test_no_description_no_colon_suffix(self):
        block = _build_sources_block([_mock_repo(description=None)])
        # Line should end after the parenthetical, no trailing ": "
        assert not block.rstrip().endswith(":")

    def test_empty_list(self):
        assert _build_sources_block([]) == ""


# ---- description cap --------------------------------------------------------

class TestDescriptionCap:
    def test_description_capped_at_limit(self):
        long_desc = "x" * 500
        block = _build_sources_block([_mock_repo(description=long_desc)])
        assert "x" * _SOURCES_DESCRIPTION_MAX in block
        assert "x" * (_SOURCES_DESCRIPTION_MAX + 1) not in block

    def test_short_description_unchanged(self):
        block = _build_sources_block([_mock_repo(description="Short desc")])
        assert "Short desc" in block
