"""
KAN-ask-cache: Tests for _normalize_question query normalization.

The normalized form is used for both the smart-route/redis cache key and the
semantic-cache embedding lookup, so trivial variants ("What is an LLM?" vs
"what is a large language model") collapse to the same cache entry. The
original question is still passed to Claude verbatim.
"""
from app.routers.intelligence import _normalize_question


def test_whitespace_is_collapsed():
    assert _normalize_question("what   is\tan\n  llm") == "what is an large language model"


def test_leading_and_trailing_whitespace_stripped():
    assert _normalize_question("   hello world   ") == "hello world"


def test_case_insensitivity():
    assert _normalize_question("WHAT IS RAG") == _normalize_question("what is rag")
    assert _normalize_question("WHAT IS RAG") == "what is rag"


def test_trailing_punctuation_stripped():
    assert _normalize_question("hello world!") == "hello world"
    assert _normalize_question("hello world?") == "hello world"
    assert _normalize_question("hello world.") == "hello world"
    assert _normalize_question("hello world;") == "hello world"
    assert _normalize_question("hello world,") == "hello world"
    # Multiple trailing punctuation characters
    assert _normalize_question("hello world?!") == "hello world"


def test_internal_punctuation_preserved():
    # Only TRAILING punctuation is stripped — internal punctuation stays
    assert "," in _normalize_question("hello, world today")


def test_synonym_expansion_llm():
    assert _normalize_question("what is an LLM?") == "what is an large language model"


def test_synonym_expansion_multiple():
    out = _normalize_question("show me ML repos vs AI repos")
    # "ml" -> "machine learning", "repos" -> "repositories", "vs" -> "versus", "ai" -> "artificial intelligence"
    assert "machine learning" in out
    assert "artificial intelligence" in out
    assert "versus" in out
    assert "repositories" in out


def test_synonym_whole_word_only():
    # "repository" should NOT match "repo" synonym (whole-word match only).
    # The real risk: inside an unrelated word like "llama" — must not match "ml".
    out = _normalize_question("llama training")
    assert "llama" in out
    # "llama" should not be corrupted by "ml" → "machine learning" substitution
    assert "machine learning training" not in out


def test_idempotency():
    inputs = [
        "What is an LLM?",
        "   show ME  ml repos vs ai repos!!!",
        "hello world",
        "",
        "Simple question",
        "repo vs ml",
    ]
    for inp in inputs:
        once = _normalize_question(inp)
        twice = _normalize_question(once)
        assert once == twice, f"Not idempotent for {inp!r}: {once!r} -> {twice!r}"


def test_empty_and_whitespace_only():
    assert _normalize_question("") == ""
    assert _normalize_question("   ") == ""


def test_variants_collapse_to_same_key():
    # The primary value prop — trivial variants share a cache entry.
    v1 = _normalize_question("What is an LLM?")
    v2 = _normalize_question("what  is  an  llm")
    v3 = _normalize_question("WHAT IS AN LLM!")
    assert v1 == v2 == v3
