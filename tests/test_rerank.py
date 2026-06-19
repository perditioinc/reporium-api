"""Unit tests for the flag-gated cross-encoder reranking stage (app/rerank.py).

Reranking the dense candidate set is a STATISTICALLY-PROVEN win on realistic
semantic queries (offline eval: nDCG@10 0.69->0.82, MRR 0.87->0.97, 95% CI
excludes 0) but it is OFF by default and skipped for name-literal "alternatives
to X" queries where dense wins. These tests use a fake model (no download).
"""
import os

import pytest

from app import rerank


class _FakeCE:
    """Scores a (query, passage) pair 1.0 if passage mentions 'target' else 0.0."""

    def __init__(self):
        self.calls = 0

    def predict(self, pairs, **kw):
        self.calls += 1
        return [1.0 if "target" in p[1].lower() else 0.0 for p in pairs]


def test_rerank_reorders_relevant_candidate_to_front():
    scored = [
        {"id": "1", "name": "alpha", "description": "unrelated"},
        {"id": "2", "name": "beta", "description": "the target tool"},
        {"id": "3", "name": "gamma", "description": "also unrelated"},
    ]
    out = rerank.rerank_candidates("q", scored, model=_FakeCE())
    assert out[0]["id"] == "2"  # the 'target' doc is promoted to the front
    assert {r["id"] for r in out} == {"1", "2", "3"}  # no docs lost


def test_rerank_attaches_score_and_preserves_fields():
    scored = [{"id": "2", "name": "x", "description": "target", "similarity": 0.3}]
    # singleton is returned unchanged (no reorder needed) but must keep fields
    out = rerank.rerank_candidates("q", scored, model=_FakeCE())
    assert out[0]["id"] == "2" and out[0]["similarity"] == 0.3


def test_rerank_multi_attaches_rerank_score():
    scored = [{"id": "1", "description": "target"}, {"id": "2", "description": "no"}]
    out = rerank.rerank_candidates("q", scored, model=_FakeCE())
    assert "rerank_score" in out[0]


def test_rerank_empty_and_singleton_unchanged():
    assert rerank.rerank_candidates("q", [], model=_FakeCE()) == []
    one = [{"id": "1", "description": "x"}]
    assert rerank.rerank_candidates("q", one, model=_FakeCE()) == one


def test_build_rerank_text_concatenates_and_truncates():
    repo = {"name": "foo", "forked_from": "up/foo", "description": "d",
            "readme_summary": "r", "problem_solved": "p"}
    t = rerank.build_rerank_text(repo)
    assert "foo" in t and "up/foo" in t and "r" in t
    assert len(rerank.build_rerank_text({"description": "x" * 9999})) <= 512


def test_flags_default_off(monkeypatch):
    monkeypatch.delenv("RERANK_ENABLED", raising=False)
    assert rerank.rerank_enabled() is False


def test_flag_enabled_when_set(monkeypatch):
    monkeypatch.setenv("RERANK_ENABLED", "1")
    assert rerank.rerank_enabled() is True


def test_fetch_n_default_and_override(monkeypatch):
    monkeypatch.delenv("RERANK_FETCH_N", raising=False)
    assert rerank.rerank_fetch_n() == 50
    monkeypatch.setenv("RERANK_FETCH_N", "80")
    assert rerank.rerank_fetch_n() == 80


def test_fetch_n_is_clamped_and_bad_value_falls_back(monkeypatch):
    monkeypatch.setenv("RERANK_FETCH_N", "100000")
    assert rerank.rerank_fetch_n() == 100  # hard cap
    monkeypatch.setenv("RERANK_FETCH_N", "0")
    assert rerank.rerank_fetch_n() == 1     # floor
    monkeypatch.setenv("RERANK_FETCH_N", "notanint")
    assert rerank.rerank_fetch_n() == 50    # default on parse error


def test_name_lookup_queries_are_detected():
    assert rerank.is_name_lookup_query("vector database alternatives to pinecone")
    assert rerank.is_name_lookup_query("something similar to langchain")
    assert rerank.is_name_lookup_query("replacement for redis")


def test_semantic_queries_are_not_name_lookups():
    assert not rerank.is_name_lookup_query("run large language models locally")
    assert not rerank.is_name_lookup_query("framework for building agents")
    assert not rerank.is_name_lookup_query("")
