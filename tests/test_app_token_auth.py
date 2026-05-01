"""
Tests for the X-App-Token contract on LLM endpoints (KAN-API-AUTH-DOC).

Background: the reporium-evals Sprint 0/1 runner spent ~3 days silently
failing because the X-App-Token contract was undocumented and the 403
detail ("App token required") didn't name the missing header. These tests
lock in the diagnostic message so future integrators can self-diagnose
straight from the response body.

Covers:
  * `app.auth.require_app_token` dependency unit-level (missing header,
    wrong token, dev-mode passthrough, correct token).
  * The diagnostic 403 detail names "X-App-Token" verbatim — locking in
    the contract so a future "clean up" rewrite can't silently regress
    the message.
  * End-to-end via /intelligence/nl-filter (the simplest X-App-Token
    endpoint, no DB coupling).
  * /openapi.json registers a distinct `AppToken` security scheme
    referencing the `X-App-Token` header so the contract is visible to
    any spec-driven client.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.auth import require_app_token


# ---------------------------------------------------------------------------
# Unit tests — require_app_token dependency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_require_app_token_dev_passthrough_when_unset(monkeypatch):
    """When APP_API_TOKEN is unset and ENVIRONMENT != production, accept any
    request (dev-safe passthrough). This matches pre-existing behavior so
    the local dev story stays unchanged."""
    monkeypatch.delenv("APP_API_TOKEN", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "test")
    # No header — must be a no-op (no exception).
    assert await require_app_token(x_app_token=None) is None


@pytest.mark.asyncio
async def test_require_app_token_403_when_header_missing(monkeypatch):
    """Missing header (None) returns 403 with a message that NAMES the
    X-App-Token header verbatim. This is the single most useful
    diagnostic for an integrator hitting /ask for the first time."""
    monkeypatch.setenv("APP_API_TOKEN", "secret-token-value")
    with pytest.raises(HTTPException) as exc_info:
        await require_app_token(x_app_token=None)
    assert exc_info.value.status_code == 403
    # Lock in the exact message — this is the contract the README + the
    # OpenAPI scheme description promise to integrators.
    assert exc_info.value.detail == "Missing X-App-Token header"


@pytest.mark.asyncio
async def test_require_app_token_403_when_header_empty_string(monkeypatch):
    """Empty-string header (header sent but no value) is treated the same
    as a missing header — the message must still name the header."""
    monkeypatch.setenv("APP_API_TOKEN", "secret-token-value")
    with pytest.raises(HTTPException) as exc_info:
        await require_app_token(x_app_token="")
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Missing X-App-Token header"


@pytest.mark.asyncio
async def test_require_app_token_403_with_invalid_token(monkeypatch):
    """A non-empty but wrong token returns a different 403 detail so
    integrators can distinguish 'forgot the header' from 'header sent
    but wrong value'."""
    monkeypatch.setenv("APP_API_TOKEN", "secret-token-value")
    with pytest.raises(HTTPException) as exc_info:
        await require_app_token(x_app_token="not-the-right-token")
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Invalid X-App-Token header"


@pytest.mark.asyncio
async def test_require_app_token_passes_with_correct_token(monkeypatch):
    """The correct token is accepted (timing-safe compare under the hood,
    covered separately in test_security_hardening_2026_04.py)."""
    monkeypatch.setenv("APP_API_TOKEN", "secret-token-value")
    # Must not raise and must return None.
    assert await require_app_token(x_app_token="secret-token-value") is None


# ---------------------------------------------------------------------------
# Integration tests — /intelligence/nl-filter end-to-end
#
# nl-filter is the cleanest X-App-Token route: no DB coupling, no
# Anthropic call when the body fails validation. We use a query that
# passes validation (3-300 chars) and assert on the auth layer alone by
# patching the LLM call so we don't actually hit Anthropic in CI.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nl_filter_403_without_header(client, monkeypatch):
    """End-to-end: a request without X-App-Token gets a 403 with the
    diagnostic detail. This is the response shape an integrator sees on
    their first failed call, and the message must name the header."""
    monkeypatch.setenv("APP_API_TOKEN", "secret-token-value")
    resp = await client.post(
        "/intelligence/nl-filter",
        json={"query": "Python RAG repos with 1000+ stars"},
    )
    assert resp.status_code == 403
    body = resp.json()
    assert body == {"detail": "Missing X-App-Token header"}


@pytest.mark.asyncio
async def test_nl_filter_403_with_wrong_header(client, monkeypatch):
    """End-to-end: a wrong token returns the 'Invalid' variant of the
    detail so callers can distinguish missing vs. wrong."""
    monkeypatch.setenv("APP_API_TOKEN", "secret-token-value")
    resp = await client.post(
        "/intelligence/nl-filter",
        json={"query": "Python RAG repos with 1000+ stars"},
        headers={"X-App-Token": "wrong-token"},
    )
    assert resp.status_code == 403
    body = resp.json()
    assert body == {"detail": "Invalid X-App-Token header"}


@pytest.mark.asyncio
async def test_nl_filter_passes_auth_with_correct_header(client, monkeypatch):
    """End-to-end: a request with the correct X-App-Token clears the auth
    layer. We don't assert on the body (the route does its own LLM call)
    — only that we didn't get the 403 the other tests assert on. A 200
    or any other downstream status proves auth passed."""
    import json
    from unittest.mock import MagicMock, patch

    monkeypatch.setenv("APP_API_TOKEN", "secret-token-value")

    # Mock Haiku so we don't hit Anthropic in CI; auth runs before this.
    parsed = {
        "language": "python",
        "category": "rag-retrieval",
        "min_stars": 1000,
        "max_stars": None,
        "sort": "stars",
        "tags": ["rag"],
        "quality": None,
        "maturity": None,
        "exclude_archived": False,
        "interpretation": "Python · RAG · 1,000+ stars",
    }
    fake_msg = MagicMock()
    fake_msg.content = [MagicMock(text=json.dumps(parsed))]
    fake_msg.usage = MagicMock(input_tokens=100, output_tokens=50)
    breaker_mock = MagicMock(call=lambda fn: fn())

    with (
        patch("app.routers.nl_filter._get_client") as MockClient,
        patch("app.routers.nl_filter.anthropic_breaker", breaker_mock),
        patch("app.routers.nl_filter.cache.get", return_value=None),
        patch("app.routers.nl_filter.cache.set"),
    ):
        MockClient.return_value.messages.create.return_value = fake_msg
        resp = await client.post(
            "/intelligence/nl-filter",
            json={"query": "Python RAG repos with 1000+ stars"},
            headers={"X-App-Token": "secret-token-value"},
        )

    # The contract this test enforces: with the correct header, we get
    # past auth — i.e. NOT a 403. The downstream behavior is covered in
    # test_nl_filter.py.
    assert resp.status_code != 403, (
        f"Expected auth to pass with correct X-App-Token; got 403: {resp.text}"
    )


# ---------------------------------------------------------------------------
# OpenAPI contract — /openapi.json self-documents the X-App-Token header
#
# This is the surface a spec-driven integrator (or codegen tool) reads
# first. If FastAPI ever stops generating a distinct security scheme for
# X-App-Token, this test fails loudly so the regression doesn't ship.
# ---------------------------------------------------------------------------


def test_openapi_registers_app_token_scheme():
    """OpenAPI introspection — does NOT use the `client` fixture so this
    runs without a test database. Reads `app.openapi()` directly."""
    from app.main import app as _app

    spec = _app.openapi()
    schemes = spec.get("components", {}).get("securitySchemes", {})

    # AppToken scheme is registered with a distinct name (not the FastAPI
    # default class-name "APIKeyHeader") AND points at the X-App-Token
    # header explicitly. Without a distinct scheme_name, AppToken would
    # collapse into AdminKey/IngestKey under a single 'APIKeyHeader' key
    # — exactly the OpenAPI ambiguity the reporium-evals runner hit.
    assert "AppToken" in schemes, (
        f"AppToken security scheme missing; only got: {list(schemes.keys())}"
    )
    app_token_scheme = schemes["AppToken"]
    assert app_token_scheme["type"] == "apiKey"
    assert app_token_scheme["in"] == "header"
    assert app_token_scheme["name"] == "X-App-Token"
    # The description must mention the header name so anyone reading the
    # OpenAPI spec (Scalar UI, codegen, swagger viewer) sees it.
    assert "X-App-Token" in app_token_scheme.get("description", "")


def test_openapi_protected_routes_reference_app_token():
    """The four LLM endpoints documented in the README all reference the
    AppToken scheme in their `security` block. Asserting this catches a
    decorator-removal regression (e.g. someone deleting Depends(require_app_token))."""
    from app.main import app as _app

    spec = _app.openapi()
    paths = spec["paths"]

    expected = {
        "/intelligence/ask": "post",
        "/intelligence/ask/stream": "post",
        "/intelligence/nl-filter": "post",
        "/intelligence/compare": "get",
    }
    for path, method in expected.items():
        op = paths.get(path, {}).get(method)
        assert op is not None, f"Route {method.upper()} {path} missing from OpenAPI"
        security = op.get("security", [])
        scheme_names = [list(s.keys())[0] for s in security if s]
        assert "AppToken" in scheme_names, (
            f"Route {method.upper()} {path} does not require AppToken; "
            f"security block: {security}"
        )


def test_openapi_admin_routes_use_distinct_scheme_from_app_token():
    """Regression guard against the ambiguity that motivated this ticket:
    /admin/* and /ingest/* must not be tagged AppToken (and vice versa).
    Distinct scheme_names per APIKeyHeader keep the spec self-documenting."""
    from app.main import app as _app

    spec = _app.openapi()
    paths = spec["paths"]

    # Sample admin/ingest routes — these should require AdminKey or
    # IngestKey, NOT AppToken.
    cases = [
        ("/admin/data-quality", "get", "AdminKey"),
        ("/ingest/repos", "post", "IngestKey"),
    ]
    for path, method, expected_scheme in cases:
        op = paths.get(path, {}).get(method)
        if op is None:
            continue  # tolerate route renames; the AppToken assertions above are the real lock
        security = op.get("security", [])
        scheme_names = [list(s.keys())[0] for s in security if s]
        assert expected_scheme in scheme_names, (
            f"Route {method.upper()} {path} expected {expected_scheme}; "
            f"got: {scheme_names}"
        )
        assert "AppToken" not in scheme_names, (
            f"Route {method.upper()} {path} unexpectedly tagged AppToken; "
            f"security block: {security}"
        )
