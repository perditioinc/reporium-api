"""Tests for dependency graph infrastructure: purl parsing, SBOM extraction,
backfill auth, and dependents lookup."""

import uuid

import pytest
from httpx import AsyncClient

from app.routers.admin import parse_purl, _extract_deps_from_sbom
from tests.conftest import TEST_API_KEY


# ── purl parsing ─────────────────────────────────────────────────────────────

class TestParsePurl:
    def test_simple_pypi(self):
        eco, name, ver = parse_purl("pkg:pypi/langchain@0.1.0")
        assert eco == "pypi"
        assert name == "langchain"
        assert ver == "0.1.0"

    def test_npm_scoped(self):
        eco, name, ver = parse_purl("pkg:npm/%40openai/api@4.0.0")
        assert eco == "npm"
        assert name == "api"
        assert ver == "4.0.0"

    def test_no_version(self):
        eco, name, ver = parse_purl("pkg:pypi/requests")
        assert eco == "pypi"
        assert name == "requests"
        assert ver is None

    def test_with_namespace(self):
        eco, name, ver = parse_purl("pkg:maven/org.apache.commons/commons-lang3@3.12.0")
        assert eco == "maven"
        assert name == "commons-lang3"
        assert ver == "3.12.0"

    def test_empty_string(self):
        eco, name, ver = parse_purl("")
        assert eco is None
        assert name is None
        assert ver is None

    def test_none(self):
        eco, name, ver = parse_purl(None)
        assert eco is None
        assert name is None
        assert ver is None

    def test_invalid_prefix(self):
        eco, name, ver = parse_purl("notpkg:pypi/foo@1.0")
        assert eco is None
        assert name is None

    def test_no_slash(self):
        eco, name, ver = parse_purl("pkg:pypi")
        assert eco is None
        assert name is None

    def test_cargo(self):
        eco, name, ver = parse_purl("pkg:cargo/serde@1.0.193")
        assert eco == "cargo"
        assert name == "serde"
        assert ver == "1.0.193"

    def test_golang(self):
        # golang purls have deeper namespace paths; split(/, 2) keeps the tail
        eco, name, ver = parse_purl("pkg:golang/github.com/gin-gonic/gin@1.9.1")
        assert eco == "golang"
        assert name == "gin-gonic/gin"
        assert ver == "1.9.1"

    def test_version_with_build_metadata(self):
        eco, name, ver = parse_purl("pkg:pypi/torch@2.1.0+cu121")
        assert eco == "pypi"
        assert name == "torch"
        assert ver == "2.1.0+cu121"


# ── SBOM extraction ──────────────────────────────────────────────────────────

class TestExtractDepsFromSbom:
    def test_basic_sbom(self):
        sbom = {
            "sbom": {
                "packages": [
                    {
                        "SPDXID": "SPDXRef-DOCUMENT",
                        "name": "root-project",
                    },
                    {
                        "SPDXID": "SPDXRef-pip-langchain",
                        "name": "langchain",
                        "externalRefs": [
                            {
                                "referenceType": "purl",
                                "referenceLocator": "pkg:pypi/langchain@0.1.0",
                            }
                        ],
                    },
                    {
                        "SPDXID": "SPDXRef-pip-requests",
                        "name": "requests",
                        "externalRefs": [
                            {
                                "referenceType": "purl",
                                "referenceLocator": "pkg:pypi/requests@2.31.0",
                            }
                        ],
                    },
                ]
            }
        }
        deps = _extract_deps_from_sbom(sbom)
        assert len(deps) == 2
        assert deps[0]["package_name"] == "langchain"
        assert deps[0]["package_ecosystem"] == "pypi"
        assert deps[0]["version_constraint"] == "0.1.0"
        assert deps[1]["package_name"] == "requests"

    def test_skips_document_root(self):
        sbom = {
            "sbom": {
                "packages": [
                    {"SPDXID": "SPDXRef-DOCUMENT", "name": "my-project"},
                ]
            }
        }
        deps = _extract_deps_from_sbom(sbom)
        assert len(deps) == 0

    def test_fallback_to_name_field(self):
        sbom = {
            "sbom": {
                "packages": [
                    {
                        "SPDXID": "SPDXRef-unknown",
                        "name": "some-lib",
                        "versionInfo": "1.2.3",
                    },
                ]
            }
        }
        deps = _extract_deps_from_sbom(sbom)
        assert len(deps) == 1
        assert deps[0]["package_name"] == "some-lib"
        assert deps[0]["package_ecosystem"] is None
        assert deps[0]["version_constraint"] == "1.2.3"

    def test_empty_sbom(self):
        deps = _extract_deps_from_sbom({})
        assert len(deps) == 0

    def test_no_packages_key(self):
        deps = _extract_deps_from_sbom({"sbom": {}})
        assert len(deps) == 0

    def test_mixed_purl_and_fallback(self):
        sbom = {
            "sbom": {
                "packages": [
                    {
                        "SPDXID": "SPDXRef-1",
                        "name": "with-purl",
                        "externalRefs": [
                            {"referenceType": "purl", "referenceLocator": "pkg:npm/react@18.0.0"}
                        ],
                    },
                    {
                        "SPDXID": "SPDXRef-2",
                        "name": "no-purl-lib",
                        "versionInfo": "0.5",
                    },
                ]
            }
        }
        deps = _extract_deps_from_sbom(sbom)
        assert len(deps) == 2
        assert deps[0]["package_ecosystem"] == "npm"
        assert deps[1]["package_ecosystem"] is None

    def test_non_purl_external_refs_ignored(self):
        sbom = {
            "sbom": {
                "packages": [
                    {
                        "SPDXID": "SPDXRef-1",
                        "name": "mylib",
                        "externalRefs": [
                            {"referenceType": "cpe23Type", "referenceLocator": "cpe:2.3:*:*:*"}
                        ],
                    },
                ]
            }
        }
        deps = _extract_deps_from_sbom(sbom)
        assert len(deps) == 1
        assert deps[0]["package_name"] == "mylib"
        assert deps[0]["package_ecosystem"] is None


# ── Backfill endpoint auth ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_backfill_dependencies_requires_api_key(client: AsyncClient):
    response = await client.post("/admin/backfill-dependencies")
    assert response.status_code in (401, 403)


@pytest.mark.asyncio
async def test_backfill_dependencies_requires_admin_key(client: AsyncClient):
    response = await client.post(
        "/admin/backfill-dependencies",
        headers={"Authorization": f"Bearer {TEST_API_KEY}"},
    )
    # Without admin key, should be rejected (admin key check uses a separate header)
    assert response.status_code in (401, 403, 500)


# ── GET /repos/{repo_id}/dependencies ────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_dependencies_404_for_unknown_repo(client: AsyncClient):
    fake_id = str(uuid.uuid4())
    response = await client.get(f"/repos/{fake_id}/dependencies")
    assert response.status_code == 404


# ── GET /dependencies/dependents ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dependents_requires_package_param(client: AsyncClient):
    response = await client.get("/dependencies/dependents")
    assert response.status_code == 422  # missing required query param


@pytest.mark.asyncio
async def test_dependents_returns_empty_for_unknown_package(client: AsyncClient):
    response = await client.get("/dependencies/dependents", params={"package": "nonexistent-pkg-xyz"})
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_dependents_with_ecosystem_filter(client: AsyncClient):
    response = await client.get(
        "/dependencies/dependents",
        params={"package": "nonexistent-pkg-xyz", "ecosystem": "pypi"},
    )
    assert response.status_code == 200
    assert response.json() == []
