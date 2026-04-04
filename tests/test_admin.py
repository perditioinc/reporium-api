import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch

from app.routers.admin import _prune_noise_tags
from tests.conftest import AUTH_HEADERS, TEST_API_KEY


@pytest.mark.asyncio
async def test_data_quality_requires_api_key(client: AsyncClient):
    response = await client.get("/admin/data-quality")
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_data_quality_returns_correct_shape(client: AsyncClient):
    response = await client.get(
        "/admin/data-quality",
        headers={"Authorization": f"Bearer {TEST_API_KEY}"},
    )
    assert response.status_code == 200
    data = response.json()
    expected_keys = {
        "total_repos",
        "owned_repos",
        "fork_repos",
        "missing_summary",
        "missing_description",
        "missing_categories",
        "missing_builders",
        "missing_embeddings",
        "category_distribution",
        "max_category_percent",
        "quality_score",
    }
    assert expected_keys == set(data.keys())
    assert isinstance(data["total_repos"], int)
    assert isinstance(data["quality_score"], int)
    assert isinstance(data["category_distribution"], dict)
    assert 0 <= data["quality_score"] <= 100


class _ScalarResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


# ---------------------------------------------------------------------------
# /admin/health/data
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_data_integrity_health_requires_api_key(client: AsyncClient):
    """Endpoint should reject requests without a valid API key."""
    response = await client.get("/admin/health/data")
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_data_integrity_health_returns_correct_shape(client: AsyncClient):
    response = await client.get("/admin/health/data", headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()

    assert set(data.keys()) == {"status", "counts", "coverage", "thresholds", "alerts"}

    # status must be one of the three sentinel strings
    assert data["status"] in {"healthy", "degraded", "critical"}

    # counts must include every monitored table
    expected_tables = {
        "repos",
        "repo_tags",
        "repo_categories",
        "repo_taxonomy",
        "taxonomy_values",
        "repo_ai_dev_skills",
        "repo_pm_skills",
        "repo_languages",
    }
    assert expected_tables == set(data["counts"].keys())
    for v in data["counts"].values():
        assert isinstance(v, int)

    # coverage must contain the three ratio keys
    assert set(data["coverage"].keys()) == {"tags_pct", "categories_pct", "languages_pct"}
    for v in data["coverage"].values():
        assert isinstance(v, float)
        assert 0.0 <= v <= 100.0

    # thresholds present
    assert "repo_tags_min_rows" in data["thresholds"]
    assert "tags_coverage_min_pct" in data["thresholds"]

    # alerts is a list
    assert isinstance(data["alerts"], list)


@pytest.mark.asyncio
async def test_data_integrity_health_status_critical_when_no_tags(client: AsyncClient):
    """With an empty database the tag count is 0 → status must be critical."""
    response = await client.get("/admin/health/data", headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    # Empty DB → repo_tags = 0 which is < 100 → critical
    if data["counts"]["repo_tags"] < 100:
        assert data["status"] == "critical"
        assert len(data["alerts"]) >= 1


@pytest.mark.asyncio
async def test_data_integrity_health_healthy_status_logic():
    """Unit-test the status logic directly by mocking the DB."""
    from app.routers.admin import data_integrity_health
    from unittest.mock import MagicMock

    # Build a mock DB that returns high counts and good coverage
    call_count = 0
    table_counts = {
        "repos": 500,
        "repo_tags": 3000,
        "repo_categories": 450,
        "repo_taxonomy": 2000,
        "taxonomy_values": 800,
        "repo_ai_dev_skills": 300,
        "repo_pm_skills": 200,
        "repo_languages": 480,
        # DISTINCT coverage queries
        "tags_distinct": 490,
        "cats_distinct": 460,
        "langs_distinct": 475,
    }

    scalar_values = [
        # 8 table COUNTs (order matches the `tables` list in the handler)
        table_counts["repos"],
        table_counts["repo_tags"],
        table_counts["repo_categories"],
        table_counts["repo_taxonomy"],
        table_counts["taxonomy_values"],
        table_counts["repo_ai_dev_skills"],
        table_counts["repo_pm_skills"],
        table_counts["repo_languages"],
        # 3 DISTINCT coverage queries
        table_counts["tags_distinct"],
        table_counts["cats_distinct"],
        table_counts["langs_distinct"],
    ]

    idx = 0

    async def fake_execute(stmt):
        nonlocal idx
        mock_result = MagicMock()
        mock_result.scalar.return_value = scalar_values[idx]
        idx += 1
        return mock_result

    db = AsyncMock()
    db.execute = fake_execute

    result = await data_integrity_health(db=db, _admin_key=None)

    assert result["status"] == "healthy"
    assert result["counts"]["repos"] == 500
    assert result["counts"]["repo_tags"] == 3000
    assert result["coverage"]["tags_pct"] == round(490 / 500 * 100, 1)
    assert result["alerts"] == []


@pytest.mark.asyncio
async def test_data_integrity_health_degraded_status_logic():
    """Status is degraded when tag count >= 100 but coverage < 50 %."""
    from app.routers.admin import data_integrity_health
    from unittest.mock import MagicMock

    # 1000 repos, 150 tags total (>= 100), but only 40 % covered
    scalar_values = [
        1000,  # repos
        150,   # repo_tags
        800,   # repo_categories
        500,   # repo_taxonomy
        200,   # taxonomy_values
        100,   # repo_ai_dev_skills
        80,    # repo_pm_skills
        950,   # repo_languages
        400,   # DISTINCT tags (40 %)
        700,   # DISTINCT categories
        900,   # DISTINCT languages
    ]

    idx = 0

    async def fake_execute(stmt):
        nonlocal idx
        mock_result = MagicMock()
        mock_result.scalar.return_value = scalar_values[idx]
        idx += 1
        return mock_result

    db = AsyncMock()
    db.execute = fake_execute

    result = await data_integrity_health(db=db, _admin_key=None)

    assert result["status"] == "degraded"
    assert len(result["alerts"]) == 1
    assert "degraded" in result["alerts"][0]


# ---------------------------------------------------------------------------
# _prune_noise_tags (existing helpers — kept below)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prune_noise_tags_dry_run_returns_counts_without_commit():
    db = AsyncMock()
    db.execute = AsyncMock(return_value=_ScalarResult([
        type("Row", (), {"tag": "python", "count": 4})(),
        type("Row", (), {"tag": "docker", "count": 2})(),
    ]))

    result = await _prune_noise_tags(db, dry_run=True)

    assert result["dry_run"] is True
    assert result["matched_rows"] == 6
    assert result["matched_tag_count"] == 2
    assert result["deleted_rows"] == 0
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_prune_noise_tags_deletes_and_invalidates_cache():
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[
        _ScalarResult([type("Row", (), {"tag": "python", "count": 3})()]),
        type("DeleteResult", (), {"rowcount": 3})(),
    ])

    with patch("app.routers.admin.cache.invalidate", new=AsyncMock()) as invalidate, \
         patch("app.routers.admin.invalidate_library_cache") as invalidate_memory:
        result = await _prune_noise_tags(db, dry_run=False)

    assert result["dry_run"] is False
    assert result["matched_rows"] == 3
    assert result["deleted_rows"] == 3
    db.commit.assert_awaited_once()
    assert invalidate.await_count == 2
    invalidate_memory.assert_called_once()
