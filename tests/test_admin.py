import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch

from app.routers.admin import _prune_noise_tags
from tests.conftest import TEST_API_KEY


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
