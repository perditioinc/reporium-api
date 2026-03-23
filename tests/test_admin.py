import pytest
from httpx import AsyncClient

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
