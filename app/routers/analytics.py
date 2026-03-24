from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.analytics import CrossDimensionCell, CrossDimensionResponse

router = APIRouter(tags=["Analytics"])


@router.get("/analytics/cross-dimension", response_model=CrossDimensionResponse)
async def cross_dimension_analytics(
    dim1: str = Query(..., min_length=1),
    dim2: str = Query(..., min_length=1),
    limit: int = Query(default=10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
) -> CrossDimensionResponse:
    if dim1 == dim2:
        raise HTTPException(status_code=400, detail="dim1 and dim2 must be different")

    result = await db.execute(
        text(
            """
            WITH dim1_values AS (
                SELECT DISTINCT rt.repo_id,
                       COALESCE(tv.name, rt.raw_value) AS dim1_value
                FROM repo_taxonomy rt
                LEFT JOIN taxonomy_values tv ON tv.id = rt.taxonomy_value_id
                WHERE rt.dimension = :dim1
            ),
            dim2_values AS (
                SELECT DISTINCT rt.repo_id,
                       COALESCE(tv.name, rt.raw_value) AS dim2_value
                FROM repo_taxonomy rt
                LEFT JOIN taxonomy_values tv ON tv.id = rt.taxonomy_value_id
                WHERE rt.dimension = :dim2
            )
            SELECT d1.dim1_value,
                   d2.dim2_value,
                   COUNT(DISTINCT d1.repo_id) AS repo_count
            FROM dim1_values d1
            JOIN dim2_values d2 ON d2.repo_id = d1.repo_id
            GROUP BY d1.dim1_value, d2.dim2_value
            ORDER BY repo_count DESC, d1.dim1_value ASC, d2.dim2_value ASC
            LIMIT :limit
            """
        ),
        {"dim1": dim1, "dim2": dim2, "limit": limit},
    )
    rows = result.fetchall()

    return CrossDimensionResponse(
        dim1=dim1,
        dim2=dim2,
        limit=limit,
        pairs=[
            CrossDimensionCell(
                dim1_value=row.dim1_value,
                dim2_value=row.dim2_value,
                repo_count=row.repo_count,
            )
            for row in rows
        ],
    )
