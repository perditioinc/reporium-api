"""
KAN-83: Knowledge graph edges endpoint.
Returns repo-to-repo relationship edges from the repo_edges table.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db

router = APIRouter(tags=["Graph"])


@router.get("/graph/edges")
async def get_graph_edges(
    limit: int = Query(default=500, ge=1, le=2000),
    edge_type: str | None = Query(default=None, description="Filter by edge type (e.g. ALTERNATIVE_TO, DEPENDS_ON)"),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns knowledge graph edges with source/target repo names.
    Only returns edges between public repos.
    """
    base_sql = """
        SELECT
            e.edge_type,
            e.weight,
            e.evidence,
            r1.name AS source_name,
            r1.description AS source_description,
            r1.primary_category AS source_category,
            r2.name AS target_name,
            r2.description AS target_description,
            r2.primary_category AS target_category
        FROM repo_edges e
        JOIN repos r1 ON r1.id = e.source_repo_id
        JOIN repos r2 ON r2.id = e.target_repo_id
        WHERE r1.is_private = false
          AND r2.is_private = false
    """
    params: dict = {"limit": limit}

    if edge_type:
        base_sql += " AND e.edge_type = :edge_type"
        params["edge_type"] = edge_type

    base_sql += " ORDER BY e.weight DESC NULLS LAST LIMIT :limit"

    result = await db.execute(text(base_sql), params)
    rows = result.fetchall()

    edges = [
        {
            "edgeType": row.edge_type,
            "weight": float(row.weight) if row.weight is not None else None,
            "evidence": row.evidence,
            "source": {
                "name": row.source_name,
                "description": row.source_description,
                "category": row.source_category,
            },
            "target": {
                "name": row.target_name,
                "description": row.target_description,
                "category": row.target_category,
            },
        }
        for row in rows
    ]

    # Collect distinct edge types for the response
    type_result = await db.execute(
        text(
            "SELECT DISTINCT e.edge_type FROM repo_edges e "
            "JOIN repos r1 ON r1.id = e.source_repo_id "
            "JOIN repos r2 ON r2.id = e.target_repo_id "
            "WHERE r1.is_private = false AND r2.is_private = false "
            "ORDER BY e.edge_type"
        )
    )
    edge_types = [r[0] for r in type_result.fetchall()]

    return {
        "total": len(edges),
        "edgeTypes": edge_types,
        "edges": edges,
    }
