"""
GET /taxonomy/skill-areas — Returns all skill areas from DB, grouped by lifecycle_group.
GET /taxonomy/skill-areas/{name}/repos — Returns repos tagged with a given skill area.
GET /taxonomy/dimensions — Returns distinct dimension strings with repo counts.
GET /taxonomy/{dimension} — Returns taxonomy values for a dimension, sorted by repo_count desc.
POST /admin/taxonomy/rebuild — Aggregates raw_values from repo_taxonomy into taxonomy_values.
POST /admin/taxonomy/embed — Generates embeddings for taxonomy_values missing embedding_vec.
POST /admin/taxonomy/assign — Similarity-assigns taxonomy_values to repos via pgvector.

Skill areas are stored in the skill_areas table; adding a new area requires only a DB insert.
"""

import logging
import time
from collections import defaultdict
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_admin_key, verify_api_key
from app.database import get_db
from app.models.repo import SkillArea, TaxonomyValue

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Taxonomy"])

# In-memory cache for skill-areas list (5 min TTL)
_taxonomy_cache: dict = {}
_TAXONOMY_CACHE_TTL = 300  # 5 minutes


@router.get("/skill-areas")
async def list_skill_areas(db: AsyncSession = Depends(get_db)) -> dict:
    """
    Return all skill areas from the DB, grouped by lifecycle_group.
    Each skill area includes a repoCount (repos in repo_ai_dev_skills).
    Skill areas with repoCount < min_repos_to_display are excluded.
    """
    now = time.monotonic()
    cached = _taxonomy_cache.get("skill_areas")
    if cached and cached.get("expires_at", 0) > now:
        return cached["data"]

    # Fetch all skill areas ordered by sort_order
    stmt = select(SkillArea).order_by(SkillArea.sort_order, SkillArea.name)
    result = await db.execute(stmt)
    skill_areas = result.scalars().all()

    # Count repos per skill area from repo_ai_dev_skills
    count_stmt = text(
        "SELECT skill, COUNT(DISTINCT repo_id) AS repo_count "
        "FROM repo_ai_dev_skills "
        "GROUP BY skill"
    )
    count_result = await db.execute(count_stmt)
    repo_counts: dict[str, int] = {row.skill: row.repo_count for row in count_result}

    grouped: dict[str, list] = defaultdict(list)
    for sa in skill_areas:
        count = repo_counts.get(sa.name, 0)
        if count < sa.min_repos_to_display:
            continue
        grouped[sa.lifecycle_group].append({
            "id": sa.id,
            "name": sa.name,
            "lifecycleGroup": sa.lifecycle_group,
            "description": sa.description,
            "icon": sa.icon,
            "color": sa.color,
            "sortOrder": sa.sort_order,
            "repoCount": count,
        })

    response = {
        "groups": [
            {
                "lifecycleGroup": group,
                "skillAreas": areas,
            }
            for group, areas in grouped.items()
        ],
        "total": sum(len(areas) for areas in grouped.values()),
    }

    _taxonomy_cache["skill_areas"] = {"data": response, "expires_at": now + _TAXONOMY_CACHE_TTL}
    return response


@router.get("/skill-areas/{name}/repos")
async def get_repos_for_skill_area(name: str, db: AsyncSession = Depends(get_db)) -> dict:
    """
    Return repos that have this skill area (via repo_ai_dev_skills join).
    Validates that the skill area exists in the skill_areas table.
    """
    # Verify skill area exists
    sa_stmt = select(SkillArea).where(SkillArea.name == name)
    sa_result = await db.execute(sa_stmt)
    skill_area = sa_result.scalar_one_or_none()
    if skill_area is None:
        raise HTTPException(status_code=404, detail=f"Skill area '{name}' not found")

    # Fetch repos with this skill
    repos_stmt = text(
        "SELECT r.name, r.owner, r.description, r.github_url, r.primary_language, "
        "       r.stargazers_count, r.parent_stars, r.activity_score, r.readme_summary "
        "FROM repos r "
        "JOIN repo_ai_dev_skills s ON s.repo_id = r.id "
        "WHERE s.skill = :skill "
        "ORDER BY COALESCE(r.stargazers_count, r.parent_stars, 0) DESC"
    )
    repos_result = await db.execute(repos_stmt, {"skill": name})
    rows = repos_result.mappings().all()

    repos = [
        {
            "name": row["name"],
            "owner": row["owner"],
            "description": row["description"],
            "githubUrl": row["github_url"],
            "primaryLanguage": row["primary_language"],
            "stars": row["stargazers_count"] or row["parent_stars"] or 0,
            "activityScore": row["activity_score"],
            "readmeSummary": row["readme_summary"],
        }
        for row in rows
    ]

    return {
        "skillArea": {
            "id": skill_area.id,
            "name": skill_area.name,
            "lifecycleGroup": skill_area.lifecycle_group,
            "description": skill_area.description,
            "icon": skill_area.icon,
            "color": skill_area.color,
        },
        "repos": repos,
        "repoCount": len(repos),
    }


# ---------------------------------------------------------------------------
# Dynamic embedding-based taxonomy endpoints
# ---------------------------------------------------------------------------

class RebuildBody(BaseModel):
    dimension: Optional[str] = None


class AssignBody(BaseModel):
    dimension: Optional[str] = None
    threshold: float = 0.65
    limit: int = 500  # Max taxonomy values to process per call; use dimension filter for large runs


@router.get("/dimensions", response_model=dict)
async def list_dimensions(db: AsyncSession = Depends(get_db)) -> dict:
    """Return distinct dimension strings from repo_taxonomy with repo counts."""
    result = await db.execute(text(
        "SELECT dimension, COUNT(DISTINCT repo_id) AS repo_count "
        "FROM repo_taxonomy "
        "GROUP BY dimension "
        "ORDER BY repo_count DESC"
    ))
    rows = result.fetchall()
    return {
        "dimensions": [
            {"dimension": row.dimension, "repoCount": row.repo_count}
            for row in rows
        ]
    }


@router.get("/{dimension}", response_model=dict)
async def list_taxonomy_values(
    dimension: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return all taxonomy_values for a given dimension, sorted by repo_count desc."""
    result = await db.execute(text(
        "SELECT id, dimension, name, description, repo_count, trending_score, "
        "       first_seen_at, last_active_at, created_at "
        "FROM taxonomy_values "
        "WHERE dimension = :dimension "
        "ORDER BY repo_count DESC, name ASC"
    ), {"dimension": dimension})
    rows = result.fetchall()
    columns = list(result.keys())
    return {
        "dimension": dimension,
        "values": [dict(zip(columns, row)) for row in rows],
        "total": len(rows),
    }


@router.post("/admin/taxonomy/rebuild", response_model=dict, dependencies=[Depends(verify_api_key), Depends(require_admin_key)])
async def rebuild_taxonomy(
    body: RebuildBody = RebuildBody(),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Aggregate distinct raw_values from repo_taxonomy per dimension and upsert
    into taxonomy_values. Updates repo_count for each value.
    Optionally filter to a single dimension.
    """
    if body.dimension:
        dimensions_result = await db.execute(
            text("SELECT DISTINCT dimension FROM repo_taxonomy WHERE dimension = :dim"),
            {"dim": body.dimension},
        )
    else:
        dimensions_result = await db.execute(
            text("SELECT DISTINCT dimension FROM repo_taxonomy")
        )
    dimensions = [row[0] for row in dimensions_result.fetchall()]

    upserted = 0
    for dim in dimensions:
        # Single bulk INSERT…SELECT per dimension — turns O(n) round-trips into one query.
        result = await db.execute(text(
            "INSERT INTO taxonomy_values (dimension, name, repo_count, last_active_at) "
            "SELECT :dim, raw_value, COUNT(DISTINCT repo_id), NOW() "
            "FROM repo_taxonomy "
            "WHERE dimension = :dim "
            "GROUP BY raw_value "
            "ON CONFLICT (dimension, name) DO UPDATE "
            "  SET repo_count = EXCLUDED.repo_count, "
            "      last_active_at = NOW()"
        ), {"dim": dim})
        upserted += result.rowcount

    await db.commit()
    return {"status": "ok", "upserted": upserted, "dimensions": dimensions}


@router.post("/admin/taxonomy/backfill", response_model=dict, dependencies=[Depends(verify_api_key), Depends(require_admin_key)])
async def backfill_taxonomy_from_repos(db: AsyncSession = Depends(get_db)) -> dict:
    """
    Backfill repo_taxonomy from JSONB columns on repos (skill_areas, industries,
    use_cases, modalities, ai_trends, deployment_context).

    Reads repos whose JSONB enrichment columns are non-null and non-empty, then
    inserts rows into repo_taxonomy using ON CONFLICT DO NOTHING — safe to re-run.

    Use this after running the AI enricher directly against the DB (which writes to
    repos.skill_areas etc. but bypasses the ingest API and _upsert_repo_taxonomy).
    """
    dimension_columns = {
        "skill_area": "skill_areas",
        "industry": "industries",
        "use_case": "use_cases",
        "modality": "modalities",
        "ai_trend": "ai_trends",
        "deployment_context": "deployment_context",
    }

    total_inserted = 0
    repos_processed = 0

    result = await db.execute(text(
        "SELECT id, skill_areas, industries, use_cases, modalities, ai_trends, deployment_context "
        "FROM repos "
        "WHERE skill_areas IS NOT NULL "
        "   OR industries IS NOT NULL "
        "   OR use_cases IS NOT NULL "
        "   OR modalities IS NOT NULL "
        "   OR ai_trends IS NOT NULL "
        "   OR deployment_context IS NOT NULL"
    ))
    rows = result.fetchall()

    for row in rows:
        repo_id = str(row[0])
        col_values = {
            "skill_area":         row[1] or [],
            "industry":           row[2] or [],
            "use_case":           row[3] or [],
            "modality":           row[4] or [],
            "ai_trend":           row[5] or [],
            "deployment_context": row[6] or [],
        }
        repo_had_data = False
        for dimension, values in col_values.items():
            if not isinstance(values, list):
                continue
            for raw_value in values:
                if not raw_value or not isinstance(raw_value, str):
                    continue
                await db.execute(text(
                    "INSERT INTO repo_taxonomy (repo_id, dimension, raw_value, assigned_by) "
                    "VALUES (:repo_id, :dimension, :raw_value, 'enrichment') "
                    "ON CONFLICT (repo_id, dimension, raw_value) DO NOTHING"
                ), {"repo_id": repo_id, "dimension": dimension, "raw_value": raw_value.strip()})
                total_inserted += 1
                repo_had_data = True
        if repo_had_data:
            repos_processed += 1

    await db.commit()
    return {
        "status": "ok",
        "repos_processed": repos_processed,
        "rows_inserted": total_inserted,
    }


@router.post("/admin/taxonomy/embed", response_model=dict, dependencies=[Depends(verify_api_key), Depends(require_admin_key)])
async def embed_taxonomy(db: AsyncSession = Depends(get_db)) -> dict:
    """
    Generate embeddings for taxonomy_values that are missing embedding_vec.
    Uses the sentence-transformers all-MiniLM-L6-v2 singleton from app.embeddings.
    """
    from app.embeddings import get_embedding_model

    result = await db.execute(text(
        "SELECT id, dimension, name FROM taxonomy_values WHERE embedding_vec IS NULL"
    ))
    rows = result.fetchall()

    if not rows:
        return {"status": "ok", "embedded": 0}

    _EMBED_WARN_THRESHOLD = 5000
    warning: Optional[str] = None
    if len(rows) > _EMBED_WARN_THRESHOLD:
        warning = (
            f"{len(rows)} rows need embedding — this may exceed Cloud Run's timeout. "
            "Consider filtering by dimension and running in batches."
        )
        logger.warning("embed_taxonomy: %s", warning)

    model = get_embedding_model()
    texts = [f"{row.dimension}: {row.name}" for row in rows]
    embeddings = model.encode(texts, normalize_embeddings=True)

    _CHUNK_SIZE = 500
    embedded = 0
    pairs = list(zip(rows, embeddings))
    for chunk_start in range(0, len(pairs), _CHUNK_SIZE):
        chunk = pairs[chunk_start : chunk_start + _CHUNK_SIZE]
        for row, emb in chunk:
            vec_str = "[" + ",".join(str(float(v)) for v in emb) + "]"
            await db.execute(text(
                "UPDATE taxonomy_values SET embedding_vec = CAST(:vec AS vector) WHERE id = :id"
            ), {"vec": vec_str, "id": row.id})
            embedded += 1
        await db.commit()

    response: dict = {"status": "ok", "embedded": embedded}
    if warning:
        response["warning"] = warning
    return response


@router.post("/admin/taxonomy/assign", response_model=dict, dependencies=[Depends(verify_api_key), Depends(require_admin_key)])
async def assign_taxonomy(
    body: AssignBody = AssignBody(),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    For each taxonomy_value with an embedding_vec, find repos whose own embedding
    is within the similarity threshold and upsert repo_taxonomy rows with
    assigned_by='similarity'.

    Because this performs one vector query per taxonomy value, large catalogues will
    exceed Cloud Run's timeout if run all at once.  Use ``dimension`` to restrict to
    a single dimension and ``limit`` (default 500) to process in manageable batches.
    """
    threshold = body.threshold
    limit = body.limit

    if body.dimension:
        tv_result = await db.execute(text(
            "SELECT id, dimension, name FROM taxonomy_values "
            "WHERE embedding_vec IS NOT NULL AND dimension = :dim "
            "LIMIT :limit"
        ), {"dim": body.dimension, "limit": limit})
    else:
        tv_result = await db.execute(text(
            "SELECT id, dimension, name FROM taxonomy_values "
            "WHERE embedding_vec IS NOT NULL "
            "LIMIT :limit"
        ), {"limit": limit})
    taxonomy_values = tv_result.fetchall()

    assigned = 0
    for tv in taxonomy_values:
        # Find repos whose embedding is within threshold of this taxonomy value's embedding
        repo_result = await db.execute(text(
            "SELECT re.repo_id, "
            "       1 - (re.embedding_vec <=> tv.embedding_vec) AS similarity "
            "FROM repo_embeddings re "
            "JOIN taxonomy_values tv ON tv.id = :tv_id "
            "WHERE re.embedding_vec IS NOT NULL "
            "  AND 1 - (re.embedding_vec <=> tv.embedding_vec) >= :threshold"
        ), {"tv_id": tv.id, "threshold": threshold})
        repo_rows = repo_result.fetchall()

        for rrow in repo_rows:
            await db.execute(text(
                "INSERT INTO repo_taxonomy "
                "  (repo_id, dimension, raw_value, taxonomy_value_id, similarity_score, assigned_by) "
                "VALUES (:repo_id, :dimension, :raw_value, :tv_id, :sim, 'similarity') "
                "ON CONFLICT (repo_id, dimension, raw_value) DO NOTHING"
            ), {
                "repo_id": str(rrow.repo_id),
                "dimension": tv.dimension,
                "raw_value": tv.name,
                "tv_id": tv.id,
                "sim": float(rrow.similarity),
            })
            assigned += 1

    await db.commit()
    return {
        "status": "ok",
        "assigned": assigned,
        "taxonomy_values_processed": len(taxonomy_values),
        "limit": limit,
    }


@router.post(
    "/admin/taxonomy/deduplicate",
    response_model=dict,
    dependencies=[Depends(verify_api_key), Depends(require_admin_key)],
    summary="Deduplicate near-identical taxonomy values within each dimension",
)
async def deduplicate_taxonomy(
    dry_run: bool = False,
    similarity_threshold: float = 0.95,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Find taxonomy_values within the same dimension that are near-identical by
    cosine similarity (>= similarity_threshold, default 0.95), then merge the
    smaller (lower repo_count) into the larger.

    Merging means:
    1. Repoint all repo_taxonomy rows from the duplicate to the canonical value.
    2. Update canonical value's repo_count to sum of both.
    3. Delete the duplicate taxonomy_value.

    Returns a list of merges performed (or that would be performed in dry_run).
    """
    # Find pairs of taxonomy values within same dimension with high similarity
    # Only consider values that have embeddings
    result = await db.execute(text(
        """
        SELECT a.id AS id_a, a.dimension, a.name AS name_a, a.repo_count AS count_a,
               b.id AS id_b, b.name AS name_b, b.repo_count AS count_b,
               1 - (a.embedding_vec <=> b.embedding_vec) AS similarity
        FROM taxonomy_values a
        JOIN taxonomy_values b
          ON a.dimension = b.dimension
          AND a.id < b.id
          AND a.embedding_vec IS NOT NULL
          AND b.embedding_vec IS NOT NULL
          AND 1 - (a.embedding_vec <=> b.embedding_vec) >= :threshold
        ORDER BY similarity DESC
        """,
    ), {"threshold": similarity_threshold})
    pairs = result.fetchall()

    if not pairs:
        return {"merged": 0, "dry_run": dry_run, "pairs": []}

    merges = []
    already_merged: set[int] = set()

    for pair in pairs:
        id_a, id_b = pair.id_a, pair.id_b
        if id_a in already_merged or id_b in already_merged:
            continue

        # Keep the one with higher repo_count as canonical
        if pair.count_a >= pair.count_b:
            canonical_id, dup_id = id_a, id_b
            canonical_name, dup_name = pair.name_a, pair.name_b
        else:
            canonical_id, dup_id = id_b, id_a
            canonical_name, dup_name = pair.name_b, pair.name_a

        merge_info = {
            "dimension": pair.dimension,
            "canonical": canonical_name,
            "duplicate": dup_name,
            "similarity": round(float(pair.similarity), 4),
        }
        merges.append(merge_info)
        already_merged.add(dup_id)

        if not dry_run:
            # Repoint repo_taxonomy rows from dup to canonical
            await db.execute(text(
                """
                UPDATE repo_taxonomy
                SET raw_value = :canonical_name, taxonomy_value_id = :canonical_id
                WHERE taxonomy_value_id = :dup_id
                  AND NOT EXISTS (
                      SELECT 1 FROM repo_taxonomy rt2
                      WHERE rt2.repo_id = repo_taxonomy.repo_id
                        AND rt2.dimension = repo_taxonomy.dimension
                        AND rt2.raw_value = :canonical_name
                  )
                """
            ), {"canonical_name": canonical_name, "canonical_id": canonical_id, "dup_id": dup_id})

            # Delete rows that would conflict (the dup is already in canonical's position)
            await db.execute(text(
                "DELETE FROM repo_taxonomy WHERE taxonomy_value_id = :dup_id"
            ), {"dup_id": dup_id})

            # Update canonical repo_count
            new_count = max(pair.count_a, pair.count_b)
            await db.execute(text(
                "UPDATE taxonomy_values SET repo_count = :count WHERE id = :id"
            ), {"count": new_count, "id": canonical_id})

            # Delete duplicate taxonomy_value
            await db.execute(text(
                "DELETE FROM taxonomy_values WHERE id = :id"
            ), {"id": dup_id})

    if not dry_run:
        await db.commit()

    return {"merged": len(merges), "dry_run": dry_run, "pairs": merges}
