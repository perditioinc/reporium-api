#!/usr/bin/env python3
"""
rebuild_snapshot.py — one-off script to rebuild the GCS knowledge-graph snapshot.

Connects to Cloud SQL via cloud-sql-proxy (must be running on localhost:5432).
Queries repos, repo_edges; builds snapshot JSON v1; uploads to GCS.

Usage (from reporium-api/ root):
    # Terminal 1: start cloud-sql-proxy
    cloud-sql-proxy perditio-platform:us-central1:reporium-db --port=5432
    # Terminal 2: run this script
    python scripts/rebuild_snapshot.py

Env vars (optional overrides):
    PROXY_DB_URL   — psycopg2 DSN (default: localhost:5432, db=reporium, user=postgres)
    SNAPSHOT_BUCKET  — GCS bucket (default: perditio-platform-bucket)
    SNAPSHOT_OBJECT  — GCS path (default: reporium/graph/knowledge-graph.json)
    ADMIN_API_KEY  — key for cache invalidation call (optional)
    API_URL        — API base URL for cache invalidation (optional)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import psycopg2

PROXY_DB_URL = os.environ.get(
    "PROXY_DB_URL",
    "postgresql://postgres@localhost:5432/reporium",
)
SNAPSHOT_BUCKET = os.environ.get("SNAPSHOT_BUCKET", "perditio-platform-bucket")
SNAPSHOT_OBJECT = os.environ.get("SNAPSHOT_OBJECT", "reporium/graph/knowledge-graph.json")
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")
API_URL = os.environ.get(
    "API_URL", "https://reporium-api-wypbzj5gpa-uc.a.run.app"
)

# Edge-type balancing caps (mirrors build_graph_payload_from_snapshot logic)
TOTAL_EDGE_BUDGET = 10000
TYPED_CAP = max(200, TOTAL_EDGE_BUDGET // 5)  # 2000


def stars_log(stars: int) -> float:
    return math.log10(max(stars, 1))


def main() -> None:
    print(f"[rebuild_snapshot] Connecting via proxy: {PROXY_DB_URL[:40]}...")
    conn = psycopg2.connect(PROXY_DB_URL)
    cur = conn.cursor()

    # -------------------------------------------------------------------------
    # 1. Fetch all public repos
    # -------------------------------------------------------------------------
    print("[rebuild_snapshot] Fetching repos...")
    cur.execute("""
        SELECT
            id,
            name,
            owner,
            description,
            primary_category,
            COALESCE(parent_stars, stargazers_count, 0) AS stars,
            updated_at,
            quality_signals
        FROM repos
        WHERE is_private = false
        ORDER BY id
    """)
    repo_rows = cur.fetchall()
    print(f"[rebuild_snapshot]   {len(repo_rows)} public repos")

    node_index: dict[int, dict] = {}
    for (repo_id, name, owner, description, primary_category, stars, updated_at, quality_signals) in repo_rows:
        quality = None
        if isinstance(quality_signals, dict):
            quality = quality_signals.get("quality")
        elif isinstance(quality_signals, str):
            try:
                qs = json.loads(quality_signals)
                quality = qs.get("quality")
            except Exception:
                pass
        node_index[repo_id] = {
            "repo_id": repo_id,
            "name": name,
            "owner": owner,
            "description": description,
            "primary_category": primary_category,
            "stars": int(stars or 0),
            "stars_log": round(stars_log(int(stars or 0)), 4),
            "quality": quality,
            "updated_at": updated_at.isoformat() if updated_at else None,
        }

    # -------------------------------------------------------------------------
    # 2. Fetch similarity edges from repo_embeddings (SIMILAR_TO)
    # -------------------------------------------------------------------------
    print("[rebuild_snapshot] Computing similarity edges via pgvector HNSW...")
    try:
        cur.execute("""
            WITH ranked AS (
                SELECT
                    e1.repo_id   AS source_id,
                    e2.repo_id   AS target_id,
                    1 - (e1.embedding_vec <=> e2.embedding_vec) AS similarity,
                    ROW_NUMBER() OVER (
                        PARTITION BY e1.repo_id
                        ORDER BY e1.embedding_vec <=> e2.embedding_vec
                    ) AS rank
                FROM repo_embeddings e1
                CROSS JOIN LATERAL (
                    SELECT e2_inner.repo_id, e2_inner.embedding_vec
                    FROM repo_embeddings e2_inner
                    WHERE e2_inner.repo_id != e1.repo_id
                    ORDER BY e1.embedding_vec <=> e2_inner.embedding_vec
                    LIMIT 8
                ) e2
                WHERE 1 - (e1.embedding_vec <=> e2.embedding_vec) >= 0.50
            )
            SELECT DISTINCT ON (LEAST(source_id, target_id), GREATEST(source_id, target_id))
                source_id, target_id, similarity, rank
            FROM ranked
            ORDER BY LEAST(source_id, target_id), GREATEST(source_id, target_id), similarity DESC
            LIMIT 20000
        """)
        sim_rows = cur.fetchall()
        print(f"[rebuild_snapshot]   {len(sim_rows)} similarity edges computed")
    except Exception as e:
        print(f"[rebuild_snapshot] WARNING: pgvector query failed ({e}); using empty similarity edges")
        sim_rows = []

    similarity_edges = []
    for (source_id, target_id, similarity, rank) in sim_rows:
        if source_id in node_index and target_id in node_index:
            similarity_edges.append({
                "source_repo_id": source_id,
                "target_repo_id": target_id,
                "weight": round(float(similarity), 4),
                "rank": int(rank),
            })

    # -------------------------------------------------------------------------
    # 3. Fetch typed edges from repo_edges
    # -------------------------------------------------------------------------
    print("[rebuild_snapshot] Fetching typed edges from repo_edges...")
    try:
        cur.execute("""
            SELECT source_repo_id, target_repo_id, edge_type,
                   COALESCE(weight, 0.5) as weight
            FROM repo_edges
            ORDER BY weight DESC
        """)
        typed_rows = cur.fetchall()
        print(f"[rebuild_snapshot]   {len(typed_rows)} typed edges")
    except Exception as e:
        print(f"[rebuild_snapshot] WARNING: repo_edges query failed ({e}); using empty typed edges")
        typed_rows = []

    # Balance typed edges: cap each non-SIMILAR_TO type at TYPED_CAP
    type_counts: dict[str, int] = {}
    typed_edges = []
    for (source_id, target_id, edge_type, weight) in typed_rows:
        if source_id not in node_index or target_id not in node_index:
            continue
        if edge_type == "SIMILAR_TO":
            continue  # handled separately
        cnt = type_counts.get(edge_type, 0)
        if cnt >= TYPED_CAP:
            continue
        typed_edges.append({
            "source_repo_id": source_id,
            "target_repo_id": target_id,
            "edge_type": edge_type,
            "weight": round(float(weight), 4),
        })
        type_counts[edge_type] = cnt + 1

    cur.close()
    conn.close()

    # -------------------------------------------------------------------------
    # 4. Build snapshot JSON
    # -------------------------------------------------------------------------
    repos_with_embeddings = len({e["source_repo_id"] for e in similarity_edges} |
                                 {e["target_repo_id"] for e in similarity_edges})
    edge_type_counts = {}
    for e in typed_edges:
        et = e["edge_type"]
        edge_type_counts[et] = edge_type_counts.get(et, 0) + 1

    snapshot = {
        "snapshot_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "nodes": list(node_index.values()),
        "similarity_edges": similarity_edges,
        "typed_edges": typed_edges,
        "parameters": {
            "min_similarity": 0.50,
            "max_neighbours": 8,
            "typed_cap_per_type": TYPED_CAP,
        },
        "stats": {
            "total_public_repos": len(node_index),
            "repos_with_embeddings": repos_with_embeddings,
            "similarity_edges": len(similarity_edges),
            "typed_edges": len(typed_edges),
            "typed_edge_breakdown": edge_type_counts,
        },
    }

    snapshot_json = json.dumps(snapshot, ensure_ascii=False, default=str)
    size_mb = len(snapshot_json.encode("utf-8")) / 1024 / 1024
    print(f"[rebuild_snapshot] Snapshot built: {len(node_index)} nodes, "
          f"{len(similarity_edges)} sim edges, {len(typed_edges)} typed edges, "
          f"{size_mb:.1f} MB")

    # -------------------------------------------------------------------------
    # 5. Upload to GCS
    # -------------------------------------------------------------------------
    print(f"[rebuild_snapshot] Uploading to gs://{SNAPSHOT_BUCKET}/{SNAPSHOT_OBJECT} ...")
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(SNAPSHOT_BUCKET)
    blob = bucket.blob(SNAPSHOT_OBJECT)
    blob.upload_from_string(
        snapshot_json.encode("utf-8"),
        content_type="application/json",
    )
    print("[rebuild_snapshot] Upload complete.")

    # -------------------------------------------------------------------------
    # 6. Invalidate API cache
    # -------------------------------------------------------------------------
    if ADMIN_API_KEY:
        import urllib.request
        req = urllib.request.Request(
            f"{API_URL}/admin/cache/graph/invalidate",
            method="POST",
            headers={"X-Admin-Key": ADMIN_API_KEY},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                print(f"[rebuild_snapshot] Cache invalidated: {resp.read()}")
        except Exception as e:
            print(f"[rebuild_snapshot] WARNING: cache invalidation failed: {e}")
    else:
        print("[rebuild_snapshot] ADMIN_API_KEY not set; skipping cache invalidation.")

    print("[rebuild_snapshot] Done.")
    print(f"  Nodes: {len(node_index)}")
    print(f"  Similarity edges: {len(similarity_edges)}")
    print(f"  Typed edges: {len(typed_edges)} ({edge_type_counts})")


if __name__ == "__main__":
    main()
