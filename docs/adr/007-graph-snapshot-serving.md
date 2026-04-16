# ADR 007: Serve The Knowledge Graph From A Published Snapshot

## Status

Accepted on 2026-04-13.

## Context

`reporium.com` depends on `GET /graph/edges` for both the home-page knowledge graph widget and the full graph page. The previous implementation built the graph on demand from live PostgreSQL reads against `repo_embeddings`, then merged typed edges from `repo_edges`.

That design created a production coupling between page availability and the primary database's read health. On 2026-04-13, a database quota/provider incident caused `/graph/edges` to fail and the production graph UI disappeared.

The graph itself is not user-generated request-specific data. It is a derived platform artifact that changes on ingestion/rebuild cadence, not on every browser request.

## Decision

The canonical serving path for `GET /graph/edges` is now:

1. Redis response cache
2. Published graph snapshot artifact
3. Live database query fallback

The snapshot artifact is produced by `reporium-ingestion` and contains:

- public graph nodes with visualization metadata
- directed similarity edges with rank and weight
- typed graph edges (`DEPENDS_ON`, `COMPATIBLE_WITH`, `ALTERNATIVE_TO`, `EXTENDS`)
- snapshot generation metadata and repo-count diagnostics

The API filters the snapshot in memory for `limit`, `min_similarity`, `neighbours`, and `since`, then returns the existing response shape. Live database reads remain available as a fallback path, not the primary contract.

## Consequences

Positive:

- production graph availability no longer depends on request-time database health
- graph serving cost and latency become predictable
- the frontend and API continue using the same endpoint contract
- the snapshot can be refreshed independently from API deployment

Tradeoffs:

- graph freshness is tied to snapshot publication cadence
- artifact publication becomes a required operational dependency
- unusual query shapes still rely on the snapshot's retained neighbour depth

## Rollout Notes

- `reporium-ingestion` publishes the artifact to `GRAPH_SNAPSHOT_BUCKET` / `GRAPH_SNAPSHOT_OBJECT`
- `scripts/publish_graph_snapshot.py` provides a read-only publication path for emergency refreshes
- `reporium-api` reads the snapshot from GCS or a local test path and caches it in-process
- regression tests cover snapshot-first serving and publisher output
