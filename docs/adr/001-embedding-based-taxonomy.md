# ADR 001: Embedding-Based Taxonomy Assignment

- Status: Accepted

## Context

Reporium needs taxonomy assignment to scale across an expanding portfolio without repeatedly paying for full Claude re-enrichment runs. The platform already stores repo embeddings in Postgres with pgvector and already performs similarity operations for semantic search. At the same time, the taxonomy itself is becoming more open-ended: new values can appear in multiple dimensions over time, and those values need to be assignable across the existing corpus without resummarizing every repo.

## Decision

Taxonomy assignment is performed through pgvector cosine similarity against precomputed repo embeddings instead of rerunning Claude for every taxonomy change. Each taxonomy value receives its own embedding once, then assignment is handled by SQL similarity queries that write the resulting matches into `repo_taxonomy` with similarity metadata and an `assigned_by` marker. Claude remains responsible for repo-local enrichment tasks such as summary generation, problem framing, and initial open-ended hints, but not for ongoing taxonomy expansion.

## Consequences

Adding a new taxonomy value becomes cheap and fast because it requires one embedding operation plus one batch assignment query. Existing repos can be reclassified across the portfolio without another per-repo model call. The system becomes more deterministic and easier to backfill in production because taxonomy rebuilds are SQL-driven and auditable. The tradeoff is that taxonomy quality now depends on embedding quality and threshold tuning rather than a bespoke prompt for each expansion. That means the platform must retain admin rebuild controls, similarity thresholds, and reviewable outputs so operators can inspect unexpected assignments.
