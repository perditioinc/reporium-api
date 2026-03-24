# ADR 002: Open Taxonomy Design

- Status: Accepted

## Context

Early taxonomy work used fixed lists and UI-side mappings for skills and related metadata. That approach created drift between ingestion, API responses, and frontend filters, and it made each new dimension expensive because every repo effectively needed to be reconsidered against a hardcoded catalog. Reporium is now evolving toward a portfolio where industries, use cases, deployment contexts, AI trends, and other dimensions should be able to grow organically as the dataset changes.

## Decision

The taxonomy is treated as open and data-driven rather than fixed and hardcoded. New values originate from enrichment output or admin rebuild flows and are stored in database tables instead of source-code enums. The API exposes those values dynamically for filtering, analytics, and search. Frontend experiences consume live dimensions and values from the API rather than bundling static skill lists as the source of truth.

## Consequences

The product can evolve dimensions without frontend redeploys or one-off migration scripts for every new label. Reporium becomes better suited to emerging AI categories because the system can capture new values as the corpus changes instead of waiting for manual taxonomy curation in code. The tradeoff is that naming quality and duplication management become an operational concern, so admin cleanup tools, pruning, embedding rebuilds, and documentation need to stay strong. This decision also increases the importance of clear analytics and moderation around taxonomy values, because the database is now the contract that every consumer sees.
