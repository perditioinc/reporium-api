# ADR 003: Pub/Sub-Driven Taxonomy Refresh

- Status: Accepted

## Context

Ingestion, taxonomy assignment, gap analysis, and portfolio intelligence are tightly related, but manual or scheduled refreshes create stale state between those layers. Reporium already has an event model and Pub/Sub direction in the broader suite. The API now exposes admin-style rebuild endpoints for taxonomy and intelligence refresh, and ingestion can publish completion events when a batch finishes.

## Decision

The platform uses Pub/Sub to trigger taxonomy and intelligence refresh work after ingestion instead of relying only on polling or cron-style periodic rebuilds. Ingestion publishes a repo-ingested event, and the API accepts a protected push callback that can trigger taxonomy embedding, assignment, gap rebuilding, and portfolio insight refresh in response to that event.

## Consequences

Taxonomy coverage and intelligence outputs stay closer to ingestion freshness with fewer manual steps. The operator gets a clearer event boundary for “new corpus state is available,” which also makes it easier to attach future notifications or audit records. This design reduces drift between the repos table, taxonomy tables, and insight surfaces. The tradeoff is that the system now has to defend an additional event-ingest surface and reason about push-auth verification, idempotency, and safe replay behavior. That is still a better fit than polling because refresh cost is tied to actual updates rather than to a blind schedule.
