# ADR 004: Header-Based Admin and Ingest Auth

- Status: Accepted

## Context

Reporium exposes public read endpoints as well as operational endpoints for ingest, taxonomy rebuilds, admin maintenance, and event callbacks. Those operational routes are used by sibling repos, scheduled jobs, and internal automation rather than by end-user browsers. The system needs simple credentials that work well in Cloud Run, GitHub Actions, and service-to-service calls without introducing a full OAuth or session stack for internal platform traffic.

## Decision

Operational endpoints use explicit header-based shared-secret auth. Admin actions require `X-Admin-Key` backed by `ADMIN_API_KEY`, ingest actions require `X-Ingest-Key` backed by `INGEST_API_KEY`, and Pub/Sub push requests may additionally validate a signed bearer token when `PUBSUB_AUDIENCE` is configured. Development remains dev-safe by allowing clearly documented passthrough behavior where appropriate, but production paths are expected to set the real secrets explicitly.

## Consequences

The platform gets a small, understandable auth surface that is easy to wire from automation and easy to inspect during incident response. This fits the current internal-platform use case and keeps ingress/auth complexity low while the product is still moving fast. The tradeoff is that key rotation and secret distribution need to be handled carefully through environment management and CI/CD configuration. This decision also means documentation and `.env.example` files matter, because operators need a reliable record of which headers and env vars are required for each class of operational endpoint.
