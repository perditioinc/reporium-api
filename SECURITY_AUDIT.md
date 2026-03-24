# Security Audit

Date: 2026-03-24
Scope: `reporium-api` endpoint surface, auth checks, error exposure, rate limiting, validation bounds, and CORS.

## Summary

No critical unauthenticated admin or ingest endpoint exposures were found in the current code review. The most important hardening change in this pass was tightening the legacy `POST /events/ingest` placeholder so it now requires both the general API key and the ingest key instead of only the broader API key.

## Reviewed Areas

- `admin/*` routes
- `ingest/*` routes and Pub/Sub push handlers
- intelligence and search routes
- taxonomy admin routes
- platform and webhook endpoints
- global CORS and security headers in `app/main.py`

## Findings

### Fixed

1. `POST /events/ingest` accepted only `verify_api_key`.
   - Risk: broader internal API credentials could hit an event-ingest surface that is operational in nature.
   - Fix: route now also requires `require_ingest_key`.

### Medium

1. Admin write routes are authenticated but do not yet have dedicated per-route rate limits.
   - Impact: abuse is limited by shared global limits, but the admin plane would benefit from explicit write throttles.

2. Several internal/admin endpoints still use generic `dict` response contracts.
   - Impact: docs clarity is weaker, and schema drift is easier to miss.
   - Follow-up: continue the endpoint response-model cleanup started in KAN-20.

### Low

1. `audit_status()` still suppresses database-check exceptions internally.
   - Impact: no sensitive error details leak to clients, which is good, but operators lose failure detail unless logs are inspected.

2. Webhook verification is dev-safe when `GITHUB_WEBHOOK_SECRET` is unset.
   - Impact: expected for local development, but production deployment should always set the secret.

## Auth Review

- `admin/*`: protected by `verify_api_key` and `require_admin_key`
- `ingest/*`: protected by `verify_api_key` and `require_ingest_key`
- `/ingest/events/repo-ingested`: protected by `require_ingest_key` and Pub/Sub push verification
- `/events/ingest`: now protected by `verify_api_key` and `require_ingest_key`
- webhooks: protected by HMAC verification when `GITHUB_WEBHOOK_SECRET` is configured

## Rate Limit Review

- Global default limits are active through SlowAPI middleware.
- Search and semantic search have explicit limits.
- Intelligence endpoints have explicit limits.
- Admin write routes should get explicit per-route limits in a follow-up.

## Input Validation Review

- Batch ingest has a hard `MAX_BATCH = 100`.
- Search endpoints bound query length and result limits.
- Semantic search uses bounded `limit <= 50`.
- Taxonomy and analytics query parameters have minimum/maximum guards where they are exposed.

## CORS Review

- Allowed origins are restricted to the current Reporium production domains and GitHub Pages origin.
- Methods are limited to `GET` and `POST`.

## Recommended Next Steps

1. Add explicit rate limits to admin write routes.
2. Replace remaining ad hoc `dict` response contracts with typed schemas where practical.
3. Keep production webhook secret configuration mandatory in deployment docs and checks.
