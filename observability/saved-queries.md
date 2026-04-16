# Reporium — Cloud Logging Saved Queries

Saved in GCP Cloud Logging under project `perditio-platform` (SHARED visibility).

## Queries

### 5xx by Route
**ID**: `projects/perditio-platform/locations/global/savedQueries/11393694659853358073`
```
resource.type="cloud_run_revision"
resource.labels.service_name="reporium-api"
severity>=ERROR
```
Use: Investigate HTTP 500/503 errors. Group by `jsonPayload.path` in Cloud Logging.

### Slow Requests (>1s)
**ID**: `projects/perditio-platform/locations/global/savedQueries/5982249146868857282`
```
resource.type="cloud_run_revision"
resource.labels.service_name="reporium-api"
jsonPayload.duration_ms>1000
```
Use: Find requests taking >1000ms. Fields available: `path`, `method`, `status_code`, `duration_ms`, `request_id`, `trace_id`.

### Auth Failures (401/403)
**ID**: `projects/perditio-platform/locations/global/savedQueries/15004764605314558217`
```
resource.type="cloud_run_revision"
resource.labels.service_name="reporium-api"
(jsonPayload.status_code=401 OR jsonPayload.status_code=403)
```
Use: Track unauthorized access attempts. Cross-reference with `request_id` for full trace.

## Access
Cloud Console → Logging → Log Explorer → Saved queries tab (filter by "SHARED").

## Structured Log Fields (as of 2026-04-15)
All request logs include:
- `timestamp` — ISO 8601
- `level` — INFO/WARNING/ERROR
- `message` — "request"
- `logger` — "app.main"
- `request_id` — 8-char UUID prefix per request (for correlation within a trace)
- `trace_id` — X-Cloud-Trace-Context header value (Cloud Run → Cloud Trace correlation)
- `method` — HTTP method
- `path` — URL path (query string redacted)
- `status_code` — HTTP response status
- `duration_ms` — end-to-end request duration in milliseconds
