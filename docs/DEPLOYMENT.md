# Deployment

## Production Deploy Sequence

1. Authenticate to GCP with a service account that can read Secret Manager and deploy Cloud Run.
2. Apply database schema changes before shipping the new image:

```bash
ENVIRONMENT=production GCP_PROJECT=perditio-platform alembic upgrade head
```

3. Deploy the application image or source bundle to Cloud Run.

## Health Probes

The Cloud Run service manifest should point startup or readiness probes at `GET /health`.

- `200` means the service and database are reachable.
- `503` means the API is up but the database check failed.

## Ingestion Recovery Runbook

Use this when the live API is returning repos with empty `enrichedTags`, `aiDevSkills`, or `taxonomy` data.

1. Run a full ingestion against production with a fresh cache database so unchanged repos are re-fetched instead of skipped.
2. Wait for the ingestion run to complete. The production API is updated near the end of the run, not incrementally.
3. Verify the enrichment tables are populated:

```sql
SELECT COUNT(*) FROM repo_tags;
SELECT COUNT(*) FROM repo_ai_dev_skills;
SELECT COUNT(*) FROM repo_pm_skills;
SELECT COUNT(*) FROM repo_taxonomy;
```

4. If taxonomy or gap data is still stale, run the admin refresh sequence manually:

```bash
curl -X POST https://reporium-api-573778300586.us-central1.run.app/taxonomy/admin/taxonomy/rebuild \
  -H "Authorization: Bearer $REPORIUM_API_KEY"

curl -X POST https://reporium-api-573778300586.us-central1.run.app/taxonomy/admin/taxonomy/embed \
  -H "Authorization: Bearer $REPORIUM_API_KEY"

curl -X POST https://reporium-api-573778300586.us-central1.run.app/taxonomy/admin/taxonomy/assign \
  -H "Authorization: Bearer $REPORIUM_API_KEY"

curl -X POST https://reporium-api-573778300586.us-central1.run.app/admin/gaps/rebuild \
  -H "Authorization: Bearer $REPORIUM_API_KEY"
```

5. Trigger a frontend redeploy so the static snapshot artifacts pick up the refreshed enrichment data.

## Current Limitation

The Pub/Sub recovery route at `POST /ingest/events/repo-ingested` can still encounter model-load failures when `embed_taxonomy()` tries to load the sentence-transformers model at runtime. The handler should continue with partial success instead of failing the entire event, but manual recovery remains the safest operational path until model loading is fully hardened in production.
