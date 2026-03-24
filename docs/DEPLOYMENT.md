# Deployment

## Production Deploy Sequence

1. Authenticate to GCP with a service account that can read Secret Manager and deploy Cloud Run.
2. Apply database schema changes before shipping the new image:

```bash
ENVIRONMENT=production GCP_PROJECT=perditio-platform alembic upgrade head
```

3. Deploy the application image or source bundle to Cloud Run.

## Health Probes

The Cloud Run service manifest in [deploy/service.yaml](/C:/DEV/PERDITIO_PLATFORM/reporium-api/deploy/service.yaml) points the startup probe at `GET /health`.

- `200` means the service and database are reachable.
- `503` means the API is up but the database check failed.

## GitHub Actions

The deploy workflow now runs `alembic upgrade head` before the Cloud Run deploy step so new code does not boot against an older schema.
