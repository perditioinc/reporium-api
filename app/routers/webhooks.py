"""
GitHub webhook receiver.

POST /webhooks/github — receives push and ping events from GitHub.
Verifies X-Hub-Signature-256 using GITHUB_WEBHOOK_SECRET env var (HMAC-SHA256).
If the secret is not set, verification is skipped (dev-safe).
"""

import hashlib
import hmac
import json
import logging
import os

from fastapi import APIRouter, Header, HTTPException, Request, Response

router = APIRouter(tags=["Webhooks"])
logger = logging.getLogger(__name__)

_IS_PRODUCTION = os.getenv("ENVIRONMENT") == "production"


def _verify_signature(body: bytes, signature_header: str | None) -> bool:
    """
    Verify GitHub HMAC-SHA256 signature.
    Returns True if signature is valid or if no secret is configured (dev only).
    Returns False if secret is configured but signature doesn't match.
    In production, missing secret rejects all requests.
    """
    secret = os.getenv("GITHUB_WEBHOOK_SECRET", "")
    if not secret:
        if _IS_PRODUCTION:
            logger.error("GITHUB_WEBHOOK_SECRET not set in production — rejecting")
            return False
        return True  # Dev-safe: skip verification only in non-production

    if not signature_header:
        return False

    expected = "sha256=" + hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header)


@router.post("/webhooks/github")
async def github_webhook(
    request: Request,
    x_hub_signature_256: str | None = Header(default=None),
    x_github_event: str | None = Header(default=None),
) -> Response:
    """
    Receive GitHub webhook events.

    Supported events:
    - ping: acknowledge receipt, return 200 OK
    - push: log the repo name, mark cache as stale for future re-fetch

    Returns 400 if signature verification fails.
    Returns 200 for all known event types.
    """
    body = await request.body()

    if not _verify_signature(body, x_hub_signature_256):
        logger.warning(
            "GitHub webhook signature mismatch (event=%s)",
            x_github_event,
        )
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    event_type = x_github_event or "unknown"
    logger.info("GitHub webhook received: event=%s", event_type)

    if event_type == "ping":
        return Response(content='{"status":"pong"}', media_type="application/json")

    if event_type == "push":
        try:
            import json
            payload = json.loads(body)
            repo_name = (
                payload.get("repository", {}).get("full_name")
                or payload.get("repository", {}).get("name")
                or "unknown"
            )
            logger.info(
                "GitHub push event for repo=%s ref=%s",
                repo_name,
                payload.get("ref", "unknown"),
            )

            # Mark cache as stale for this repo so the next read triggers a re-fetch.
            # Full re-ingestion is handled by the scheduled ingestion pipeline;
            # here we just invalidate the cached response for this repo.
            from app.cache import cache
            await cache.invalidate(f"repos:detail:{repo_name}")
            await cache.invalidate("repos:list:*")
            await cache.invalidate("stats:overview")

        except Exception as exc:
            logger.warning("Failed to process push webhook: %s", exc)

        return Response(
            content='{"status":"accepted","event":"push"}',
            media_type="application/json",
        )

    # Unknown/unsupported event — return 200 to avoid GitHub retries
    logger.info("Unhandled GitHub event type: %s", event_type)
    return Response(
        content=json.dumps({"status": "ignored", "event": event_type}),
        media_type="application/json",
    )
