"""
KAN-155: POST /intelligence/nl-filter

Translates a natural language query into structured filter params for the
/repos and /library/full endpoints. Single Haiku call per request — ~$0.0005.

Example input:  "actively maintained Python RAG repos with over 1000 stars"
Example output: {
    "language": "python",
    "category": "rag-retrieval",
    "min_stars": 1000,
    "sort": "stars",
    "tags": ["rag", "retrieval"],
    "quality": "high",
    "interpretation": "Python · RAG & Retrieval · 1,000+ stars · sorted by stars"
}

The structured output maps directly to existing query params on GET /repos.
"""

import asyncio
import hashlib
import json
import logging
import os

import anthropic
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.auth import require_app_token
from app.cache import cache
from app.circuit_breaker import anthropic_breaker
from app.cost_tracker import check_budget, record_cost
from app.rate_limit import rate_limit_storage
from app.slo_observer import token_observer
from app.utils import get_anthropic_key, log_nonfatal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# KAN-197 / Issue #215: Lazy singleton Anthropic client lives in app.utils.
# This thin wrapper is preserved so existing tests that patch
# ``app.routers.nl_filter._get_client`` still work.
# ---------------------------------------------------------------------------
def _get_client() -> anthropic.Anthropic:
    from app.utils import get_anthropic_client
    return get_anthropic_client()


# Per-model pricing (per 1M tokens)
_MODEL_PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20250414": {"input": 0.80, "output": 4.00},
    "claude-haiku-4-5": {"input": 0.80, "output": 4.00},
}


def _estimate_cost(input_tokens: int, output_tokens: int, model: str = "claude-haiku-4-5") -> float:
    pricing = _MODEL_PRICING.get(model, _MODEL_PRICING["claude-haiku-4-5"])
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
router = APIRouter(prefix="/intelligence", tags=["Intelligence"])
_limiter = Limiter(key_func=get_remote_address, storage_uri=rate_limit_storage)

# Valid categories in the Reporium taxonomy
_VALID_CATEGORIES = [
    "agents", "rag-retrieval", "llm-serving", "fine-tuning", "evaluation",
    "orchestration", "vector-databases", "observability", "security-safety",
    "code-generation", "data-processing", "computer-vision", "nlp-text",
    "speech-audio", "generative-media", "infrastructure",
]

_NL_FILTER_PROMPT = """You are a search query parser for Reporium — a curated library of AI/ML GitHub repositories.

Convert the following natural language search into structured filter parameters.

Natural language query: {query}

Valid categories (pick the single best match, or null if not applicable):
{categories}

Return ONLY valid JSON matching this schema — no markdown, no explanation:
{{
  "language": "<lowercase language name or null>",
  "category": "<one of the valid categories above, or null>",
  "min_stars": <integer >= 0 or null>,
  "max_stars": <integer or null>,
  "sort": "<one of: stars | updated | name, or null>",
  "tags": ["<relevant tag keywords — lowercase, specific>"],
  "quality": "<high | medium | low | null — only set if explicitly mentioned>",
  "maturity": "<production | beta | prototype | research | null — only if mentioned>",
  "exclude_archived": <true if user wants active repos, false otherwise>,
  "interpretation": "<short human-readable summary: what filters were applied, e.g. 'Python · RAG & Retrieval · 1,000+ stars'>"
}}

Rules:
- Only set fields that are clearly implied by the query — don't guess
- min_stars: "popular" → 500, "widely used" → 1000, "very popular" → 5000
- sort: default to "stars" if popularity is mentioned, "updated" if recency is mentioned
- tags: extract meaningful keywords (e.g. "rag", "langchain", "fine-tuning") — max 5
- exclude_archived: true if query mentions "active", "maintained", "recent", "working"
- interpretation: be concise, use · as separator"""


class NLFilterRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=300,
                       description="Natural language filter query")


class NLFilterResponse(BaseModel):
    language: str | None = None
    category: str | None = None
    min_stars: int | None = None
    max_stars: int | None = None
    sort: str | None = None
    tags: list[str] = Field(default_factory=list)
    quality: str | None = None
    maturity: str | None = None
    exclude_archived: bool = False
    interpretation: str
    # Convenience: ready-to-use query string for GET /repos
    query_params: str = ""


def _build_query_params(r: NLFilterResponse) -> str:
    """Build a URL query string from the filter response."""
    parts = []
    if r.language:
        parts.append(f"language={r.language}")
    if r.category:
        parts.append(f"category={r.category}")
    if r.min_stars is not None:
        parts.append(f"min_stars={r.min_stars}")
    if r.sort:
        parts.append(f"sort={r.sort}")
    if r.exclude_archived:
        parts.append("exclude_archived=true")
    return "&".join(parts)


@router.post("/nl-filter", response_model=NLFilterResponse)
@_limiter.limit("15/minute")
async def nl_filter(
    request: Request,
    body: NLFilterRequest,
    _app: None = Depends(require_app_token),
) -> NLFilterResponse:
    """
    Translate a natural language query into structured filter params.
    Single Haiku call — ~$0.0005 per request. Results cached 1 hour by query hash.
    Requires X-App-Token header, rate-limited to 15 req/min per IP.
    """
    cache_key = f"nl_filter:{hashlib.sha256(body.query.lower().strip().encode()).hexdigest()[:16]}"
    cached = await cache.get(cache_key)
    if cached:
        # KAN-ask-spend: record cache hit for /metrics/spend hit-rate.
        try:
            token_observer.record_cache_hit("/intelligence/nl-filter")
        except Exception:
            log_nonfatal("token_observer.record_cache_hit")
        return NLFilterResponse(**cached)

    if not await check_budget():
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable — daily usage limit reached. Try again tomorrow.",
        )

    prompt = _NL_FILTER_PROMPT.format(
        query=body.query,
        categories="\n".join(f"  - {c}" for c in _VALID_CATEGORIES),
    )

    try:
        def _call_haiku():
            return _get_client().messages.create(
                model="claude-haiku-4-5",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )

        with anthropic_breaker:
            response = await asyncio.to_thread(_call_haiku)
        actual_cost = _estimate_cost(response.usage.input_tokens, response.usage.output_tokens, "claude-haiku-4-5")
        await record_cost(actual_cost, model="claude-haiku-4-5")
        # KAN-ask-spend: per-route token + cost accumulator for /metrics/spend.
        try:
            token_observer.record_tokens(
                route="/intelligence/nl-filter",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                usd_cost=actual_cost,
                model="claude-haiku-4-5",
            )
        except Exception:
            log_nonfatal("token_observer.record_tokens")
        raw = response.content[0].text.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        data = json.loads(raw)

        # Validate/normalise
        if data.get("category") not in _VALID_CATEGORIES:
            data["category"] = None
        if data.get("sort") not in {"stars", "updated", "name", None}:
            data["sort"] = None
        if data.get("quality") not in {"high", "medium", "low", None}:
            data["quality"] = None
        if data.get("maturity") not in {"production", "beta", "prototype", "research", None}:
            data["maturity"] = None
        data.setdefault("tags", [])
        data["tags"] = [str(t).lower().strip() for t in data["tags"][:5] if t]
        data.setdefault("exclude_archived", False)
        data.setdefault("interpretation", body.query)

        result = NLFilterResponse(**data)
        result.query_params = _build_query_params(result)

        await cache.set(cache_key, result.model_dump(), ttl=3600)  # 1h cache
        logger.info("nl_filter: query_hash=%s → %s", hashlib.sha256(body.query.encode()).hexdigest()[:12], result.query_params)
        return result

    except json.JSONDecodeError as e:
        logger.error("nl_filter: JSON parse failed for query_hash=%s: %s", hashlib.sha256(body.query.encode()).hexdigest()[:12], e)
        raise HTTPException(status_code=502, detail="Filter parsing failed — try rephrasing")
    except Exception as e:
        logger.error("nl_filter: unexpected error: %s", e)
        raise HTTPException(status_code=500, detail="Filter service unavailable")
