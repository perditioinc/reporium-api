"""
POST /intelligence/query — Semantic search + Claude-powered answers over the repo knowledge base.

Requires X-API-Key header matching INGESTION_API_KEY.
Cost: ~$0.01 per query (Claude API for answer generation).
"""

import json
import logging
import os
from datetime import datetime, timezone

import anthropic
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sentence_transformers import SentenceTransformer

from app.auth import verify_api_key
from app.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/intelligence", tags=["Intelligence"])

# Load embedding model once at startup
_model = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading sentence-transformers model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Model loaded")
    return _model


def _get_anthropic_key() -> str:
    """Get Anthropic API key from env or Secret Manager. Strip whitespace."""
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    try:
        from google.cloud import secretmanager
        client = secretmanager.SecretManagerServiceClient()
        project = os.getenv("GCP_PROJECT", "perditio-platform")
        name = f"projects/{project}/secrets/anthropic-api-key/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8").strip()
    except Exception:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=10, ge=1, le=50)


class SourceRepo(BaseModel):
    name: str
    owner: str
    forked_from: str | None
    description: str | None
    stars: int | None
    relevance_score: float
    problem_solved: str | None
    integration_tags: list[str]


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceRepo]
    question: str
    model: str
    answered_at: str
    embedding_candidates: int
    tokens_used: dict


@router.post("/query", response_model=QueryResponse)
async def intelligence_query(
    req: QueryRequest,
    db: AsyncSession = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    """
    Ask a natural language question about the repo knowledge base.

    1. Embed the question with sentence-transformers
    2. Find top-K most similar repos via cosine similarity
    3. Send repo context + question to Claude for answer generation
    4. Return answer with source repos and relevance scores
    """
    model = _get_model()

    # 1. Embed the question
    query_embedding = model.encode(req.question)

    # 2. Get all embeddings and compute similarity
    result = await db.execute(text("""
        SELECT r.id, r.name, r.owner, r.forked_from, r.description,
               r.parent_stars, r.readme_summary, r.problem_solved,
               r.integration_tags, r.dependencies,
               e.embedding
        FROM repo_embeddings e
        JOIN repos r ON r.id = e.repo_id;
    """))
    rows = result.fetchall()

    scored = []
    for row in rows:
        repo_embedding = np.array(json.loads(row.embedding))
        sim = cosine_similarity(query_embedding, repo_embedding)
        scored.append({
            "id": row.id,
            "name": row.name,
            "owner": row.owner,
            "forked_from": row.forked_from,
            "description": row.description,
            "stars": row.parent_stars,
            "readme_summary": row.readme_summary,
            "problem_solved": row.problem_solved,
            "integration_tags": row.integration_tags if isinstance(row.integration_tags, list)
                                else json.loads(row.integration_tags) if row.integration_tags else [],
            "dependencies": row.dependencies if isinstance(row.dependencies, list)
                           else json.loads(row.dependencies) if row.dependencies else [],
            "similarity": sim,
        })

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    top_candidates = scored[:20]  # Get 20 for context, return top_k
    top_for_answer = top_candidates[:req.top_k]

    # 3. Build context for Claude
    context_parts = []
    for i, repo in enumerate(top_for_answer, 1):
        upstream = repo["forked_from"] or f"{repo['owner']}/{repo['name']}"
        tags_str = ", ".join(repo["integration_tags"]) if repo["integration_tags"] else "none"
        deps_str = ", ".join(repo["dependencies"][:10]) if repo["dependencies"] else "none"

        context_parts.append(f"""
Repo {i}: {upstream} ({repo['stars'] or 0} stars)
Description: {repo['description'] or 'N/A'}
Summary: {(repo['readme_summary'] or 'N/A')[:300]}
Problem solved: {repo['problem_solved'] or 'N/A'}
Integration tags: {tags_str}
Key dependencies: {deps_str}
Relevance score: {repo['similarity']:.4f}
""".strip())

    context = "\n\n".join(context_parts)

    # Also check knowledge graph edges for related repos
    if top_for_answer:
        top_ids = [str(r["id"]) for r in top_for_answer[:5]]
        placeholders = ", ".join([f"'{tid}'" for tid in top_ids])
        edge_result = await db.execute(text(f"""
            SELECT e.edge_type, e.weight, e.evidence,
                   r1.name as source_name, r1.forked_from as source_upstream,
                   r2.name as target_name, r2.forked_from as target_upstream
            FROM repo_edges e
            JOIN repos r1 ON r1.id = e.source_repo_id
            JOIN repos r2 ON r2.id = e.target_repo_id
            WHERE e.source_repo_id::text IN ({placeholders})
               OR e.target_repo_id::text IN ({placeholders})
            LIMIT 20;
        """))
        edge_rows = edge_result.fetchall()
        if edge_rows:
            edge_context = "\n\nKnowledge graph relationships:\n"
            for er in edge_rows:
                src = er.source_upstream or er.source_name
                tgt = er.target_upstream or er.target_name
                edge_context += f"- {src} {er.edge_type} {tgt} (evidence: {er.evidence})\n"
            context += edge_context

    # 4. Call Claude
    api_key = _get_anthropic_key()
    client = anthropic.Anthropic(api_key=api_key)

    system_prompt = """You are the Reporium Intelligence assistant. You answer questions about AI development tools and GitHub repositories tracked in the Reporium platform.

Rules:
- Only cite repos that appear in the provided context. Never make up repo names.
- Include the upstream repo name (owner/name) when citing a repo.
- Include star count when relevant for credibility.
- Be specific about what each repo does based on its summary and problem_solved fields.
- If the context doesn't contain enough information to answer, say so honestly.
- Keep answers concise but informative — 2-4 paragraphs max."""

    user_prompt = f"""Based on the following repos from the Reporium knowledge base, answer this question:

Question: {req.question}

Available repos (ranked by relevance):
{context}

Answer the question using only the repos above. Cite specific repos by name."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    answer = message.content[0].text
    tokens_used = {
        "input": message.usage.input_tokens,
        "output": message.usage.output_tokens,
        "total": message.usage.input_tokens + message.usage.output_tokens,
    }

    # 5. Build response
    sources = []
    for repo in top_for_answer:
        sources.append(SourceRepo(
            name=repo["name"],
            owner=repo["owner"],
            forked_from=repo["forked_from"],
            description=repo["description"],
            stars=repo["stars"],
            relevance_score=round(repo["similarity"], 4),
            problem_solved=repo["problem_solved"],
            integration_tags=repo["integration_tags"],
        ))

    return QueryResponse(
        answer=answer,
        sources=sources,
        question=req.question,
        model="claude-sonnet-4-20250514",
        answered_at=datetime.now(timezone.utc).isoformat(),
        embedding_candidates=len(scored),
        tokens_used=tokens_used,
    )
