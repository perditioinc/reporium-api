from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, field_validator


class TrendSnapshotOut(BaseModel):
    id: UUID
    snapshotted_at: datetime
    tag: str
    category: str | None = None
    repo_count: int
    commit_count_7d: int = 0


class TrendSnapshotIn(BaseModel):
    tag: str
    category: str | None = None
    repo_count: int
    commit_count_7d: int = 0


class GapAnalysisOut(BaseModel):
    id: UUID
    generated_at: datetime
    skill: str
    severity: str
    repo_count: int
    why: str | None = None
    trend: str | None = None
    essential_repos: list | dict | None = None


class GapAnalysisIn(BaseModel):
    skill: str
    severity: str
    repo_count: int
    why: str | None = None
    trend: str | None = None
    essential_repos: list | dict | None = None


class IngestionLogOut(BaseModel):
    id: UUID
    started_at: datetime
    completed_at: datetime | None = None
    mode: str
    repos_fetched: int = 0
    repos_updated: int = 0
    api_calls_made: int = 0
    errors: list = []
    status: str

    @field_validator("errors", mode="before")
    @classmethod
    def coerce_null_errors(cls, v):
        """JSONB errors column stores null for old rows — coerce to empty list."""
        return v if v is not None else []


class IngestionLogIn(BaseModel):
    mode: str
    repos_fetched: int | None = None
    repos_updated: int | None = None
    api_calls_made: int | None = None
    errors: list | None = None
    status: str | None = None
    completed_at: datetime | None = None


class StatsResponse(BaseModel):
    total_repos: int
    total_forks: int
    total_non_forks: int
    languages: dict[str, int]
    categories: dict[str, int]
    top_tags: list[str]
    sync_states: dict[str, int]
    last_ingestion: IngestionLogOut | None = None


class TaxonomyGapItem(BaseModel):
    dimension: str
    name: str
    repo_count: int
    gap_score: float   # 0.0 (well covered) → 1.0 (completely absent)
    severity: str      # "low" | "medium" | "high"
