from pydantic import BaseModel

from app.schemas.repo import RepoSummary


class LibraryStats(BaseModel):
    total_repos: int
    total_forks: int
    total_non_forks: int
    languages: dict[str, int]
    top_tags: list[str]
    last_updated: str | None = None


class CategorySummary(BaseModel):
    id: str
    name: str
    count: int


class TagMetric(BaseModel):
    tag: str
    count: int
    commit_velocity: float


class LibraryResponse(BaseModel):
    repos: list[RepoSummary]
    stats: LibraryStats
    categories: list[CategorySummary]
    tag_metrics: list[TagMetric]
    total: int
    page: int
    limit: int
