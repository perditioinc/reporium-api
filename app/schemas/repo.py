from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


# --- Sub-schemas ---

class TaxonomyEntry(BaseModel):
    dimension: str
    value: str
    similarityScore: float | None = None
    assignedBy: str = "enrichment"


class CategoryRef(BaseModel):
    category_id: str
    category_name: str
    is_primary: bool = False


class BuilderRef(BaseModel):
    login: str
    display_name: str | None = None
    org_category: str | None = None
    is_known_org: bool = False


class LanguageRef(BaseModel):
    language: str
    bytes: int
    percentage: float


class CommitRef(BaseModel):
    sha: str
    message: str
    author: str | None = None
    committed_at: datetime
    url: str | None = None


# --- Repo output schemas ---

class RepoSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    owner: str
    description: str | None = None
    is_fork: bool
    forked_from: str | None = None
    primary_language: str | None = None
    github_url: str

    fork_sync_state: str | None = None
    behind_by: int = 0
    ahead_by: int = 0

    upstream_created_at: datetime | None = None
    forked_at: datetime | None = None
    your_last_push_at: datetime | None = None
    upstream_last_push_at: datetime | None = None

    parent_stars: int | None = None
    parent_forks: int | None = None
    parent_is_archived: bool = False
    stargazers_count: int | None = None
    open_issues_count: int = 0

    commits_last_7_days: int = 0
    commits_last_30_days: int = 0
    commits_last_90_days: int = 0

    readme_summary: str | None = None
    activity_score: int = 0
    problem_solved: str | None = None
    license_spdx: str | None = None

    ingested_at: datetime
    updated_at: datetime
    github_updated_at: datetime | None = None

    tags: list[str] = []
    categories: list[CategoryRef] = []
    allCategories: list[str] = []
    builders: list[BuilderRef] = []
    ai_dev_skills: list[str] = []
    pm_skills: list[str] = []
    languages: list[LanguageRef] = []
    taxonomy: list[TaxonomyEntry] = []


class RepoDetail(RepoSummary):
    commits: list[CommitRef] = []


class RepoSemanticResult(RepoSummary):
    similarity: float


# --- Ingest input schemas ---

class CategoryIngest(BaseModel):
    category_id: str
    category_name: str
    is_primary: bool = False


class BuilderIngest(BaseModel):
    login: str
    display_name: str | None = None
    org_category: str | None = None
    is_known_org: bool = False


class LanguageIngest(BaseModel):
    language: str
    bytes: int
    percentage: float


class CommitIngest(BaseModel):
    sha: str
    message: str
    author: str | None = None
    committed_at: datetime
    url: str | None = None


class RepoIngestItem(BaseModel):
    name: str
    owner: str
    description: str | None = None
    is_fork: bool = False
    is_private: bool = False
    forked_from: str | None = None
    primary_language: str | None = None
    github_url: str

    fork_sync_state: str | None = None
    behind_by: int = 0
    ahead_by: int = 0

    github_created_at: datetime | None = None
    upstream_created_at: datetime | None = None
    forked_at: datetime | None = None
    your_last_push_at: datetime | None = None
    upstream_last_push_at: datetime | None = None

    parent_stars: int | None = None
    parent_forks: int | None = None
    parent_is_archived: bool = False
    stargazers_count: int | None = None
    open_issues_count: int = 0

    commits_last_7_days: int = 0
    commits_last_30_days: int = 0
    commits_last_90_days: int = 0

    readme_summary: str | None = None
    activity_score: int = 0

    github_updated_at: datetime | None = None

    tags: list[str] = []
    categories: list[CategoryIngest] = []
    builders: list[BuilderIngest] = []
    ai_dev_skills: list[str] = []
    pm_skills: list[str] = []
    languages: list[LanguageIngest] = []
    commits: list[CommitIngest] = []

    license_spdx: str | None = None

    # Dynamic taxonomy dimensions
    skill_areas: list[str] = []
    industries: list[str] = []
    use_cases: list[str] = []
    modalities: list[str] = []
    ai_trends: list[str] = []
    deployment_context: list[str] = []
    dependencies: list[str] = []


class RepoEnrichItem(BaseModel):
    readme_summary: str | None = None
    activity_score: int | None = None
    tags: list[str] | None = None
    ai_dev_skills: list[str] | None = None
    pm_skills: list[str] | None = None

    # Dynamic taxonomy dimensions
    skill_areas: list[str] | None = None
    industries: list[str] | None = None
    use_cases: list[str] | None = None
    modalities: list[str] | None = None
    ai_trends: list[str] | None = None
    deployment_context: list[str] | None = None


class IngestResponse(BaseModel):
    upserted: int
    errors: list[str] = []
