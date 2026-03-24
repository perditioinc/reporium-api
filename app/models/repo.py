import uuid as _uuid_mod
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Boolean, Float, ForeignKey, Integer, Text, TIMESTAMP
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.database import Base


class Repo(Base):
    __tablename__ = "repos"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    owner: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    is_fork: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    is_private: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    forked_from: Mapped[str | None] = mapped_column(Text)
    primary_language: Mapped[str | None] = mapped_column(Text)
    github_url: Mapped[str] = mapped_column(Text, nullable=False)

    # Sync status
    fork_sync_state: Mapped[str | None] = mapped_column(Text)
    behind_by: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    ahead_by: Mapped[int] = mapped_column(Integer, default=0, server_default="0")

    # Dates
    github_created_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    upstream_created_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    forked_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    your_last_push_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    upstream_last_push_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))

    # Parent stats (for forks — points to upstream repo)
    parent_stars: Mapped[int | None] = mapped_column(Integer)
    parent_forks: Mapped[int | None] = mapped_column(Integer)
    parent_is_archived: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")

    # Own star count (for non-fork / built repos)
    stargazers_count: Mapped[int | None] = mapped_column(Integer)
    open_issues_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")

    # Activity
    commits_last_7_days: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    commits_last_30_days: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    commits_last_90_days: Mapped[int] = mapped_column(Integer, default=0, server_default="0")

    # Enrichment
    readme_summary: Mapped[str | None] = mapped_column(Text)
    activity_score: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    quality_signals: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Metadata
    ingested_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    github_updated_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))

    # Enrichment extras
    problem_solved: Mapped[str | None] = mapped_column(Text)
    license_spdx: Mapped[str | None] = mapped_column(Text)

    # Relationships
    tags: Mapped[list["RepoTag"]] = relationship(
        back_populates="repo", cascade="all, delete-orphan"
    )
    categories: Mapped[list["RepoCategory"]] = relationship(
        back_populates="repo", cascade="all, delete-orphan"
    )
    builders: Mapped[list["RepoBuilder"]] = relationship(
        back_populates="repo", cascade="all, delete-orphan"
    )
    ai_dev_skills: Mapped[list["RepoAIDevSkill"]] = relationship(
        back_populates="repo", cascade="all, delete-orphan"
    )
    pm_skills: Mapped[list["RepoPMSkill"]] = relationship(
        back_populates="repo", cascade="all, delete-orphan"
    )
    languages: Mapped[list["RepoLanguage"]] = relationship(
        back_populates="repo", cascade="all, delete-orphan"
    )
    commits: Mapped[list["RepoCommit"]] = relationship(
        back_populates="repo", cascade="all, delete-orphan"
    )
    embedding: Mapped["RepoEmbedding | None"] = relationship(
        back_populates="repo", cascade="all, delete-orphan", uselist=False
    )
    taxonomy: Mapped[list["RepoTaxonomy"]] = relationship(
        "RepoTaxonomy", lazy="select", cascade="all, delete-orphan"
    )


class RepoTag(Base):
    __tablename__ = "repo_tags"

    repo_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("repos.id", ondelete="CASCADE"), primary_key=True
    )
    tag: Mapped[str] = mapped_column(Text, primary_key=True)

    repo: Mapped["Repo"] = relationship(back_populates="tags")


class RepoCategory(Base):
    __tablename__ = "repo_categories"

    repo_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("repos.id", ondelete="CASCADE"), primary_key=True
    )
    category_id: Mapped[str] = mapped_column(Text, primary_key=True)
    category_name: Mapped[str] = mapped_column(Text, nullable=False)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")

    repo: Mapped["Repo"] = relationship(back_populates="categories")


class RepoBuilder(Base):
    __tablename__ = "repo_builders"

    repo_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("repos.id", ondelete="CASCADE"), primary_key=True
    )
    login: Mapped[str] = mapped_column(Text, primary_key=True)
    display_name: Mapped[str | None] = mapped_column(Text)
    org_category: Mapped[str | None] = mapped_column(Text)  # big-tech, ai-lab, startup, individual
    is_known_org: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")

    repo: Mapped["Repo"] = relationship(back_populates="builders")


class RepoAIDevSkill(Base):
    __tablename__ = "repo_ai_dev_skills"

    repo_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("repos.id", ondelete="CASCADE"), primary_key=True
    )
    skill: Mapped[str] = mapped_column(Text, primary_key=True)

    repo: Mapped["Repo"] = relationship(back_populates="ai_dev_skills")


class RepoPMSkill(Base):
    __tablename__ = "repo_pm_skills"

    repo_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("repos.id", ondelete="CASCADE"), primary_key=True
    )
    skill: Mapped[str] = mapped_column(Text, primary_key=True)

    repo: Mapped["Repo"] = relationship(back_populates="pm_skills")


class RepoLanguage(Base):
    __tablename__ = "repo_languages"

    repo_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("repos.id", ondelete="CASCADE"), primary_key=True
    )
    language: Mapped[str] = mapped_column(Text, primary_key=True)
    bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    percentage: Mapped[float] = mapped_column(Float, nullable=False)

    repo: Mapped["Repo"] = relationship(back_populates="languages")


class RepoCommit(Base):
    __tablename__ = "repo_commits"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    repo_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("repos.id", ondelete="CASCADE"), nullable=False
    )
    sha: Mapped[str] = mapped_column(Text, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    author: Mapped[str | None] = mapped_column(Text)
    committed_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    url: Mapped[str | None] = mapped_column(Text)

    repo: Mapped["Repo"] = relationship(back_populates="commits")


class RepoEmbedding(Base):
    __tablename__ = "repo_embeddings"

    repo_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("repos.id", ondelete="CASCADE"), primary_key=True
    )
    # Stored as JSON array; pgvector column handled at DB level via migration
    embedding: Mapped[str | None] = mapped_column(Text)
    model: Mapped[str] = mapped_column(Text, nullable=False, default="nomic-embed-text")
    generated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    repo: Mapped["Repo"] = relationship(back_populates="embedding")


class SkillArea(Base):
    __tablename__ = "skill_areas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    lifecycle_group: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text)
    icon: Mapped[str | None] = mapped_column(Text)
    color: Mapped[str | None] = mapped_column(Text)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    min_repos_to_display: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())


class TaxonomyValue(Base):
    __tablename__ = "taxonomy_values"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dimension: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    # embedding_vec handled via raw SQL for pgvector compatibility
    repo_count: Mapped[int] = mapped_column(Integer, default=0)
    trending_score: Mapped[float] = mapped_column(Float, default=0.0)
    first_seen_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
    last_active_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())


class RepoTaxonomy(Base):
    __tablename__ = "repo_taxonomy"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_id: Mapped[_uuid_mod.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("repos.id", ondelete="CASCADE"), nullable=False
    )
    dimension: Mapped[str] = mapped_column(Text, nullable=False)
    raw_value: Mapped[str] = mapped_column(Text, nullable=False)
    taxonomy_value_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("taxonomy_values.id", ondelete="SET NULL")
    )
    similarity_score: Mapped[float | None] = mapped_column(Float)
    assigned_by: Mapped[str] = mapped_column(Text, default="enrichment")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
