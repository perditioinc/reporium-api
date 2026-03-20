from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Integer, Text, TIMESTAMP, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.database import Base


class TrendSnapshot(Base):
    __tablename__ = "trend_snapshots"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    snapshotted_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    tag: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str | None] = mapped_column(Text)
    repo_count: Mapped[int] = mapped_column(Integer, nullable=False)
    commit_count_7d: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")


class GapAnalysis(Base):
    __tablename__ = "gap_analysis"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    generated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    skill: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(Text, nullable=False)  # missing, weak, moderate, strong
    repo_count: Mapped[int] = mapped_column(Integer, nullable=False)
    why: Mapped[str | None] = mapped_column(Text)
    trend: Mapped[str | None] = mapped_column(Text)
    essential_repos: Mapped[dict | None] = mapped_column(JSONB)


class IngestionLog(Base):
    __tablename__ = "ingestion_log"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    started_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    mode: Mapped[str] = mapped_column(Text, nullable=False)  # quick, weekly, full
    repos_fetched: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    repos_updated: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    api_calls_made: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    errors: Mapped[list | None] = mapped_column(JSONB, default=list, server_default=text("'[]'"))
    status: Mapped[str] = mapped_column(Text, nullable=False, default="running")  # running, success, failed
