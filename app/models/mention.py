from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Integer, String, Text, TIMESTAMP, UniqueConstraint, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.database import Base


class RepoMention(Base):
    __tablename__ = "repo_mentions"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    repo_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("repos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source: Mapped[str] = mapped_column(String, nullable=False)  # "hackernews", "reddit", "youtube"
    external_id: Mapped[str] = mapped_column(String, nullable=False)  # HN story ID
    title: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)  # link to the discussion
    score: Mapped[int | None] = mapped_column(Integer, nullable=True)  # HN points
    comment_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    author: Mapped[str | None] = mapped_column(String, nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("repo_id", "source", "external_id", name="uq_repo_mentions_repo_source_ext"),
    )
