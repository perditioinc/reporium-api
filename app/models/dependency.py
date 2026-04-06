from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Boolean, ForeignKey, String, Text, TIMESTAMP, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.database import Base


class RepoDependency(Base):
    __tablename__ = "repo_dependencies"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    repo_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("repos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    package_name: Mapped[str] = mapped_column(Text, nullable=False)
    package_ecosystem: Mapped[str | None] = mapped_column(String, nullable=True)
    version_constraint: Mapped[str | None] = mapped_column(String, nullable=True)
    is_direct: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    fetched_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("repo_id", "package_name", "package_ecosystem", name="uq_repo_dep_repo_pkg_eco"),
    )
