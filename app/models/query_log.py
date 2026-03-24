from datetime import datetime

from sqlalchemy import BigInteger, Boolean, Integer, Numeric, Text, TIMESTAMP
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.database import Base


class QueryLog(Base):
    """One row per /intelligence/ask (or /query) call.

    Used for:
      - Cost tracking  (tokens_prompt + tokens_completion → cost_usd)
      - Semantic caching (question similarity search)
      - Abuse detection (hashed_ip anomaly detection)
    """

    __tablename__ = "query_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), index=True
    )

    # Question and answer
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer_truncated: Mapped[str | None] = mapped_column(Text)  # first 500 chars

    # Sources returned [{name: "owner/repo", score: 0.88}]
    sources: Mapped[dict | None] = mapped_column(JSONB)

    # Token usage and cost
    tokens_prompt: Mapped[int | None] = mapped_column(Integer)
    tokens_completion: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[float | None] = mapped_column(Numeric(10, 6))

    # Privacy-safe caller identity
    hashed_ip: Mapped[str | None] = mapped_column(Text, index=True)  # SHA-256 hex

    # Performance
    latency_ms: Mapped[int | None] = mapped_column(Integer)

    # Model metadata
    model: Mapped[str | None] = mapped_column(Text)
    cache_hit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
