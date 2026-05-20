from datetime import datetime

from sqlalchemy import BigInteger, Boolean, Integer, Numeric, String, Text, TIMESTAMP
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
      - JIRA workflow integration (via admin endpoints)
    """

    __tablename__ = "query_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), index=True
    )

    # Question and answer
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer_truncated: Mapped[str | None] = mapped_column(Text)  # first 500 chars
    answer_full: Mapped[str | None] = mapped_column(Text)

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

    # ── JIRA integration fields (KAN-165) ────────────────────────────────────
    # Set by external automation after creating a JIRA ticket from the ask
    jira_ticket_key: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # "open" | "in_progress" | "done" | "rejected"
    jira_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Freeform note of what was done in response
    action_taken: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Computed spend in cents (input_tokens + output_tokens × model rate)
    cost_cents: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # "positive" | "negative" | "neutral" — optional, written by external automation
    sentiment: Mapped[str | None] = mapped_column(String(16), nullable=True)
