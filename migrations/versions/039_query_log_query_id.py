"""Add query_id UUID handle to query_log for thumbs feedback (PR3 of Ask UX).

Adds a stable client-facing identifier so the frontend can post a sentiment
update for a specific answered query. The existing BIGINT primary key is
unsuitable because:

  1. The /intelligence/ask/stream INSERT is fire-and-forget — the response is
     yielded to the client BEFORE the row is committed, so the PK isn't yet
     known when we need to send it.
  2. Exposing the monotonic PK to clients leaks query volume metadata.

A UUID is generated at the start of every ask request, emitted in the
streamed `done` event, and written into this column. The new feedback
endpoint then updates the `sentiment` column WHERE query_id = ?.

Changes:
  1. ADD COLUMN query_id UUID NULL  — populated for new rows; older rows
     (pre-this migration) stay NULL and aren't addressable by feedback.
  2. CREATE UNIQUE INDEX ... WHERE query_id IS NOT NULL  — sparse, so the
     historical NULL rows don't block the unique constraint.

Revision ID: 039
Revises: 038
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "039"
down_revision = "038"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("query_log", sa.Column("query_id", postgresql.UUID(as_uuid=False), nullable=True))
    # Sparse unique index so old NULL rows don't conflict, but every newly
    # written query_id is guaranteed unique (defends against accidental reuse
    # if a client retries with the same UUID).
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_query_log_query_id
        ON query_log (query_id)
        WHERE query_id IS NOT NULL
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_query_log_query_id")
    op.drop_column("query_log", "query_id")
