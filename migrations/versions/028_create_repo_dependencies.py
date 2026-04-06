"""Create repo_dependencies table for SBOM-based dependency tracking.

Revision ID: 028
Revises: 027
"""

import sqlalchemy as sa
from alembic import op

revision = "028"
down_revision = "027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "repo_dependencies",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "repo_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("repos.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("package_name", sa.Text, nullable=False),
        sa.Column("package_ecosystem", sa.String, nullable=True),
        sa.Column("version_constraint", sa.String, nullable=True),
        sa.Column("is_direct", sa.Boolean, nullable=False, server_default="true"),
        sa.Column(
            "fetched_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_index("idx_repo_dependencies_repo_id", "repo_dependencies", ["repo_id"])
    op.create_index("idx_repo_dependencies_package_name", "repo_dependencies", ["package_name"])
    op.create_unique_constraint(
        "uq_repo_dep_repo_pkg_eco",
        "repo_dependencies",
        ["repo_id", "package_name", "package_ecosystem"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_repo_dep_repo_pkg_eco", "repo_dependencies", type_="unique")
    op.drop_index("idx_repo_dependencies_package_name", table_name="repo_dependencies")
    op.drop_index("idx_repo_dependencies_repo_id", table_name="repo_dependencies")
    op.drop_table("repo_dependencies")
