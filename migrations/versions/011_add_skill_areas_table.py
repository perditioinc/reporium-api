"""add skill_areas table with seed data

Revision ID: 011
Revises: 010
Create Date: 2026-03-24
"""
from alembic import op

revision = '011'
down_revision = '010'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        CREATE TABLE IF NOT EXISTS skill_areas (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            lifecycle_group TEXT NOT NULL,
            description TEXT,
            icon TEXT,
            color TEXT,
            sort_order INT DEFAULT 0,
            min_repos_to_display INT DEFAULT 1,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_skill_areas_lifecycle_group "
        "ON skill_areas (lifecycle_group)"
    )

    # Foundation & Training
    op.execute(
        "INSERT INTO skill_areas (name, lifecycle_group, sort_order) "
        "VALUES ('Foundation Model Architecture', 'Foundation & Training', 1) "
        "ON CONFLICT (name) DO NOTHING"
    )
    op.execute(
        "INSERT INTO skill_areas (name, lifecycle_group, sort_order) "
        "VALUES ('Fine-tuning & Alignment', 'Foundation & Training', 2) "
        "ON CONFLICT (name) DO NOTHING"
    )
    op.execute(
        "INSERT INTO skill_areas (name, lifecycle_group, sort_order) "
        "VALUES ('Data Engineering', 'Foundation & Training', 3) "
        "ON CONFLICT (name) DO NOTHING"
    )
    op.execute(
        "INSERT INTO skill_areas (name, lifecycle_group, sort_order) "
        "VALUES ('Synthetic Data', 'Foundation & Training', 4) "
        "ON CONFLICT (name) DO NOTHING"
    )

    # Inference & Deployment
    op.execute(
        "INSERT INTO skill_areas (name, lifecycle_group, sort_order) "
        "VALUES ('Inference & Serving', 'Inference & Deployment', 5) "
        "ON CONFLICT (name) DO NOTHING"
    )
    op.execute(
        "INSERT INTO skill_areas (name, lifecycle_group, sort_order) "
        "VALUES ('Model Compression', 'Inference & Deployment', 6) "
        "ON CONFLICT (name) DO NOTHING"
    )
    op.execute(
        "INSERT INTO skill_areas (name, lifecycle_group, sort_order) "
        "VALUES ('Edge AI', 'Inference & Deployment', 7) "
        "ON CONFLICT (name) DO NOTHING"
    )

    # LLM Application Layer
    op.execute(
        "INSERT INTO skill_areas (name, lifecycle_group, sort_order) "
        "VALUES ('Agents & Orchestration', 'LLM Application Layer', 8) "
        "ON CONFLICT (name) DO NOTHING"
    )
    op.execute(
        "INSERT INTO skill_areas (name, lifecycle_group, sort_order) "
        "VALUES ('RAG & Retrieval', 'LLM Application Layer', 9) "
        "ON CONFLICT (name) DO NOTHING"
    )
    op.execute(
        "INSERT INTO skill_areas (name, lifecycle_group, sort_order) "
        "VALUES ('Context Engineering', 'LLM Application Layer', 10) "
        "ON CONFLICT (name) DO NOTHING"
    )
    op.execute(
        "INSERT INTO skill_areas (name, lifecycle_group, sort_order) "
        "VALUES ('Tool Use', 'LLM Application Layer', 11) "
        "ON CONFLICT (name) DO NOTHING"
    )
    op.execute(
        "INSERT INTO skill_areas (name, lifecycle_group, sort_order) "
        "VALUES ('Structured Output', 'LLM Application Layer', 12) "
        "ON CONFLICT (name) DO NOTHING"
    )
    op.execute(
        "INSERT INTO skill_areas (name, lifecycle_group, sort_order) "
        "VALUES ('Prompt Engineering', 'LLM Application Layer', 13) "
        "ON CONFLICT (name) DO NOTHING"
    )
    op.execute(
        "INSERT INTO skill_areas (name, lifecycle_group, sort_order) "
        "VALUES ('Knowledge Graphs', 'LLM Application Layer', 14) "
        "ON CONFLICT (name) DO NOTHING"
    )

    # Eval / Safety / Ops
    op.execute(
        "INSERT INTO skill_areas (name, lifecycle_group, sort_order) "
        "VALUES ('Evaluation', 'Eval / Safety / Ops', 15) "
        "ON CONFLICT (name) DO NOTHING"
    )
    op.execute(
        "INSERT INTO skill_areas (name, lifecycle_group, sort_order) "
        "VALUES ('Security & Guardrails', 'Eval / Safety / Ops', 16) "
        "ON CONFLICT (name) DO NOTHING"
    )
    op.execute(
        "INSERT INTO skill_areas (name, lifecycle_group, sort_order) "
        "VALUES ('Observability', 'Eval / Safety / Ops', 17) "
        "ON CONFLICT (name) DO NOTHING"
    )
    op.execute(
        "INSERT INTO skill_areas (name, lifecycle_group, sort_order) "
        "VALUES ('MLOps', 'Eval / Safety / Ops', 18) "
        "ON CONFLICT (name) DO NOTHING"
    )
    op.execute(
        "INSERT INTO skill_areas (name, lifecycle_group, sort_order) "
        "VALUES ('AI Governance', 'Eval / Safety / Ops', 19) "
        "ON CONFLICT (name) DO NOTHING"
    )

    # Modality-Specific
    op.execute(
        "INSERT INTO skill_areas (name, lifecycle_group, sort_order) "
        "VALUES ('Computer Vision', 'Modality-Specific', 20) "
        "ON CONFLICT (name) DO NOTHING"
    )
    op.execute(
        "INSERT INTO skill_areas (name, lifecycle_group, sort_order) "
        "VALUES ('Speech & Audio', 'Modality-Specific', 21) "
        "ON CONFLICT (name) DO NOTHING"
    )
    op.execute(
        "INSERT INTO skill_areas (name, lifecycle_group, sort_order) "
        "VALUES ('Generative Media', 'Modality-Specific', 22) "
        "ON CONFLICT (name) DO NOTHING"
    )
    op.execute(
        "INSERT INTO skill_areas (name, lifecycle_group, sort_order) "
        "VALUES ('NLP', 'Modality-Specific', 23) "
        "ON CONFLICT (name) DO NOTHING"
    )
    op.execute(
        "INSERT INTO skill_areas (name, lifecycle_group, sort_order) "
        "VALUES ('Multimodal', 'Modality-Specific', 24) "
        "ON CONFLICT (name) DO NOTHING"
    )

    # Applied AI
    op.execute(
        "INSERT INTO skill_areas (name, lifecycle_group, sort_order) "
        "VALUES ('Coding Assistants', 'Applied AI', 25) "
        "ON CONFLICT (name) DO NOTHING"
    )
    op.execute(
        "INSERT INTO skill_areas (name, lifecycle_group, sort_order) "
        "VALUES ('Robotics', 'Applied AI', 26) "
        "ON CONFLICT (name) DO NOTHING"
    )
    op.execute(
        "INSERT INTO skill_areas (name, lifecycle_group, sort_order) "
        "VALUES ('AI for Science', 'Applied AI', 27) "
        "ON CONFLICT (name) DO NOTHING"
    )
    op.execute(
        "INSERT INTO skill_areas (name, lifecycle_group, sort_order) "
        "VALUES ('Recommendation Systems', 'Applied AI', 28) "
        "ON CONFLICT (name) DO NOTHING"
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS ix_skill_areas_lifecycle_group")
    op.execute("DROP TABLE IF EXISTS skill_areas")
