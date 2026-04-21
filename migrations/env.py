import asyncio
from logging.config import fileConfig

from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context

# Import settings and models so metadata is populated
from app.config import settings
from app.database import Base
import app.models.repo  # noqa
import app.models.trend  # noqa

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    # transaction_per_migration=True is REQUIRED for op.get_context().autocommit_block()
    # to work. autocommit_block() suspends the current Alembic-managed transaction and
    # runs DDL in AUTOCOMMIT mode — needed for CREATE INDEX CONCURRENTLY (migration 038).
    # Without this flag, autocommit_block() raises AssertionError because there's no
    # Alembic-scoped transaction to suspend.
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        transaction_per_migration=True,
    )
    context.run_migrations()


async def run_async_migrations() -> None:
    # Use engine.connect() (not .begin()) so Alembic owns transaction boundaries via
    # transaction_per_migration=True. engine.begin() would start a wrapping transaction
    # that autocommit_block() can't suspend.
    engine = create_async_engine(settings.database_url)
    async with engine.connect() as conn:
        await conn.run_sync(do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
