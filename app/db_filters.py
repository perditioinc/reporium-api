"""Centralized DB visibility predicates.

Single source of truth for "what does a public client see?".

Why centralized:
  - Issue #414 (2026-04-23) added `WHERE is_private = false` to repo
    list/detail/search/ask. The fix was correct but every endpoint owned
    its own copy of the predicate, and new endpoints kept being added
    without it (most recently the `/forks`, `/repos/{repo_id}/dependencies`,
    and `/repos/{repo_id}/mentions` paths discovered on 2026-04-28).
  - The 2026-04-28 hotfix audit found a live `/repos/{owner}/{repo}` leak
    of `perditioinc/hippo-harvest-assignment`. Even though the on-disk code
    had the predicate, the per-endpoint copy-paste pattern meant the next
    new handler could (and did) skip it.

Use:
  ORM (SQLAlchemy 2.x):
      from app.db_filters import public_repo_filter
      stmt = select(Repo).where(public_repo_filter())

  Raw SQL (asyncpg / text()):
      from app.db_filters import PUBLIC_REPO_SQL_PREDICATE
      f"SELECT ... FROM repos WHERE {PUBLIC_REPO_SQL_PREDICATE} AND ..."

  When `repos` is aliased (e.g. `r1`/`r2` in JOINs):
      f"... JOIN repos r1 ON r1.id = ... AND r1.is_private = false ..."
  — the constant covers the unaliased case; aliased paths must inline,
  but they should still pivot on the same column name (`is_private = false`)
  so a future audit can grep for stragglers.

The returned ORM expression compares against `False` (not `is None`) so
the predicate excludes rows where `is_private IS NULL` — defensive against
a possible NULL drift if the ingestion path ever stops setting the column.
This matches the production guarantee: the column is `NOT NULL DEFAULT false`
in migration 003 and `is_private` is a required Pydantic field on
`RepoIngestItem` (no default — see app/schemas/repo.py:154).
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.sql import Select
from sqlalchemy.sql.elements import BinaryExpression


# Raw-SQL constant — for `text()` / asyncpg call sites that operate on the
# unaliased `repos` table. Aliased JOINs must still inline (`r1.is_private =
# false`) but should keep the same column name so audits can grep.
PUBLIC_REPO_SQL_PREDICATE: str = "is_private = false"


def public_repo_filter() -> BinaryExpression:
    """Return the SQLAlchemy predicate that hides private repos.

    Use as the first `.where(...)` clause in any ORM query that returns
    repo rows (or rows derived from repos) to a public client.

    Imported lazily so callers don't have to know the model lives in
    app.models.repo, and so this module stays import-cycle-safe.
    """
    from app.models.repo import Repo  # local import to avoid cycle

    # noqa: E712 — SQLAlchemy requires `== False`, not `is False` / `is None`.
    return Repo.is_private == False  # noqa: E712


def public_repos_select() -> Select:
    """Return ``select(Repo)`` already filtered to public (non-private) repos.

    Sugar for ``select(Repo).where(public_repo_filter())`` — keeps the public
    filter inseparable from the SELECT for new code that does a plain repo list.
    """
    from app.models.repo import Repo  # local import to avoid cycle

    return select(Repo).where(public_repo_filter())


def sql_public_filter(repos_alias: str = "r") -> str:
    """Return a SQL fragment to AND into a WHERE / JOIN clause for the given
    ``repos`` table alias.

    Example:
        text(f\"\"\"
            SELECT ... FROM repos r1
            JOIN repos r2 ON r2.id = e2.repo_id AND {sql_public_filter('r2')}
            WHERE {sql_public_filter('r1')} AND ...
        \"\"\")

    The alias is validated as a Python identifier to defend against SQL
    injection through accidental string concatenation.
    """
    if not repos_alias.isidentifier():
        raise ValueError(f"invalid repos alias: {repos_alias!r}")
    return f"{repos_alias}.is_private = false"
