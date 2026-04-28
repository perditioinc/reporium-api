"""Canonicalize ASK source rows to the upstream parent for forks.

The Reporium org `perditioinc` mirrors many upstream repos (microsoft/markitdown,
mendableai/firecrawl, etc.) under its account so the platform can run sync,
embedding, and edge-building without paying upstream API rate limits.

When ASK returns a `sources` array citing those rows, the `name` and `owner`
fields previously echoed `perditioinc/<repo>` even though the row's
`forked_from` column held the canonical upstream `<owner>/<name>`. Users who
asked "Which repos support MCP?" got back `perditioinc/markitdown`,
`perditioinc/firecrawl` instead of `microsoft/markitdown`, `mendableai/firecrawl`
— citing our internal mirror rather than the project they'd actually use.

There was already an LLM-prompt-side canonicalization (intelligence.py
`_build_sources_block`, lines ~1413-1421) but it only affected the prompt
sent to Claude, not the JSON shape returned to the client. This module fixes
the client-facing side.

Behavior:
  - When `forked_from` is set and contains "/", split into upstream
    owner/name and use those.
  - When `forked_from` is null/empty, return the row's own owner/name.
  - The original `forked_from` field is preserved in the response so the
    client can still display "(forked from upstream/repo)" if it wants.
  - `forked_from` strings without "/" (malformed) are treated as no fork —
    fall back to the row's own owner/name so we never silently drop the
    citation and never invent a parent.

Do NOT use this for ingestion / DB writes — it only re-shapes the shape
ASK returns. The DB column stays as-is.
"""

from __future__ import annotations


def canonical_owner_name(
    *,
    forked_from: str | None,
    own_owner: str | None,
    own_name: str | None,
) -> tuple[str | None, str | None]:
    """Return (owner, name) to cite in a /intelligence/ask source.

    Parameters mirror the row columns: `forked_from` is the DB column
    (`<upstream_owner>/<upstream_name>` or NULL), and `own_owner`/`own_name`
    are the row's own `repos.owner` / `repos.name`.

    A malformed `forked_from` (no "/" or empty after split) falls back to
    the row's own owner/name. Never invents data.
    """
    if forked_from and isinstance(forked_from, str) and "/" in forked_from:
        parent_owner, parent_name = forked_from.split("/", 1)
        parent_owner = parent_owner.strip()
        parent_name = parent_name.strip()
        if parent_owner and parent_name:
            return parent_owner, parent_name
    return own_owner, own_name
