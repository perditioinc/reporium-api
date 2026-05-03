"""
Unit tests for library_full.py helper functions.

Covers:
- _build_ai_dev_skill_stats: skill group mapping from enrichedTags
- _build_builder_stats: known-org category overrides + sort order
- _build_tag_metrics: system tag filtering
- sanitize_repo: upstreamCreatedAt fallback safety
"""

from app.routers.library_full import (
    KNOWN_ORG_CATEGORIES,
    LIFECYCLE_GROUPS,
    SYSTEM_TAGS,
    _AI_DEV_SKILLS_ORDERED,
    _SKILL_TAG_TO_GROUP,
    _build_ai_dev_skill_stats,
    _build_builder_stats,
    _build_enriched_repo,
    _build_tag_metrics,
    sanitize_repo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_repo(name: str, tags: list[str], ai_skills: list[str] = None,
               is_fork: bool = False, stars: int = 0,
               builders: list[dict] = None) -> dict:
    return {
        "name": name,
        "isFork": is_fork,
        "stars": stars,
        "language": "Python",
        "lastUpdated": "2024-01-01T00:00:00",
        "enrichedTags": tags,
        "aiDevSkills": ai_skills or [],
        "builders": builders or [],
        "allCategories": [],
        "languageBreakdown": {},
        "languagePercentages": {},
    }


# ---------------------------------------------------------------------------
# _build_ai_dev_skill_stats
# ---------------------------------------------------------------------------

class TestBuildAiDevSkillStats:

    def test_vllm_maps_to_inference_serving(self):
        repos = [_make_repo("r1", ["vLLM", "Python"])]
        stats = {s["skill"]: s for s in _build_ai_dev_skill_stats(repos)}
        assert "Inference & Serving" in stats
        assert stats["Inference & Serving"]["repoCount"] == 1

    def test_langchain_maps_to_agents_orchestration(self):
        repos = [_make_repo("r1", ["LangChain"])]
        stats = {s["skill"]: s for s in _build_ai_dev_skill_stats(repos)}
        assert stats["Agents & Orchestration"]["repoCount"] == 1

    def test_multiple_tags_same_group_counted_once_per_repo(self):
        # A repo with both vLLM and SGLang should count as 1 for Inference & Serving
        repos = [_make_repo("r1", ["vLLM", "SGLang", "TensorRT"])]
        stats = {s["skill"]: s for s in _build_ai_dev_skill_stats(repos)}
        assert stats["Inference & Serving"]["repoCount"] == 1

    def test_repo_counted_in_multiple_groups(self):
        repos = [_make_repo("r1", ["vLLM", "RAG"])]
        stats = {s["skill"]: s for s in _build_ai_dev_skill_stats(repos)}
        assert stats["Inference & Serving"]["repoCount"] == 1
        assert stats["RAG & Retrieval"]["repoCount"] == 1

    def test_unknown_tags_produce_zero_counts(self):
        repos = [_make_repo("r1", ["unknown-tag-xyz"])]
        stats = {s["skill"]: s for s in _build_ai_dev_skill_stats(repos)}
        # Every skill exists in output (one per taxonomy skill — 28 total)
        assert len(stats) == len(_AI_DEV_SKILLS_ORDERED)
        for s in stats.values():
            assert s["repoCount"] == 0

    def test_empty_repos_returns_all_skills_with_zero(self):
        stats = _build_ai_dev_skill_stats([])
        assert len(stats) == len(_AI_DEV_SKILLS_ORDERED)
        assert all(s["repoCount"] == 0 for s in stats)

    def test_coverage_field_strong_when_over_10_percent(self):
        # 1 out of 5 repos = 20% → "strong"
        repos = [_make_repo(f"r{i}", [] if i > 0 else ["vLLM"]) for i in range(5)]
        stats = {s["skill"]: s for s in _build_ai_dev_skill_stats(repos)}
        assert stats["Inference & Serving"]["coverage"] == "strong"

    def test_coverage_field_none_when_zero(self):
        repos = [_make_repo("r1", ["some-random-tag"])]
        stats = {s["skill"]: s for s in _build_ai_dev_skill_stats(repos)}
        assert stats["Inference & Serving"]["coverage"] == "none"

    def test_canonical_skill_in_ai_dev_skills_field_counted(self):
        # Canonical 28-skill names in aiDevSkills should be counted directly
        repos = [_make_repo("r1", [], ai_skills=["Inference & Serving"])]
        stats = {s["skill"]: s for s in _build_ai_dev_skill_stats(repos)}
        assert stats["Inference & Serving"]["repoCount"] == 1

    def test_legacy_tag_in_ai_dev_skills_field_not_counted_directly(self):
        # Legacy tool names (e.g. "vLLM") in aiDevSkills are not canonical skill names;
        # they only match via enrichedTags fallback path.
        repos = [_make_repo("r1", [], ai_skills=["vLLM"])]
        stats = {s["skill"]: s for s in _build_ai_dev_skill_stats(repos)}
        # "vLLM" is not a canonical skill name, so it won't directly match
        assert stats["Inference & Serving"]["repoCount"] == 0

    def test_case_insensitive_tag_lookup(self):
        repos = [_make_repo("r1", ["VLLM"])]
        # _SKILL_TAG_TO_GROUP is lowercase — "vllm" matches "vLLM"
        # The tags come in as stored; test that mixed-case still works
        stats = {s["skill"]: s for s in _build_ai_dev_skill_stats(repos)}
        # "VLLM" lowercased = "vllm" which should match "vLLM" in the taxonomy
        assert stats["Inference & Serving"]["repoCount"] == 1

    def test_output_order_matches_taxonomy_order(self):
        stats = _build_ai_dev_skill_stats([])
        skill_names = [s["skill"] for s in stats]
        assert skill_names == _AI_DEV_SKILLS_ORDERED

    def test_lifecycle_group_present_in_each_stat(self):
        stats = _build_ai_dev_skill_stats([])
        for s in stats:
            assert "lifecycleGroup" in s
            assert s["lifecycleGroup"] in LIFECYCLE_GROUPS.values(), (
                f"skill '{s['skill']}' has unknown lifecycleGroup '{s['lifecycleGroup']}'"
            )

    def test_top_repos_sorted_by_stars(self):
        repos = [
            _make_repo("low", ["vLLM"], stars=10),
            _make_repo("high", ["vLLM"], stars=9999),
        ]
        stats = {s["skill"]: s for s in _build_ai_dev_skill_stats(repos)}
        top = stats["Inference & Serving"]["topRepos"]
        assert top[0] == "high"


# ---------------------------------------------------------------------------
# _build_builder_stats
# ---------------------------------------------------------------------------

class TestBuildBuilderStats:

    def test_anthropics_overridden_to_ai_lab(self):
        repos = [_make_repo("r1", [], builders=[{
            "login": "anthropics", "orgCategory": "individual", "avatarUrl": ""
        }])]
        stats = {s["login"]: s for s in _build_builder_stats(repos)}
        assert stats["anthropics"]["category"] == "ai-lab"
        assert stats["anthropics"]["displayName"] == "Anthropic"

    def test_facebookresearch_overridden_to_ai_lab(self):
        repos = [_make_repo("r1", [], builders=[{
            "login": "facebookresearch", "orgCategory": "individual", "avatarUrl": ""
        }])]
        stats = {s["login"]: s for s in _build_builder_stats(repos)}
        assert stats["facebookresearch"]["category"] == "ai-lab"
        assert stats["facebookresearch"]["displayName"] == "Meta Research"

    def test_huggingface_overridden_to_ai_lab(self):
        repos = [_make_repo("r1", [], builders=[{
            "login": "huggingface", "orgCategory": "individual", "avatarUrl": ""
        }])]
        stats = {s["login"]: s for s in _build_builder_stats(repos)}
        assert stats["huggingface"]["category"] == "ai-lab"

    def test_langchain_ai_overridden_to_startup(self):
        repos = [_make_repo("r1", [], builders=[{
            "login": "langchain-ai", "orgCategory": "individual", "avatarUrl": ""
        }])]
        stats = {s["login"]: s for s in _build_builder_stats(repos)}
        assert stats["langchain-ai"]["category"] == "startup"
        assert stats["langchain-ai"]["displayName"] == "LangChain"

    def test_unknown_org_keeps_db_category(self):
        repos = [_make_repo("r1", [], builders=[{
            "login": "some-random-user", "orgCategory": "research", "avatarUrl": ""
        }])]
        stats = {s["login"]: s for s in _build_builder_stats(repos)}
        assert stats["some-random-user"]["category"] == "research"

    def test_unknown_org_defaults_to_individual(self):
        repos = [_make_repo("r1", [], builders=[{
            "login": "random-person", "orgCategory": None, "avatarUrl": ""
        }])]
        stats = {s["login"]: s for s in _build_builder_stats(repos)}
        assert stats["random-person"]["category"] == "individual"

    def test_sorted_by_repo_count_descending(self):
        repos = (
            [_make_repo(f"r{i}", [], builders=[{"login": "small-org", "orgCategory": "startup", "avatarUrl": ""}])
             for i in range(2)] +
            [_make_repo(f"b{i}", [], builders=[{"login": "big-org", "orgCategory": "big-tech", "avatarUrl": ""}])
             for i in range(5)]
        )
        stats = _build_builder_stats(repos)
        assert stats[0]["login"] == "big-org"
        assert stats[1]["login"] == "small-org"

    def test_repo_count_aggregated_correctly(self):
        repos = [
            _make_repo("r1", [], builders=[{"login": "openai", "orgCategory": "ai-lab", "avatarUrl": ""}]),
            _make_repo("r2", [], builders=[{"login": "openai", "orgCategory": "ai-lab", "avatarUrl": ""}]),
        ]
        stats = {s["login"]: s for s in _build_builder_stats(repos)}
        assert stats["openai"]["repoCount"] == 2

    def test_returns_at_most_50(self):
        repos = [
            _make_repo(f"r{i}", [], builders=[{"login": f"user{i}", "orgCategory": None, "avatarUrl": ""}])
            for i in range(100)
        ]
        stats = _build_builder_stats(repos)
        assert len(stats) <= 50

    def test_all_known_orgs_in_mapping_have_valid_category(self):
        valid_categories = {"big-tech", "ai-lab", "startup", "research", "individual"}
        for login, (cat, _) in KNOWN_ORG_CATEGORIES.items():
            assert cat in valid_categories, f"{login} has invalid category: {cat}"


# ---------------------------------------------------------------------------
# _build_tag_metrics
# ---------------------------------------------------------------------------

class TestBuildTagMetrics:

    def test_active_tag_excluded(self):
        repos = [_make_repo("r1", ["Active", "Python"])]
        metrics = {m["tag"]: m for m in _build_tag_metrics(repos)}
        assert "Active" not in metrics
        assert "Python" in metrics

    def test_forked_tag_excluded(self):
        repos = [_make_repo("r1", ["Forked", "LangChain"])]
        metrics = {m["tag"]: m for m in _build_tag_metrics(repos)}
        assert "Forked" not in metrics

    def test_built_by_me_excluded(self):
        repos = [_make_repo("r1", ["Built by Me", "RAG"])]
        metrics = {m["tag"]: m for m in _build_tag_metrics(repos)}
        assert "Built by Me" not in metrics

    def test_all_system_tags_excluded(self):
        repos = [_make_repo("r1", list(SYSTEM_TAGS) + ["real-tag"])]
        metrics = {m["tag"]: m for m in _build_tag_metrics(repos)}
        for st in SYSTEM_TAGS:
            assert st not in metrics, f"System tag '{st}' should be excluded"
        assert "real-tag" in metrics

    def test_real_tags_counted_correctly(self):
        repos = [
            _make_repo("r1", ["vLLM", "Python"]),
            _make_repo("r2", ["vLLM", "Rust"]),
        ]
        metrics = {m["tag"]: m for m in _build_tag_metrics(repos)}
        assert metrics["vLLM"]["repoCount"] == 2
        assert metrics["Python"]["repoCount"] == 1

    def test_empty_repos_returns_empty(self):
        assert _build_tag_metrics([]) == []

    def test_repo_with_only_system_tags_contributes_nothing(self):
        repos = [_make_repo("r1", ["Active", "Forked", "Built by Me"])]
        assert _build_tag_metrics(repos) == []

    def test_kan_193_no_per_tag_repos_array(self):
        """KAN-193: tagMetric entries no longer carry a per-tag `repos[]` array.

        That array dominated the 3.8 MB /library/aggregates payload and was
        dropped after a consumer audit found no production reader for it.
        Both /library/full and /library/aggregates inherit this trim because
        they share build_tag_metrics().
        """
        repos = [
            _make_repo("r1", ["RAG", "Python"]),
            _make_repo("r2", ["RAG"]),
        ]
        metrics = _build_tag_metrics(repos)
        assert metrics, "expected at least one tagMetric for non-system tags"
        for m in metrics:
            assert "repos" not in m, (
                "KAN-193 regression: build_tag_metrics emitted a per-tag "
                "`repos` array. That field was dropped intentionally."
            )


# ---------------------------------------------------------------------------
# sanitize_repo — upstreamCreatedAt fallback fix
# ---------------------------------------------------------------------------

class TestSanitizeRepoDateFallback:

    def test_upstream_created_at_not_set_from_ingested_at(self):
        """upstreamCreatedAt must NOT be populated from createdAt/ingested_at fallback.
        Showing the ingestion date as 'Project created' is misleading."""
        repo = {
            "name": "test-fork",
            "isFork": True,
            "upstreamCreatedAt": "",       # empty — no real data yet
            "createdAt": "2026-03-20T00:00:00",  # this is ingested_at, not upstream creation
            "lastUpdated": "2026-03-20T00:00:00",
            "enrichedTags": [],
        }
        result = sanitize_repo(repo)
        # Must stay empty — should NOT be set to the ingestion date
        assert result.get("upstreamCreatedAt") == ""

    def test_upstream_created_at_preserved_when_real_value_present(self):
        real_date = "2020-06-15T00:00:00"
        repo = {
            "name": "test-fork",
            "isFork": True,
            "upstreamCreatedAt": real_date,
            "createdAt": "2026-03-20T00:00:00",
            "lastUpdated": "2026-03-20T00:00:00",
            "enrichedTags": [],
        }
        result = sanitize_repo(repo)
        assert result["upstreamCreatedAt"] == real_date

    def test_upstream_last_push_at_falls_back_to_last_updated_for_forks(self):
        """upstreamLastPushAt CAN fall back to lastUpdated — that's a safe proxy."""
        repo = {
            "name": "test-fork",
            "isFork": True,
            "upstreamLastPushAt": "",
            "lastUpdated": "2024-06-01T00:00:00",
            "enrichedTags": [],
        }
        result = sanitize_repo(repo)
        assert result["upstreamLastPushAt"] == "2024-06-01T00:00:00"


# ---------------------------------------------------------------------------
# _build_enriched_repo — stars/forks for fork vs built repos  (issue #13)
# ---------------------------------------------------------------------------

def _make_db_repo(**kwargs) -> dict:
    """Minimal DB row dict for _build_enriched_repo."""
    defaults = {
        "id": "00000000-0000-0000-0000-000000000001",
        "name": "test-repo",
        "owner": "perditioinc",
        "description": "A test repo",
        "is_fork": False,
        "forked_from": None,
        "primary_language": "Python",
        "github_url": "https://github.com/perditioinc/test-repo",
        "fork_sync_state": None,
        "behind_by": 0,
        "ahead_by": 0,
        "upstream_created_at": None,
        "forked_at": None,
        "your_last_push_at": None,
        "upstream_last_push_at": None,
        "parent_stars": None,
        "parent_forks": None,
        "parent_is_archived": False,
        "stargazers_count": None,
        "open_issues_count": 0,
        "commits_last_7_days": 0,
        "commits_last_30_days": 0,
        "commits_last_90_days": 0,
        "readme_summary": None,
        "activity_score": 0,
        "ingested_at": None,
        "updated_at": None,
        "github_updated_at": None,
    }
    defaults.update(kwargs)
    return defaults


class TestBuildEnrichedRepoStars:

    def test_fork_uses_parent_stars(self):
        """Fork repos must show the upstream repo's star count."""
        repo = _make_db_repo(
            is_fork=True,
            forked_from="openai/openai-cookbook",
            parent_stars=45000,
            parent_forks=7000,
            stargazers_count=3,
        )
        enriched = _build_enriched_repo(repo, [], [], [], [], [])
        assert enriched["stars"] == 45000
        assert enriched["forks"] == 7000

    def test_built_repo_uses_own_stargazers_count(self):
        """Non-fork (built) repos must show their own star count, not parent_stars."""
        repo = _make_db_repo(
            is_fork=False,
            forked_from=None,
            parent_stars=None,
            parent_forks=None,
            stargazers_count=42,
        )
        enriched = _build_enriched_repo(repo, [], [], [], [], [])
        assert enriched["stars"] == 42

    def test_built_repo_with_null_stargazers_count_shows_zero(self):
        """Built repo with no star data must show 0, not None."""
        repo = _make_db_repo(
            is_fork=False,
            forked_from=None,
            parent_stars=None,
            parent_forks=None,
            stargazers_count=None,
        )
        enriched = _build_enriched_repo(repo, [], [], [], [], [])
        assert enriched["stars"] == 0

    def test_fork_with_null_parent_stars_uses_none_not_own_stars(self):
        """Fork repos should not fall back to their own stargazers_count."""
        repo = _make_db_repo(
            is_fork=True,
            forked_from="some-org/some-repo",
            parent_stars=None,
            stargazers_count=99,
        )
        enriched = _build_enriched_repo(repo, [], [], [], [], [])
        # parent_stars is None — stays None (frontend renders parentStats.stars)
        assert enriched["stars"] is None

    def test_built_repo_forks_always_zero(self):
        """Built repos show 0 for forks (we don't track how many times our own repos are forked)."""
        repo = _make_db_repo(
            is_fork=False,
            parent_forks=None,
            stargazers_count=10,
        )
        enriched = _build_enriched_repo(repo, [], [], [], [], [])
        assert enriched["forks"] == 0

    def test_open_issues_count_round_trips_from_repo_row(self):
        """Repo open issue counts should be exposed on the frontend contract."""
        repo = _make_db_repo(
            is_fork=False,
            stargazers_count=10,
            open_issues_count=17,
        )
        enriched = _build_enriched_repo(repo, [], [], [], [], [])
        assert enriched["openIssuesCount"] == 17


# ---------------------------------------------------------------------------
# /library/full privacy integration — 2026-04-23 leak regression guard
# (contract updated 2026-04-28 to coordinate with Lane 2 / Lane 4).
# ---------------------------------------------------------------------------
#
# On 2026-04-23 at 05:03:48 UTC, 44 private perditioinc/* repos surfaced in
# public/data/library.json and were served on reporium.com for ~40 minutes.
# The /library/full SQL already had `WHERE is_private = false`, but the DB
# column was stale for the 44 repos (ingestion defaults is_private=False when
# the field is missing, and sync_is_private.py had not run).
#
# Then on 2026-04-27, perditioinc/hippo-harvest-assignment leaked again
# because no privacy field was emitted on the wire — downstream gates
# (frontend `validate:privacy`, reporium-audit `check_contract`) had
# nothing to assert against, so they silently passed for ~22 hours.
#
# Updated contract enforced by these tests:
#   1. No private repo ever appears in any paginated /library/full page.
#   2. Every repo response object MUST carry `isPrivate` (camelCase) AND
#      it MUST be `False`. This is the inverse of the prior (#414) shape —
#      the field is now PRESENT-AND-FALSE so downstream gates have a
#      structural signal to validate. A missing field is a build-blocking
#      failure for Lane 2's `validate-privacy.ts` and Lane 4's
#      `check_contract` audit.
#   3. Ingesting with is_private=true keeps the repo out of the public feed
#      across all pages (covers the pagination-edge case — the incident
#      surfaced on page 10 of the live response).

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_library_full_excludes_private_repos_across_all_pages(client: AsyncClient):
    """Regression test for the 2026-04-23 + 2026-04-27 private-repo leaks.

    Ingests a mix of public + private repos and verifies every page of
    /library/full:
      - omits private repos entirely (Guard 1),
      - emits ``isPrivate: False`` on every surviving repo (Guard 2 — the
        post-2026-04-28 contract that gives Lane 2's validate-privacy and
        Lane 4's audit a structural signal to assert against).
    """
    from tests.conftest import AUTH_HEADERS, TEST_REPO_FIXTURE

    public_payload = {
        **TEST_REPO_FIXTURE,
        "name": "library-public-probe",
        "github_url": "https://github.com/testuser/library-public-probe",
        "is_private": False,
    }
    private_payload = {
        **TEST_REPO_FIXTURE,
        "name": "library-private-probe",
        "github_url": "https://github.com/testuser/library-private-probe",
        "is_private": True,
    }

    r = await client.post(
        "/ingest/repos",
        json=[public_payload, private_payload],
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 200

    # Walk every page at the smallest possible page size to exercise pagination.
    page = 1
    seen_names: set[str] = set()
    while True:
        resp = await client.get(f"/library/full?page={page}&page_size=1")
        assert resp.status_code == 200, f"page {page} returned {resp.status_code}"
        body = resp.json()
        repos = body.get("repos", [])

        for repo in repos:
            # Guard 2: isPrivate must be PRESENT and FALSE on every surviving
            # repo. Missing field would block Lane 2's prebuild validator and
            # FAIL Lane 4's nightly audit.
            assert "isPrivate" in repo, (
                f"missing isPrivate on page {page} for {repo.get('name')!r} — "
                "Lane 2 validate-privacy and Lane 4 audit need this field"
            )
            assert repo["isPrivate"] is False, (
                f"PRIVACY LEAK: isPrivate={repo['isPrivate']!r} on page "
                f"{page} for {repo.get('name')!r} — only public repos "
                "should reach the wire"
            )
            seen_names.add(repo.get("name"))

        total_pages = body.get("totalPages", 1)
        if page >= total_pages or not repos:
            break
        page += 1

    # Guard 1: the private repo must never appear in any paginated page
    assert "library-public-probe" in seen_names, (
        "public probe repo should be in /library/full"
    )
    assert "library-private-probe" not in seen_names, (
        "PRIVACY LEAK: private-probe repo appeared in /library/full — "
        "this reproduces the 2026-04-23 incident"
    )


@pytest.mark.asyncio
async def test_library_full_total_count_excludes_private(client: AsyncClient):
    """/library/full stats.total and totalRepos must exclude private repos.

    Asserts by delta against a baseline so this stays green regardless of what
    prior tests in the run have already seeded into the shared DB.
    """
    from tests.conftest import AUTH_HEADERS, TEST_REPO_FIXTURE

    baseline = (await client.get("/library/full?page=1&page_size=1")).json()
    baseline_total = baseline["totalRepos"]
    baseline_stats_total = baseline["stats"]["total"]

    payloads = [
        {**TEST_REPO_FIXTURE, "name": "count-public-1",
         "github_url": "https://github.com/testuser/count-public-1", "is_private": False},
        {**TEST_REPO_FIXTURE, "name": "count-public-2",
         "github_url": "https://github.com/testuser/count-public-2", "is_private": False},
        {**TEST_REPO_FIXTURE, "name": "count-private-1",
         "github_url": "https://github.com/testuser/count-private-1", "is_private": True},
        {**TEST_REPO_FIXTURE, "name": "count-private-2",
         "github_url": "https://github.com/testuser/count-private-2", "is_private": True},
    ]
    r = await client.post("/ingest/repos", json=payloads, headers=AUTH_HEADERS)
    assert r.status_code == 200

    resp = await client.get("/library/full?page=1&page_size=500")
    assert resp.status_code == 200
    body = resp.json()

    # Only the 2 public repos must move the needle. The 2 private ones are invisible.
    assert body["totalRepos"] == baseline_total + 2, (
        f"totalRepos should only count public repos, "
        f"got {baseline_total} -> {body['totalRepos']} (expected +2)"
    )
    assert body["stats"]["total"] == baseline_stats_total + 2, (
        f"stats.total should only count public repos, "
        f"got {baseline_stats_total} -> {body['stats']['total']} (expected +2)"
    )


# ---------------------------------------------------------------------------
# /library/full schema contract — `isPrivate` field guarantee
# ---------------------------------------------------------------------------
#
# Lane 2 (`scripts/validate-privacy.ts`) and Lane 4 (`reporium_audit/checks/
# contract.py::check_contract`) both require that every repo on the wire
# carry a privacy field. Missing field is a build/audit failure on those
# downstream gates. These tests pin the field-emission contract here so a
# future API change that drops the field is caught on the API side
# instead of cascading into the frontend build and the nightly audit.


@pytest.mark.asyncio
async def test_library_full_response_repo_carries_isprivate_field(
    client: AsyncClient,
):
    """Every repo on /library/full carries ``isPrivate`` (camelCase, bool).

    Camel-case matches the existing wire format (``isFork``, ``forkedFrom``,
    etc.). Lane 2's `validate-privacy.ts` and Lane 4's contract check both
    accept this naming.
    """
    from tests.conftest import AUTH_HEADERS, TEST_REPO_FIXTURE

    payload = {
        **TEST_REPO_FIXTURE,
        "name": "schema-public-probe",
        "github_url": "https://github.com/testuser/schema-public-probe",
        "is_private": False,
    }
    r = await client.post(
        "/ingest/repos", json=[payload], headers=AUTH_HEADERS
    )
    assert r.status_code == 200

    resp = await client.get("/library/full?page=1&page_size=500")
    assert resp.status_code == 200
    body = resp.json()
    assert body["repos"], "library/full should return at least one repo"

    for repo in body["repos"]:
        assert "isPrivate" in repo, (
            f"missing isPrivate on {repo.get('name')!r} — Lane 2's "
            "validate-privacy.ts will treat this as a build-blocking failure"
        )
        assert isinstance(repo["isPrivate"], bool), (
            f"isPrivate must be a bool on {repo.get('name')!r}, got "
            f"{type(repo['isPrivate']).__name__}"
        )
        # Public-endpoint contract: the only valid serialized value is False.
        assert repo["isPrivate"] is False, (
            f"PRIVACY LEAK: {repo.get('name')!r} surfaced with isPrivate=True"
        )


@pytest.mark.asyncio
async def test_library_full_isprivate_field_uses_camelcase_only(
    client: AsyncClient,
):
    """The wire shape uses camelCase for the privacy field.

    Lane 2's filter happens to also accept ``is_private`` and ``visibility``,
    but the API contract for /library/full is camelCase to match every
    other field on the same wire. Avoiding both names prevents drift.
    """
    resp = await client.get("/library/full?page=1&page_size=2")
    assert resp.status_code == 200
    body = resp.json()
    if not body.get("repos"):
        pytest.skip("library/full empty in this DB — nothing to assert")

    sample = body["repos"][0]
    assert "isPrivate" in sample
    # The snake_case shape stays internal — should NOT bleed onto the wire.
    assert "is_private" not in sample, (
        "/library/full must not emit both isPrivate (camelCase) AND "
        "is_private (snake_case) — pick one to avoid drift between "
        "Lane 2 and Lane 4 gate semantics. The contract is camelCase."
    )


@pytest.mark.asyncio
async def test_library_full_repo_contract_blocks_no_repo_without_isprivate(
    client: AsyncClient,
):
    """Cross-validate against Lane 2 / Lane 4 contract: every repo emitted
    by /library/full must have a privacy verdict that downstream gates
    can read. Equivalent of running ``classifyPrivacy()`` from the
    frontend's privacy-filter.ts on every wire repo.
    """
    resp = await client.get("/library/full?page=1&page_size=500")
    assert resp.status_code == 200
    repos = resp.json().get("repos", [])
    if not repos:
        pytest.skip("library/full empty in this DB — nothing to assert")

    missing = [
        r.get("name", "?")
        for r in repos
        if r.get("isPrivate") is None and r.get("is_private") is None
    ]
    assert not missing, (
        f"{len(missing)} repos missing privacy field — would fail Lane 2's "
        f"validate-privacy.ts and Lane 4's audit. Sample: {missing[:5]}"
    )


# ---------------------------------------------------------------------------
# KAN-188 — back-compat invariant: /library/full must STILL ship the aggregate
# fields alongside the per-repo array even after KAN-188 splits them out into
# /library/aggregates. Workato + MCP + eval runner all consume the legacy
# shape; breaking it is a P0 outage. Pin the contract here.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_library_full_response_shape_preserved(client: AsyncClient):
    """KAN-188 back-compat: /library/full keeps `repos` AND every aggregate.

    The aggregate builders moved into library_aggregates_helpers as part of
    KAN-188 so /library/aggregates can reuse them, but /library/full's wire
    shape is unchanged — every field that consumers (Workato, MCP, eval
    runner) read today must still be present.
    """
    resp = await client.get("/library/full?page=1&page_size=2")
    assert resp.status_code == 200
    body = resp.json()

    # The per-repo array is still THE primary payload of /library/full.
    assert "repos" in body, "KAN-188 regression: /library/full dropped `repos`"
    assert isinstance(body["repos"], list)

    # Every aggregate field that has shipped on /library/full must still ship.
    # KAN-188 made these available on a separate /library/aggregates endpoint
    # too — but /library/full retains them for back-compat.
    for k in (
        "stats", "gapAnalysis", "tagMetrics", "categories",
        "builderStats", "aiDevSkillStats", "pmSkillStats",
    ):
        assert k in body, (
            f"KAN-188 back-compat regression: /library/full dropped {k!r}; "
            "Workato + MCP + eval runner all read this field"
        )

    # Envelope keys that have shipped since KAN-151
    for k in ("username", "generatedAt", "page", "pageSize",
              "totalRepos", "totalPages"):
        assert k in body, f"/library/full envelope dropped {k!r}"


@pytest.mark.asyncio
async def test_library_full_tag_metrics_does_NOT_include_per_tag_repos_array(client: AsyncClient):
    """KAN-193: /library/full also drops tagMetrics[].repos.

    Both /library/full and /library/aggregates share the build_tag_metrics
    helper; the trim applies to both. Consumer audit (perditioinc) showed
    no production reader of tagMetric.repos in reporium frontend,
    reporium-mcp, reporium-evals, or reporium-audit, so dropping it from
    both endpoints is back-compat-safe.
    """
    resp = await client.get("/library/full?page=1&page_size=2")
    assert resp.status_code == 200
    body = resp.json()
    tag_metrics = body.get("tagMetrics") or []
    for tm in tag_metrics:
        assert "repos" not in tm, (
            "KAN-193 regression: /library/full's tagMetrics entry leaked "
            "the per-tag `repos` array."
        )
