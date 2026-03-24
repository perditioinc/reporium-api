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
    SYSTEM_TAGS,
    _AI_DEV_SKILL_GROUPS,
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

    def test_langchain_maps_to_ai_agents(self):
        repos = [_make_repo("r1", ["LangChain"])]
        stats = {s["skill"]: s for s in _build_ai_dev_skill_stats(repos)}
        assert stats["AI Agents & Orchestration"]["repoCount"] == 1

    def test_multiple_tags_same_group_counted_once_per_repo(self):
        # A repo with both vLLM and SGLang should count as 1 for Inference & Serving
        repos = [_make_repo("r1", ["vLLM", "SGLang", "TensorRT"])]
        stats = {s["skill"]: s for s in _build_ai_dev_skill_stats(repos)}
        assert stats["Inference & Serving"]["repoCount"] == 1

    def test_repo_counted_in_multiple_groups(self):
        repos = [_make_repo("r1", ["vLLM", "RAG"])]
        stats = {s["skill"]: s for s in _build_ai_dev_skill_stats(repos)}
        assert stats["Inference & Serving"]["repoCount"] == 1
        assert stats["RAG & Knowledge"]["repoCount"] == 1

    def test_unknown_tags_produce_zero_counts(self):
        repos = [_make_repo("r1", ["unknown-tag-xyz"])]
        stats = {s["skill"]: s for s in _build_ai_dev_skill_stats(repos)}
        # Every group exists in output (one per taxonomy group)
        assert len(stats) == len(_AI_DEV_SKILL_GROUPS)
        for s in stats.values():
            assert s["repoCount"] == 0

    def test_empty_repos_returns_all_groups_with_zero(self):
        stats = _build_ai_dev_skill_stats([])
        assert len(stats) == len(_AI_DEV_SKILL_GROUPS)
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

    def test_skill_tags_in_ai_dev_skills_field_also_counted(self):
        # Tags in aiDevSkills (not enrichedTags) should also map to groups
        repos = [_make_repo("r1", [], ai_skills=["vLLM"])]
        stats = {s["skill"]: s for s in _build_ai_dev_skill_stats(repos)}
        assert stats["Inference & Serving"]["repoCount"] == 1

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
        assert skill_names == list(_AI_DEV_SKILL_GROUPS.keys())

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
