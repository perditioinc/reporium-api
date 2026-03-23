"""Tests for CONTRACT.md enforcement — sanitize_repo() must never return null fields."""
from app.routers.library_full import sanitize_repo


def test_sanitize_repo_never_returns_null_fields():
    """Pass a repo with all fields null — every field must get a fallback."""
    empty_repo = {
        "name": "test-repo",
        "fullName": "perditioinc/test-repo",
        "description": None,
        "readmeSummary": None,
        "primaryCategory": None,
        "allCategories": None,
        "enrichedTags": None,
        "builders": None,
        "pmSkills": None,
        "industries": None,
        "aiDevSkills": None,
        "programmingLanguages": None,
        "topics": None,
        "commitStats": None,
        "recentCommits": None,
        "commitsLast7Days": None,
        "commitsLast30Days": None,
        "commitsLast90Days": None,
        "languageBreakdown": None,
        "languagePercentages": None,
        "stars": None,
        "forks": None,
        "weeklyCommitCount": None,
        "totalCommitsFetched": None,
    }
    result = sanitize_repo(empty_repo)

    # No field should be None
    assert result["description"] is not None and result["description"] != ""
    assert result["readmeSummary"] is not None
    assert result["primaryCategory"] is not None
    assert isinstance(result["allCategories"], list) and len(result["allCategories"]) > 0
    assert isinstance(result["enrichedTags"], list)
    assert isinstance(result["builders"], list) and len(result["builders"]) > 0
    assert isinstance(result["pmSkills"], list)
    assert isinstance(result["industries"], list)
    assert isinstance(result["aiDevSkills"], list)
    assert isinstance(result["programmingLanguages"], list)
    assert isinstance(result["commitStats"], dict)
    assert isinstance(result["recentCommits"], list)
    assert isinstance(result["languageBreakdown"], dict)
    assert isinstance(result["languagePercentages"], dict)
    assert isinstance(result["stars"], int)
    assert isinstance(result["forks"], int)


def test_sanitize_repo_preserves_existing_data():
    """If a repo already has data, sanitize must not overwrite it."""
    good_repo = {
        "name": "reporium",
        "fullName": "perditioinc/reporium",
        "description": "AI tool discovery",
        "readmeSummary": "Reporium is a platform.",
        "primaryCategory": "Dev Tools & Automation",
        "allCategories": ["Dev Tools & Automation", "AI Agents"],
        "enrichedTags": ["ai", "python"],
        "builders": [{"login": "perditioinc", "name": None, "type": "user",
                      "avatarUrl": "", "isKnownOrg": False, "orgCategory": "individual"}],
        "pmSkills": ["Developer Platform"],
        "industries": ["Developer Tools"],
        "aiDevSkills": ["tooling"],
        "programmingLanguages": ["TypeScript"],
        "topics": ["ai"],
        "commitStats": {"today": 1, "last7Days": 5, "last30Days": 20,
                        "last90Days": 60, "recentCommits": []},
        "recentCommits": [],
        "commitsLast7Days": [],
        "commitsLast30Days": [],
        "commitsLast90Days": [],
        "languageBreakdown": {"TypeScript": 50000},
        "languagePercentages": {"TypeScript": 100.0},
        "stars": 10,
        "forks": 2,
        "weeklyCommitCount": 5,
        "totalCommitsFetched": 20,
    }
    result = sanitize_repo(good_repo)

    # All existing values must be preserved
    assert result["description"] == "AI tool discovery"
    assert result["primaryCategory"] == "Dev Tools & Automation"
    assert result["enrichedTags"] == ["ai", "python"]
    assert result["stars"] == 10
    assert result["commitStats"]["last7Days"] == 5


def test_sanitize_description_falls_back_to_summary():
    """If description is null but summary exists, use first 150 chars of summary."""
    repo = {
        "name": "test",
        "fullName": "perditioinc/test",
        "description": None,
        "readmeSummary": "This is a very long summary that describes the repository in detail.",
    }
    result = sanitize_repo(repo)
    assert result["description"] == "This is a very long summary that describes the repository in detail."


def test_sanitize_builders_falls_back_to_owner():
    """If builders is empty, derive from fullName owner."""
    repo = {
        "name": "test",
        "fullName": "perditioinc/test",
        "builders": [],
    }
    result = sanitize_repo(repo)
    assert len(result["builders"]) == 1
    assert result["builders"][0]["login"] == "perditioinc"
