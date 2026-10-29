from app.github_client import GitHubError
from app.service import ContributionFinder, build_queries


def test_build_queries_per_language():
    qs = build_queries(["Python", "TypeScript"])
    assert len(qs) == 2
    assert all("is:open" in q and "no:assignee" in q and "good first issue" in q for q in qs)
    assert 'language:"Python"' in qs[0]


def test_build_queries_without_languages():
    assert len(build_queries([])) == 1


def test_find_end_to_end(demo_client, model, now):
    finder = ContributionFinder(demo_client, model, max_issues=50)
    result = finder.find("demo-user", now=now)
    assert result.profile.username == "demo-user"
    assert result.issues, "expected ranked issues"
    assert result.projects, "expected project recommendations"
    assert set(result.searched_languages) <= {"Python", "TypeScript", "JavaScript", "Shell"}
    ids = [s.issue.id for s in result.issues]
    assert len(ids) == len(set(ids)), "issues should be de-duplicated across queries"


def test_find_respects_overrides(demo_client, model, now):
    finder = ContributionFinder(demo_client, model)
    result = finder.find("demo-user", level="advanced", languages=["Python"], now=now)
    assert result.level == "advanced"
    assert result.searched_languages == ["Python"]
    assert all(s.repo is None or s.repo.language == "Python" for s in result.issues)


def test_find_skips_repos_that_fail(demo_client, model, now, monkeypatch):
    original = demo_client.get_repo

    def flaky(name):
        if name.startswith("openlearn/"):
            raise GitHubError("boom", 500)
        return original(name)

    monkeypatch.setattr(demo_client, "get_repo", flaky)
    result = ContributionFinder(demo_client, model).find("demo-user", now=now)
    assert any(s.repo is None for s in result.issues)
