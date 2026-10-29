from app.models import Issue
from app.profile import build_profile
from app.ranker import WEIGHTS, availability_score, rank_issues, repo_health_score
from app.recommender import recommend_projects


def _setup(demo_client, now):
    profile = build_profile(demo_client.get_user("u"), demo_client.get_user_repos("u"), now)
    issues = demo_client.search_issues("is:issue", limit=100)
    repos = {i.repo_full_name: demo_client.get_repo(i.repo_full_name) for i in issues}
    return profile, issues, repos


def test_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_scores_are_sorted_and_bounded(demo_client, model, now):
    profile, issues, repos = _setup(demo_client, now)
    ranked = rank_issues(profile, issues, repos, model, now=now)
    assert len(ranked) == len(issues)
    scores = [s.score for s in ranked]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= s <= 1 for s in scores)
    for s in ranked:
        assert set(s.breakdown) == set(WEIGHTS)


def test_assigned_and_archived_issues_rank_low(demo_client, model, now):
    profile, issues, repos = _setup(demo_client, now)
    ranked = rank_issues(profile, issues, repos, model, target_level="beginner", now=now)
    names = [s.issue.title for s in ranked]
    assigned = next(s for s in ranked if s.issue.assignee)
    archived = next(s for s in ranked if s.repo and s.repo.archived)
    assert assigned.breakdown["availability"] == 0
    assert archived.breakdown["repo_health"] == 0
    assert names.index(archived.issue.title) > len(names) // 2


def test_target_level_changes_ranking(demo_client, model, now):
    profile, issues, repos = _setup(demo_client, now)
    beginner = rank_issues(profile, issues, repos, model, target_level="beginner", now=now)
    advanced = rank_issues(profile, issues, repos, model, target_level="advanced", now=now)
    assert beginner[0].difficulty == "beginner"
    top_adv = [s.difficulty for s in advanced[:5]]
    assert "advanced" in top_adv or "intermediate" in top_adv


def test_empty_issue_list(demo_client, model, now):
    profile, _, _ = _setup(demo_client, now)
    assert rank_issues(profile, [], {}, model, now=now) == []


def test_availability_penalizes_busy_threads():
    quiet = Issue(id=1, number=1, title="t", body="", repo_full_name="a/b", html_url="", comments=0)
    busy = Issue(id=2, number=2, title="t", body="", repo_full_name="a/b", html_url="", comments=20)
    taken = Issue(id=3, number=3, title="t", body="", repo_full_name="a/b", html_url="", assignee="x")
    assert availability_score(quiet) > availability_score(busy) > availability_score(taken) == 0


def test_repo_health_unknown_repo_is_neutral(now):
    assert 0 < repo_health_score(None, now) < 1


def test_project_recommendations(demo_client, model, now):
    profile, issues, repos = _setup(demo_client, now)
    ranked = rank_issues(profile, issues, repos, model, now=now)
    projects = recommend_projects(profile, ranked, now=now)
    names = [p.repo.full_name for p in projects]
    assert "oldtools/legacy-grid" not in names  # archived
    assert len(names) == len(set(names))
    assert all(p.matching_issues >= 1 and p.reasons for p in projects)
    assert [p.score for p in projects] == sorted((p.score for p in projects), reverse=True)
