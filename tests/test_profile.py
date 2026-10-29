from datetime import datetime, timezone

import pytest

from app.models import Repo
from app.profile import build_profile, estimate_level, language_profile, repo_weight


def make_repo(name, language, stars=0, pushed="2026-10-01", fork=False, archived=False, topics=()):
    return Repo(
        full_name=name,
        language=language,
        stars=stars,
        pushed_at=datetime.fromisoformat(pushed).replace(tzinfo=timezone.utc),
        fork=fork,
        archived=archived,
        topics=list(topics),
    )


def test_recent_and_popular_repos_weigh_more(now):
    fresh = make_repo("a/fresh", "Python", stars=50, pushed="2026-10-20")
    stale = make_repo("a/stale", "Python", stars=50, pushed="2022-10-20")
    unpopular = make_repo("a/small", "Python", stars=0, pushed="2026-10-20")
    assert repo_weight(fresh, now) > repo_weight(stale, now)
    assert repo_weight(fresh, now) > repo_weight(unpopular, now)


def test_language_profile_ignores_forks_and_archived(now):
    repos = [
        make_repo("a/one", "Python"),
        make_repo("a/two", "Rust", fork=True),
        make_repo("a/three", "Go", archived=True),
        make_repo("a/four", None),
    ]
    langs = language_profile(repos, now)
    assert set(langs) == {"Python"}
    assert langs["Python"] == pytest.approx(1.0)


def test_language_profile_sums_to_one(now):
    repos = [make_repo("a/x", "Python", stars=10), make_repo("a/y", "TypeScript"), make_repo("a/z", "TypeScript")]
    langs = language_profile(repos, now)
    assert sum(langs.values()) == pytest.approx(1.0)


def test_estimate_level_scales_with_activity():
    newbie = estimate_level([make_repo("a/x", "Python")], followers=0, account_age_years=0.2)
    veteran = estimate_level([make_repo(f"a/{i}", "Go", stars=300) for i in range(30)], followers=400, account_age_years=8)
    assert newbie == "beginner"
    assert veteran == "advanced"


def test_build_profile_from_demo_data(demo_client, now):
    user = demo_client.get_user("someone")
    repos = demo_client.get_user_repos("someone")
    profile = build_profile(user, repos, now)

    assert profile.username == "someone"
    assert profile.top_languages(2)[0] in {"Python", "TypeScript"}
    assert "Shell" not in profile.top_languages(2)  # old, unstarred repo
    assert "machine-learning" in profile.topics
    assert profile.level in {"beginner", "intermediate"}
    assert "keywords_text" not in profile.to_dict()
