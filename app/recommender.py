"""Turn ranked issues into project-level recommendations."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from .models import ContributorProfile, ProjectRecommendation, Repo, ScoredIssue
from .ranker import language_score, repo_health_score


def recommend_projects(
    profile: ContributorProfile,
    scored: list[ScoredIssue],
    limit: int = 6,
    now: datetime | None = None,
) -> list[ProjectRecommendation]:
    """A project is worth joining if it has several good-fit issues and is healthy.

    score = 0.6 * mean(top 3 issue scores) + 0.2 * repo health + 0.2 * language fit,
    with a small bonus for having more than one matching issue.
    """
    now = now or datetime.now(timezone.utc)
    by_repo: dict[str, list[ScoredIssue]] = defaultdict(list)
    repos: dict[str, Repo] = {}
    for s in scored:
        if s.repo is None:
            continue
        by_repo[s.repo.full_name].append(s)
        repos[s.repo.full_name] = s.repo

    recs = []
    for name, items in by_repo.items():
        repo = repos[name]
        if repo.archived:
            continue
        top = sorted((i.score for i in items), reverse=True)[:3]
        issue_quality = sum(top) / len(top)
        health = repo_health_score(repo, now)
        lang = language_score(profile, repo)
        depth_bonus = min(len(items) - 1, 4) * 0.02
        score = 0.6 * issue_quality + 0.2 * health + 0.2 * lang + depth_bonus

        reasons = [f"{len(items)} open issue{'s' if len(items) != 1 else ''} that fit you"]
        if lang >= 0.5 and repo.language:
            reasons.append(f"Uses {repo.language}")
        shared = sorted(set(t.lower() for t in repo.topics) & set(profile.topics))
        if shared:
            reasons.append("Shares topics with your work: " + ", ".join(shared[:3]))
        if health >= 0.6:
            reasons.append("Active and well maintained")

        recs.append(ProjectRecommendation(repo=repo, score=score, matching_issues=len(items), reasons=reasons))

    recs.sort(key=lambda r: r.score, reverse=True)
    return recs[:limit]
