"""Score open issues for a contributor.

Each issue gets six 0-1 feature scores that are combined with fixed weights.
The breakdown is kept so the UI can explain every recommendation.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .difficulty import LEVELS, DifficultyModel
from .models import ContributorProfile, Issue, Repo, ScoredIssue

WEIGHTS = {
    "language": 0.25,
    "similarity": 0.25,
    "difficulty": 0.20,
    "freshness": 0.10,
    "availability": 0.10,
    "repo_health": 0.10,
}

# How well each issue level suits each contributor level (rows: contributor, cols: issue).
DIFFICULTY_FIT = np.array(
    [
        [1.0, 0.4, 0.0],  # beginner
        [0.6, 1.0, 0.4],  # intermediate
        [0.2, 0.7, 1.0],  # advanced
    ]
)


def _days_since(ts: datetime | None, now: datetime) -> float | None:
    if ts is None:
        return None
    return max(0.0, (now - ts).total_seconds() / 86400)


def _decay(days: float | None, half_life_days: float, default: float = 0.3) -> float:
    return default if days is None else 0.5 ** (days / half_life_days)


def language_score(profile: ContributorProfile, repo: Repo | None) -> float:
    if not profile.languages:
        return 0.5  # nothing to go on; stay neutral
    if repo is None or not repo.language:
        return 0.3
    top = max(profile.languages.values())
    return profile.languages.get(repo.language, 0.0) / top


def repo_health_score(repo: Repo | None, now: datetime) -> float:
    if repo is None:
        return 0.4
    if repo.archived:
        return 0.0
    popularity = min(1.0, math.log1p(repo.stars) / math.log1p(5000))
    activity = _decay(_days_since(repo.pushed_at, now), half_life_days=90)
    return 0.5 * popularity + 0.5 * activity


def availability_score(issue: Issue) -> float:
    if issue.assignee:
        return 0.0
    # Long threads usually mean the issue is contested or harder than it looks.
    return 1.0 / (1.0 + issue.comments / 5.0)


def text_similarities(profile: ContributorProfile, docs: list[str]) -> np.ndarray:
    if not docs or not profile.keywords_text.strip():
        return np.zeros(len(docs))
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    try:
        matrix = vec.fit_transform([profile.keywords_text, *docs])
    except ValueError:  # empty vocabulary
        return np.zeros(len(docs))
    sims = cosine_similarity(matrix[0], matrix[1:]).ravel()
    # Raw cosine between a profile and a short issue is small; rescale to use 0-1.
    return np.clip(sims / 0.3, 0.0, 1.0)


def rank_issues(
    profile: ContributorProfile,
    issues: list[Issue],
    repos: dict[str, Repo],
    model: DifficultyModel,
    target_level: str | None = None,
    now: datetime | None = None,
) -> list[ScoredIssue]:
    if not issues:
        return []
    now = now or datetime.now(timezone.utc)
    level = target_level if target_level in LEVELS else profile.level
    fit_row = DIFFICULTY_FIT[LEVELS.index(level)]

    docs = []
    for issue in issues:
        repo = repos.get(issue.repo_full_name)
        context = f"{repo.description} {' '.join(repo.topics)} {repo.language or ''}" if repo else ""
        docs.append(f"{issue.text} {' '.join(issue.labels)} {context}")
    sims = text_similarities(profile, docs)

    results: list[ScoredIssue] = []
    for issue, sim in zip(issues, sims):
        repo = repos.get(issue.repo_full_name)
        pred = model.predict(issue.text, issue.labels)
        probs = np.array([pred.probabilities[lvl] for lvl in LEVELS])

        breakdown = {
            "language": language_score(profile, repo),
            "similarity": float(sim),
            "difficulty": float(probs @ fit_row),
            "freshness": _decay(_days_since(issue.updated_at, now), half_life_days=30),
            "availability": availability_score(issue),
            "repo_health": repo_health_score(repo, now),
        }
        score = sum(WEIGHTS[k] * v for k, v in breakdown.items())

        reasons = []
        if repo and repo.language and breakdown["language"] >= 0.5:
            reasons.append(f"Written in {repo.language}, one of your main languages")
        if breakdown["similarity"] >= 0.5:
            reasons.append("Close to topics in your own repositories")
        if pred.level == level:
            reasons.append(f"Looks {pred.level}-friendly ({pred.confidence:.0%} confidence)")
        if issue.comments == 0 and not issue.assignee:
            reasons.append("No one has commented yet")
        if repo and repo.stars >= 1000:
            reasons.append(f"Popular project ({repo.stars:,} stars)")

        results.append(
            ScoredIssue(
                issue=issue,
                repo=repo,
                score=score,
                difficulty=pred.level,
                difficulty_confidence=pred.confidence,
                breakdown=breakdown,
                reasons=reasons,
            )
        )

    results.sort(key=lambda s: s.score, reverse=True)
    return results
