"""Build a contributor profile from a GitHub user and their repositories."""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from .models import ContributorProfile, Repo, parse_ts

LEVELS = ("beginner", "intermediate", "advanced")
RECENCY_HALF_LIFE_YEARS = 2.0


def _years_between(start: datetime | None, end: datetime) -> float:
    if start is None:
        return 0.0
    return max(0.0, (end - start).total_seconds() / (365.25 * 24 * 3600))


def repo_weight(repo: Repo, now: datetime) -> float:
    """Popular and recently active repos say more about someone's current skills."""
    popularity = 1.0 + 0.5 * math.log1p(repo.stars)
    age = _years_between(repo.pushed_at, now) if repo.pushed_at else RECENCY_HALF_LIFE_YEARS
    recency = 0.5 ** (age / RECENCY_HALF_LIFE_YEARS)
    return popularity * recency


def language_profile(repos: list[Repo], now: datetime) -> dict[str, float]:
    totals: Counter[str] = Counter()
    for repo in repos:
        if repo.fork or repo.archived or not repo.language:
            continue
        totals[repo.language] += repo_weight(repo, now)
    total = sum(totals.values())
    if total == 0:
        return {}
    return {lang: w / total for lang, w in totals.most_common()}


def estimate_level(own_repos: list[Repo], followers: int, account_age_years: float) -> str:
    """Rough experience level from public signals. Users can override it in the UI."""
    stars = sum(r.stars for r in own_repos)
    points = 0
    points += min(len(own_repos), 30) / 10  # up to 3
    points += min(math.log1p(stars), 6) / 2  # up to 3
    points += min(math.log1p(followers), 5) / 2.5  # up to 2
    points += min(account_age_years, 6) / 3  # up to 2
    if points >= 6:
        return "advanced"
    if points >= 3:
        return "intermediate"
    return "beginner"


def build_profile(user: dict[str, Any], repos: list[Repo], now: datetime | None = None) -> ContributorProfile:
    now = now or datetime.now(timezone.utc)
    own = [r for r in repos if not r.fork]

    languages = language_profile(repos, now)

    topic_counts: Counter[str] = Counter()
    for r in own:
        topic_counts.update(t.lower() for t in r.topics)
    topics = [t for t, _ in topic_counts.most_common(15)]

    bio = user.get("bio") or ""
    descriptions = " ".join(r.description for r in own if r.description)
    # Repeat languages and topics so they carry more weight than incidental words.
    lang_terms = " ".join(f"{lang} " * max(1, round(w * 5)) for lang, w in languages.items())
    keywords_text = " ".join([bio, descriptions, " ".join(topics) * 2, lang_terms]).strip()

    account_age = _years_between(parse_ts(user.get("created_at")), now)
    followers = int(user.get("followers") or 0)

    return ContributorProfile(
        username=user.get("login", ""),
        name=user.get("name"),
        bio=bio,
        public_repos=int(user.get("public_repos") or len(repos)),
        followers=followers,
        languages=languages,
        topics=topics,
        keywords_text=keywords_text,
        level=estimate_level(own, followers, account_age),
        account_age_years=round(account_age, 1),
    )
