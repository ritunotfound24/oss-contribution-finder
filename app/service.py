"""Orchestrates profile building, issue search, ranking and recommendations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from .difficulty import LEVELS, DifficultyModel
from .github_client import GitHubError, GitHubSource
from .models import ContributorProfile, ProjectRecommendation, Repo, ScoredIssue
from .profile import build_profile
from .ranker import rank_issues
from .recommender import recommend_projects

log = logging.getLogger(__name__)

# Labels maintainers use to invite outside contributors. A comma means OR in GitHub search.
INVITE_LABELS = '"good first issue","help wanted","good-first-issue","first-timers-only"'
MAX_REPO_LOOKUPS = 20


@dataclass
class FinderResult:
    profile: ContributorProfile
    issues: list[ScoredIssue]
    projects: list[ProjectRecommendation]
    searched_languages: list[str]
    level: str


def build_queries(languages: list[str]) -> list[str]:
    base = f"is:issue is:open archived:false no:assignee label:{INVITE_LABELS}"
    if not languages:
        return [base]
    return [f'{base} language:"{lang}"' for lang in languages]


class ContributionFinder:
    def __init__(self, source: GitHubSource, model: DifficultyModel, max_issues: int = 60):
        self.source = source
        self.model = model
        self.max_issues = max_issues

    def profile(self, username: str) -> ContributorProfile:
        user = self.source.get_user(username)
        repos = self.source.get_user_repos(username)
        return build_profile(user, repos)

    def find(
        self,
        username: str,
        level: str | None = None,
        languages: list[str] | None = None,
        now: datetime | None = None,
    ) -> FinderResult:
        now = now or datetime.now(timezone.utc)
        profile = self.profile(username)
        level = level if level in LEVELS else profile.level
        search_langs = [lang for lang in (languages or profile.top_languages(3)) if lang]

        per_query = max(10, self.max_issues // max(1, len(search_langs)))
        seen: set[int] = set()
        issues = []
        for query in build_queries(search_langs):
            for issue in self.source.search_issues(query, limit=per_query):
                if issue.id not in seen:
                    seen.add(issue.id)
                    issues.append(issue)

        repos: dict[str, Repo] = {}
        for name in list(dict.fromkeys(i.repo_full_name for i in issues))[:MAX_REPO_LOOKUPS]:
            try:
                repos[name] = self.source.get_repo(name)
            except GitHubError as exc:
                log.warning("Skipping repo %s: %s", name, exc)

        scored = rank_issues(profile, issues, repos, self.model, target_level=level, now=now)
        projects = recommend_projects(profile, scored, now=now)
        return FinderResult(profile=profile, issues=scored, projects=projects, searched_languages=search_langs, level=level)
