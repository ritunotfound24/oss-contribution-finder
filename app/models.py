"""Plain data classes shared across the app."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Repo:
    full_name: str
    description: str = ""
    language: str | None = None
    topics: list[str] = field(default_factory=list)
    stars: int = 0
    forks: int = 0
    open_issues: int = 0
    pushed_at: datetime | None = None
    fork: bool = False
    archived: bool = False
    html_url: str = ""

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Repo:
        return cls(
            full_name=data["full_name"],
            description=data.get("description") or "",
            language=data.get("language"),
            topics=list(data.get("topics") or []),
            stars=int(data.get("stargazers_count") or 0),
            forks=int(data.get("forks_count") or 0),
            open_issues=int(data.get("open_issues_count") or 0),
            pushed_at=parse_ts(data.get("pushed_at")),
            fork=bool(data.get("fork")),
            archived=bool(data.get("archived")),
            html_url=data.get("html_url") or f"https://github.com/{data['full_name']}",
        )


@dataclass
class Issue:
    id: int
    number: int
    title: str
    body: str
    repo_full_name: str
    html_url: str
    labels: list[str] = field(default_factory=list)
    comments: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    assignee: str | None = None

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.body}"

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Issue:
        # The search API gives repository_url like https://api.github.com/repos/owner/name
        repo_url: str = data.get("repository_url", "")
        repo_full_name = "/".join(repo_url.rstrip("/").split("/")[-2:]) if repo_url else ""
        assignee = (data.get("assignee") or {}).get("login")
        return cls(
            id=int(data["id"]),
            number=int(data["number"]),
            title=data.get("title") or "",
            body=(data.get("body") or "")[:4000],
            repo_full_name=repo_full_name,
            html_url=data.get("html_url") or "",
            labels=[lbl["name"] if isinstance(lbl, dict) else str(lbl) for lbl in data.get("labels", [])],
            comments=int(data.get("comments") or 0),
            created_at=parse_ts(data.get("created_at")),
            updated_at=parse_ts(data.get("updated_at")),
            assignee=assignee,
        )


@dataclass
class ContributorProfile:
    username: str
    name: str | None
    bio: str
    public_repos: int
    followers: int
    languages: dict[str, float]  # language -> weight, sums to 1
    topics: list[str]
    keywords_text: str  # text blob used for similarity
    level: str  # beginner | intermediate | advanced
    account_age_years: float = 0.0

    def top_languages(self, n: int = 5) -> list[str]:
        return [lang for lang, _ in sorted(self.languages.items(), key=lambda kv: -kv[1])[:n]]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["top_languages"] = self.top_languages()
        d.pop("keywords_text")
        return d


@dataclass
class ScoredIssue:
    issue: Issue
    repo: Repo | None
    score: float
    difficulty: str
    difficulty_confidence: float
    breakdown: dict[str, float]
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.issue.title,
            "url": self.issue.html_url,
            "repo": self.issue.repo_full_name,
            "labels": self.issue.labels,
            "comments": self.issue.comments,
            "score": round(self.score, 3),
            "difficulty": self.difficulty,
            "difficulty_confidence": round(self.difficulty_confidence, 3),
            "breakdown": {k: round(v, 3) for k, v in self.breakdown.items()},
            "reasons": self.reasons,
        }


@dataclass
class ProjectRecommendation:
    repo: Repo
    score: float
    matching_issues: int
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo.full_name,
            "url": self.repo.html_url,
            "description": self.repo.description,
            "language": self.repo.language,
            "stars": self.repo.stars,
            "score": round(self.score, 3),
            "matching_issues": self.matching_issues,
            "reasons": self.reasons,
        }


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
