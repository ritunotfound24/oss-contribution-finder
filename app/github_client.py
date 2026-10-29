"""Thin GitHub REST API client with a TTL cache, plus an offline demo client."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Protocol

import requests

from .models import Issue, Repo

API = "https://api.github.com"


class GitHubError(Exception):
    """Raised for API errors we want to show the user (not found, rate limited...)."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class GitHubSource(Protocol):
    def get_user(self, username: str) -> dict[str, Any]: ...
    def get_user_repos(self, username: str, limit: int = 100) -> list[Repo]: ...
    def search_issues(self, query: str, limit: int = 30) -> list[Issue]: ...
    def get_repo(self, full_name: str) -> Repo: ...


class GitHubClient:
    def __init__(self, token: str | None = None, cache_ttl: int = 900, session: requests.Session | None = None):
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "oss-contribution-finder",
            }
        )
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self.cache_ttl = cache_ttl
        self._cache: dict[str, tuple[float, Any]] = {}

    # ---- low level ----
    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        key = path + "?" + json.dumps(params or {}, sort_keys=True)
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < self.cache_ttl:
            return hit[1]

        try:
            resp = self.session.get(f"{API}{path}", params=params, timeout=15)
        except requests.RequestException as exc:
            raise GitHubError(f"Could not reach GitHub: {exc}") from exc

        if resp.status_code == 404:
            raise GitHubError("Not found on GitHub", 404)
        if resp.status_code in (403, 429) and resp.headers.get("X-RateLimit-Remaining") == "0":
            reset = resp.headers.get("X-RateLimit-Reset")
            wait = max(0, int(reset) - int(time.time())) if reset and reset.isdigit() else None
            hint = f" Try again in {wait // 60 + 1} min" if wait is not None else ""
            raise GitHubError(f"GitHub rate limit reached.{hint} Set GITHUB_TOKEN for a higher limit.", resp.status_code)
        if resp.status_code >= 400:
            message = ""
            try:
                message = resp.json().get("message", "")
            except ValueError:
                pass
            raise GitHubError(f"GitHub API error {resp.status_code}: {message}".strip(), resp.status_code)

        data = resp.json()
        self._cache[key] = (time.monotonic(), data)
        return data

    # ---- endpoints ----
    def get_user(self, username: str) -> dict[str, Any]:
        return self._get(f"/users/{username}")

    def get_user_repos(self, username: str, limit: int = 100) -> list[Repo]:
        data = self._get(
            f"/users/{username}/repos",
            {"per_page": min(limit, 100), "sort": "pushed", "type": "owner"},
        )
        return [Repo.from_api(r) for r in data]

    def search_issues(self, query: str, limit: int = 30) -> list[Issue]:
        data = self._get(
            "/search/issues",
            {"q": query, "per_page": min(limit, 100), "sort": "updated", "order": "desc"},
        )
        return [Issue.from_api(i) for i in data.get("items", []) if "pull_request" not in i]

    def get_repo(self, full_name: str) -> Repo:
        return Repo.from_api(self._get(f"/repos/{full_name}"))


class DemoGitHubClient:
    """Serves bundled JSON so the app runs without network access or a token."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def _load(self, name: str) -> Any:
        return json.loads((self.data_dir / name).read_text(encoding="utf-8"))

    def get_user(self, username: str) -> dict[str, Any]:
        user = self._load("user.json")
        # Any username works in demo mode; show it so the UI reflects the input.
        return {**user, "login": username}

    def get_user_repos(self, username: str, limit: int = 100) -> list[Repo]:
        return [Repo.from_api(r) for r in self._load("user_repos.json")][:limit]

    def search_issues(self, query: str, limit: int = 30) -> list[Issue]:
        issues = [Issue.from_api(i) for i in self._load("issues.json")]
        langs = [part.split(":", 1)[1].strip('"').lower() for part in query.split() if part.startswith("language:")]
        if langs:
            repo_lang = {r["full_name"]: (r.get("language") or "").lower() for r in self._load("repos.json")}
            issues = [i for i in issues if repo_lang.get(i.repo_full_name) in langs]
        return issues[:limit]

    def get_repo(self, full_name: str) -> Repo:
        for r in self._load("repos.json"):
            if r["full_name"] == full_name:
                return Repo.from_api(r)
        raise GitHubError("Not found on GitHub", 404)
