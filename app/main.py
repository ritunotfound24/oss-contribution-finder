"""FastAPI app: HTML pages plus a JSON API."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import get_settings
from .difficulty import LEVELS, load_or_train
from .github_client import DemoGitHubClient, GitHubClient, GitHubError
from .service import ContributionFinder

HERE = Path(__file__).resolve().parent
USERNAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")

app = FastAPI(
    title="Open-Source Contribution Finder",
    description="Recommends open-source issues and projects that fit a GitHub user's skills.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")


@lru_cache
def get_finder() -> ContributionFinder:
    settings = get_settings()
    source = (
        DemoGitHubClient(settings.demo_data_dir)
        if settings.demo_mode
        else GitHubClient(settings.github_token, cache_ttl=settings.cache_ttl_seconds)
    )
    model = load_or_train(settings.model_path, settings.seed_data_path)
    return ContributionFinder(source, model, max_issues=settings.max_issues)


def _check_username(username: str) -> str:
    username = username.strip().lstrip("@")
    if not USERNAME_RE.match(username):
        raise HTTPException(status_code=422, detail="That doesn't look like a GitHub username")
    return username


def _parse_languages(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    langs = [part.strip() for part in raw.split(",") if part.strip()]
    return langs[:5] or None


def _github_http_error(exc: GitHubError) -> HTTPException:
    status = {404: 404, 403: 429, 429: 429}.get(exc.status or 0, 502)
    return HTTPException(status_code=status, detail=str(exc))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request, "index.html", {"levels": LEVELS, "demo": get_settings().demo_mode}
    )


@app.get("/find", response_class=HTMLResponse)
def find_page(
    request: Request,
    username: str = Query(..., min_length=1, max_length=39),
    level: str | None = Query(None),
    languages: str | None = Query(None, max_length=120),
    finder: ContributionFinder = Depends(get_finder),
):
    context = {"username": username, "levels": LEVELS, "level": level, "languages": languages or ""}
    try:
        result = finder.find(_check_username(username), level=level or None, languages=_parse_languages(languages))
    except HTTPException as exc:
        return templates.TemplateResponse(request, "index.html", {**context, "error": exc.detail}, status_code=exc.status_code)
    except GitHubError as exc:
        err = _github_http_error(exc)
        return templates.TemplateResponse(request, "index.html", {**context, "error": err.detail}, status_code=err.status_code)
    return templates.TemplateResponse(request, "results.html", {**context, "result": result})


@app.get("/api/profile/{username}")
def api_profile(username: str, finder: ContributionFinder = Depends(get_finder)) -> dict:
    try:
        return finder.profile(_check_username(username)).to_dict()
    except GitHubError as exc:
        raise _github_http_error(exc) from exc


@app.get("/api/recommendations/{username}")
def api_recommendations(
    username: str,
    level: str | None = Query(None, description="beginner, intermediate or advanced"),
    languages: str | None = Query(None, description="Comma-separated languages to search"),
    limit: int = Query(20, ge=1, le=100),
    finder: ContributionFinder = Depends(get_finder),
) -> dict:
    if level and level not in LEVELS:
        raise HTTPException(status_code=422, detail=f"level must be one of {', '.join(LEVELS)}")
    try:
        result = finder.find(_check_username(username), level=level, languages=_parse_languages(languages))
    except GitHubError as exc:
        raise _github_http_error(exc) from exc
    return {
        "profile": result.profile.to_dict(),
        "level": result.level,
        "searched_languages": result.searched_languages,
        "issues": [s.to_dict() for s in result.issues[:limit]],
        "projects": [p.to_dict() for p in result.projects],
    }
