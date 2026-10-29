from unittest.mock import MagicMock

import pytest

from app.github_client import GitHubClient, GitHubError


def _response(status=200, payload=None, headers=None):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload if payload is not None else {}
    resp.headers = headers or {}
    return resp


def _client(*responses):
    session = MagicMock()
    session.headers = {}
    session.get.side_effect = list(responses)
    return GitHubClient(token="t", session=session), session


def test_sets_auth_header():
    client, session = _client()
    assert session.headers["Authorization"] == "Bearer t"


def test_search_parses_issues_and_skips_pull_requests():
    payload = {
        "items": [
            {"id": 1, "number": 5, "title": "Bug", "body": "x", "html_url": "u",
             "repository_url": "https://api.github.com/repos/acme/widgets", "labels": [{"name": "good first issue"}],
             "comments": 2, "updated_at": "2026-10-01T00:00:00Z"},
            {"id": 2, "number": 6, "title": "PR", "pull_request": {}, "repository_url": "https://api.github.com/repos/acme/widgets"},
        ]
    }
    client, _ = _client(_response(payload=payload))
    issues = client.search_issues("q")
    assert len(issues) == 1
    assert issues[0].repo_full_name == "acme/widgets"
    assert issues[0].labels == ["good first issue"]


def test_responses_are_cached():
    client, session = _client(_response(payload={"login": "a"}))
    client.get_user("a")
    client.get_user("a")
    assert session.get.call_count == 1


def test_not_found_raises():
    client, _ = _client(_response(status=404))
    with pytest.raises(GitHubError) as exc:
        client.get_user("missing")
    assert exc.value.status == 404


def test_rate_limit_message():
    client, _ = _client(_response(status=403, headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "9999999999"}))
    with pytest.raises(GitHubError) as exc:
        client.get_user("a")
    assert "rate limit" in str(exc.value).lower()
