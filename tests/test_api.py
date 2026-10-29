import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app import main  # noqa: E402
from app.service import ContributionFinder  # noqa: E402


@pytest.fixture()
def client(demo_client, model):
    main.app.dependency_overrides[main.get_finder] = lambda: ContributionFinder(demo_client, model)
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "GitHub username" in resp.text


def test_results_page(client):
    resp = client.get("/find", params={"username": "demo-user"})
    assert resp.status_code == 200
    assert "Issues ranked for you" in resp.text


def test_invalid_username(client):
    resp = client.get("/api/profile/not a user!")
    assert resp.status_code == 422


def test_recommendations_api(client):
    resp = client.get("/api/recommendations/demo-user", params={"level": "beginner", "limit": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["level"] == "beginner"
    assert len(body["issues"]) <= 5
    assert {"title", "url", "score", "difficulty", "breakdown", "reasons"} <= set(body["issues"][0])


def test_bad_level(client):
    assert client.get("/api/recommendations/demo-user", params={"level": "wizard"}).status_code == 422
