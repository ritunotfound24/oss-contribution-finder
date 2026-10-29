from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.config import get_settings
from app.difficulty import DifficultyModel, load_labeled_csv
from app.github_client import DemoGitHubClient

NOW = datetime(2026, 10, 28, tzinfo=timezone.utc)


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def model(settings) -> DifficultyModel:
    texts, labels = load_labeled_csv(settings.seed_data_path)
    return DifficultyModel().fit(texts, labels)


@pytest.fixture()
def demo_client(settings) -> DemoGitHubClient:
    return DemoGitHubClient(settings.demo_data_dir)


@pytest.fixture()
def now() -> datetime:
    return NOW
