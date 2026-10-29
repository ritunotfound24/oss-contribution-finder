"""Runtime settings read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    github_token: str | None
    demo_mode: bool
    model_path: Path
    seed_data_path: Path
    demo_data_dir: Path
    cache_ttl_seconds: int
    max_issues: int


def get_settings() -> Settings:
    token = os.getenv("GITHUB_TOKEN") or None
    return Settings(
        github_token=token,
        # Demo mode serves bundled sample data so the app works offline and in tests.
        demo_mode=os.getenv("DEMO_MODE", "0").lower() in {"1", "true", "yes"},
        model_path=Path(os.getenv("MODEL_PATH", ROOT / "models" / "difficulty.joblib")),
        seed_data_path=ROOT / "data" / "seed_issues.csv",
        demo_data_dir=ROOT / "data" / "demo",
        cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "900")),
        max_issues=int(os.getenv("MAX_ISSUES", "60")),
    )
