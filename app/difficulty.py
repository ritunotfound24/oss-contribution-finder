"""Issue difficulty classifier.

A TF-IDF + logistic regression model predicts beginner / intermediate / advanced
from an issue's title and body. When maintainers have already labeled an issue
(e.g. "good first issue"), that label is blended with the model's prediction.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

LEVELS = ("beginner", "intermediate", "advanced")

_BEGINNER_LABELS = re.compile(
    r"good[ -]?first[ -]?issue|first[ -]?timers?[ -]?only|beginner|easy|starter|"
    r"difficulty[:/ -]*(low|easy)|low[ -]hanging|documentation|docs|typo",
    re.I,
)
_ADVANCED_LABELS = re.compile(r"difficulty[:/ -]*(high|hard)|\bhard\b|complex|expert|advanced", re.I)
_INTERMEDIATE_LABELS = re.compile(r"difficulty[:/ -]*(medium|moderate)|intermediate|medium", re.I)

# How much a maintainer label counts against the model when both exist.
LABEL_BLEND = 0.6


def level_from_labels(labels: list[str]) -> str | None:
    """Map GitHub labels to a difficulty level, or None if they say nothing about it."""
    joined = " | ".join(labels)
    if _BEGINNER_LABELS.search(joined):
        return "beginner"
    if _ADVANCED_LABELS.search(joined):
        return "advanced"
    if _INTERMEDIATE_LABELS.search(joined):
        return "intermediate"
    return None


def load_labeled_csv(path: Path) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            label = row["label"].strip().lower()
            if label not in LEVELS:
                continue
            texts.append(f"{row['title']}\n{row.get('body') or ''}")
            labels.append(label)
    return texts, labels


@dataclass
class Prediction:
    level: str
    confidence: float
    probabilities: dict[str, float]
    source: str  # "model" | "label+model"


class DifficultyModel:
    def __init__(self, pipeline: Pipeline | None = None):
        self.pipeline = pipeline or self._new_pipeline()

    @staticmethod
    def _new_pipeline() -> Pipeline:
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, stop_words="english", max_features=20000),
                ),
                ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")),
            ]
        )

    def fit(self, texts: list[str], labels: list[str]) -> DifficultyModel:
        if len(set(labels)) < 2:
            raise ValueError("Need examples from at least two difficulty levels to train")
        self.pipeline.fit(texts, labels)
        return self

    @property
    def classes(self) -> list[str]:
        return list(self.pipeline.classes_)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Probabilities in LEVELS order (missing classes get 0)."""
        raw = self.pipeline.predict_proba(texts)
        out = np.zeros((len(texts), len(LEVELS)))
        for j, cls in enumerate(self.pipeline.classes_):
            out[:, LEVELS.index(cls)] = raw[:, j]
        return out

    def predict(self, text: str, labels: list[str] | None = None) -> Prediction:
        probs = self.predict_proba([text])[0]
        source = "model"
        hinted = level_from_labels(labels or [])
        if hinted:
            onehot = np.zeros(len(LEVELS))
            onehot[LEVELS.index(hinted)] = 1.0
            probs = LABEL_BLEND * onehot + (1 - LABEL_BLEND) * probs
            source = "label+model"
        idx = int(np.argmax(probs))
        return Prediction(
            level=LEVELS[idx],
            confidence=float(probs[idx]),
            probabilities={lvl: float(p) for lvl, p in zip(LEVELS, probs)},
            source=source,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: Path) -> DifficultyModel:
        return cls(joblib.load(path))


def load_or_train(model_path: Path, seed_csv: Path) -> DifficultyModel:
    """Load a saved model, or train one from the bundled seed data on first run."""
    if model_path.exists():
        try:
            return DifficultyModel.load(model_path)
        except Exception:  # noqa: BLE001 - a stale/incompatible pickle shouldn't crash the app
            pass
    texts, labels = load_labeled_csv(seed_csv)
    model = DifficultyModel().fit(texts, labels)
    try:
        model.save(model_path)
    except OSError:
        pass  # read-only deploys still work with the in-memory model
    return model
