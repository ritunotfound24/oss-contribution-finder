import pytest

from app.difficulty import LEVELS, DifficultyModel, level_from_labels, load_or_train


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (["good first issue"], "beginner"),
        (["Good-First-Issue", "bug"], "beginner"),
        (["documentation"], "beginner"),
        (["difficulty: hard"], "advanced"),
        (["Difficulty/Medium"], "intermediate"),
        (["bug", "help wanted"], None),
        ([], None),
    ],
)
def test_level_from_labels(labels, expected):
    assert level_from_labels(labels) == expected


def test_model_separates_obvious_cases(model):
    easy = model.predict("Fix typo in README docs", [])
    hard = model.predict("Race condition in distributed lock under concurrency; redesign the scheduler", [])
    assert easy.level == "beginner"
    assert hard.level == "advanced"


def test_probabilities_sum_to_one(model):
    pred = model.predict("Add pagination to the API endpoint", [])
    assert set(pred.probabilities) == set(LEVELS)
    assert sum(pred.probabilities.values()) == pytest.approx(1.0)
    assert 0 <= pred.confidence <= 1


def test_maintainer_label_shifts_prediction(model):
    text = "Implement streaming parser with backpressure for huge files"
    plain = model.predict(text, [])
    labeled = model.predict(text, ["good first issue"])
    assert labeled.source == "label+model"
    assert labeled.probabilities["beginner"] > plain.probabilities["beginner"]
    assert labeled.level == "beginner"


def test_fit_requires_two_classes():
    with pytest.raises(ValueError):
        DifficultyModel().fit(["a", "b"], ["beginner", "beginner"])


def test_save_and_load_roundtrip(model, tmp_path):
    path = tmp_path / "m.joblib"
    model.save(path)
    loaded = DifficultyModel.load(path)
    text = "Fix broken link in docs"
    assert loaded.predict(text).level == model.predict(text).level


def test_load_or_train_creates_model(settings, tmp_path):
    path = tmp_path / "nested" / "model.joblib"
    m = load_or_train(path, settings.seed_data_path)
    assert path.exists()
    assert sorted(m.classes) == sorted(LEVELS)
