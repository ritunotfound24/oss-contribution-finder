"""Train and evaluate the issue difficulty classifier.

Usage:
    python -m scripts.train                          # seed data only
    python -m scripts.train --extra data/github_issues.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from app.config import get_settings
from app.difficulty import LEVELS, DifficultyModel, load_labeled_csv


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extra", type=Path, nargs="*", default=[], help="More labeled CSVs (title,body,label)")
    parser.add_argument("--out", type=Path, default=settings.model_path)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    texts, labels = load_labeled_csv(settings.seed_data_path)
    for path in args.extra:
        t, lab = load_labeled_csv(path)
        texts += t
        labels += lab

    counts = {lvl: labels.count(lvl) for lvl in LEVELS}
    print(f"Training on {len(texts)} issues: {counts}")

    folds = min(args.folds, min(c for c in counts.values() if c) or 2)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    preds = cross_val_predict(DifficultyModel().pipeline, texts, labels, cv=cv)

    print(f"\n{folds}-fold cross-validation:")
    print(classification_report(labels, preds, labels=list(LEVELS), digits=3, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted):", list(LEVELS))
    print(np.array2string(confusion_matrix(labels, preds, labels=list(LEVELS))))

    model = DifficultyModel().fit(texts, labels)
    model.save(args.out)
    print(f"\nSaved model to {args.out}")


if __name__ == "__main__":
    main()
