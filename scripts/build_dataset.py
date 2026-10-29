"""Collect real labeled issues from GitHub to grow the training set.

Maintainers often tag issues with difficulty labels. This script searches for
issues carrying those labels and writes them to a CSV (title, body, label, url)
that scripts/train.py can use with --extra.

Usage:
    GITHUB_TOKEN=... python -m scripts.build_dataset --per-query 100 --out data/github_issues.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

from app.difficulty import level_from_labels
from app.github_client import GitHubClient, GitHubError

QUERIES = {
    "beginner": ['label:"good first issue"', 'label:"difficulty: easy"', 'label:"first-timers-only"'],
    "intermediate": ['label:"difficulty: medium"', 'label:"difficulty/medium"', 'label:"intermediate"'],
    "advanced": ['label:"difficulty: hard"', 'label:"difficulty/hard"', 'label:"expert"'],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-query", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("data/github_issues.csv"))
    args = parser.parse_args()

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("Tip: set GITHUB_TOKEN — unauthenticated search allows only 10 requests per minute.")
    client = GitHubClient(token)

    rows, seen = [], set()
    for expected, queries in QUERIES.items():
        for q in queries:
            try:
                issues = client.search_issues(f"is:issue {q}", limit=args.per_query)
            except GitHubError as exc:
                print(f"  skipped {q!r}: {exc}")
                continue
            kept = 0
            for issue in issues:
                # Keep only issues whose labels agree with the bucket we searched for.
                if issue.id in seen or level_from_labels(issue.labels) != expected:
                    continue
                seen.add(issue.id)
                rows.append({"title": issue.title, "body": issue.body[:1500], "label": expected, "url": issue.html_url})
                kept += 1
            print(f"{expected:>12} {q:<32} +{kept}")
            time.sleep(2 if token else 7)  # stay under the search rate limit

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "body", "label", "url"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} issues to {args.out}")


if __name__ == "__main__":
    main()
