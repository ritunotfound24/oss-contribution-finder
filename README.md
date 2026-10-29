# firstPR — Open-Source Contribution Finder

Give it a GitHub username and it recommends open-source issues and projects that fit that person's skills and experience level.

It reads the user's public repositories to build a **contributor profile**, searches GitHub for open issues that maintainers have flagged for outside help, predicts each issue's **difficulty with a machine-learning classifier**, and ranks everything with an explainable scoring model.

Built with **Python**, **FastAPI**, **scikit-learn**, and the **GitHub REST API**.

## Features

- **Contributor profiles** — languages weighted by repository popularity and recency (recent, starred repos count more; forks and archived repos are ignored), top topics, and an estimated experience level
- **Issue search** — open, unassigned issues labeled `good first issue`, `help wanted` or `first-timers-only` in the user's top languages
- **Difficulty classifier** — TF-IDF + logistic regression predicts beginner / intermediate / advanced from the issue text, blended with maintainer labels when they exist
- **Explainable ranking** — every issue gets a 0–100 score with a per-feature breakdown and plain-language reasons
- **Project recommendations** — repositories with several good-fit issues, healthy activity and matching topics
- **Web UI and JSON API** — server-rendered pages plus interactive API docs at `/docs`
- **Offline demo mode** — bundled sample data so it runs without network access or a token

## How ranking works

Each issue gets six feature scores between 0 and 1 ([`app/ranker.py`](app/ranker.py)):

| Feature | Weight | What it measures |
|---|---|---|
| Language | 25% | How much of the user's code is in the repo's language |
| Similarity | 25% | TF-IDF cosine similarity between the user's repo descriptions/topics/bio and the issue + repo text |
| Difficulty fit | 20% | Predicted difficulty probabilities × a fit matrix for the user's level |
| Freshness | 10% | Exponential decay on last update (30-day half-life) |
| Availability | 10% | Zero if assigned; lower for long comment threads |
| Repo health | 10% | Stars (log-scaled) and recent pushes; zero for archived repos |

Projects are scored as `0.6 × mean(top 3 issue scores) + 0.2 × repo health + 0.2 × language fit`, plus a small bonus for each additional matching issue ([`app/recommender.py`](app/recommender.py)).

## The difficulty model

[`app/difficulty.py`](app/difficulty.py) trains a scikit-learn pipeline (word 1–2-gram TF-IDF → class-balanced logistic regression) on labeled issues. When an issue already carries a maintainer label such as `good first issue` or `difficulty: hard`, that label is blended 60/40 with the model's probabilities.

The repo ships with a small hand-labeled seed set (`data/seed_issues.csv`, 74 issues). 5-fold cross-validation on it gives:

| | precision | recall | f1 |
|---|---|---|---|
| beginner | 0.79 | 0.88 | 0.84 |
| intermediate | 0.58 | 0.60 | 0.59 |
| advanced | 0.63 | 0.52 | 0.57 |
| **accuracy** | | | **0.68** |

Beginner issues are easy to spot; intermediate vs advanced is harder with so little data. To train on real data:

```bash
export GITHUB_TOKEN=ghp_...                      # recommended for search rate limits
python -m scripts.build_dataset --per-query 100  # writes data/github_issues.csv
python -m scripts.train --extra data/github_issues.csv
```

`build_dataset.py` collects issues that maintainers have tagged with difficulty labels and keeps only those whose labels agree with the bucket being searched.

## Getting started

Requires Python 3.10+.

```bash
git clone https://github.com/ritunotfound24/oss-contribution-finder.git
cd oss-contribution-finder
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# optional: a token raises GitHub's limit from 60 to 5,000 requests/hour
export GITHUB_TOKEN=ghp_...        # Windows PowerShell: $env:GITHUB_TOKEN="ghp_..."

uvicorn app.main:app --reload
```

Open http://localhost:8000 and enter a username. The model trains automatically on first run (about a second) and is saved to `models/`.

To try it without network access, set `DEMO_MODE=1`.

### Docker

```bash
docker build -t firstpr .
docker run -p 8000:8000 -e GITHUB_TOKEN=ghp_... firstpr
```

## API

| Endpoint | Description |
|---|---|
| `GET /api/profile/{username}` | Contributor profile: languages, topics, estimated level |
| `GET /api/recommendations/{username}?level=&languages=&limit=` | Ranked issues and project recommendations |
| `GET /health` | Health check |

Example:

```bash
curl "http://localhost:8000/api/recommendations/octocat?level=beginner&languages=Python,TypeScript&limit=5"
```

## Tests

```bash
pytest
```

The tests run fully offline against the bundled demo data and mocked HTTP responses. They cover label parsing, the classifier, profile building, ranking invariants, project recommendations, the GitHub client (caching, errors, rate limits) and the API routes.

## Project structure

```
app/
  main.py            FastAPI routes (HTML + JSON)
  service.py         Orchestrates search → ranking → recommendations
  github_client.py   GitHub REST client with TTL cache; offline demo client
  profile.py         Contributor profile and level estimation
  difficulty.py      Difficulty classifier
  ranker.py          Issue scoring
  recommender.py     Project recommendations
  templates/, static/
data/
  seed_issues.csv    Hand-labeled training data
  demo/              Sample data for demo mode and tests
scripts/
  train.py           Cross-validated training
  build_dataset.py   Collect labeled issues from GitHub
tests/
```

## Ideas for later

- Sentence embeddings instead of TF-IDF for similarity
- Use the user's merged pull requests to refine their profile
- Email digest of new matching issues

## License

MIT
