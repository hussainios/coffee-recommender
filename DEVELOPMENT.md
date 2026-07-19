# Development

This document is for local setup, operational workflows, and project maintenance.

## Environment

Copy `.env.example` to `.env` and set the values you need.

Common settings:

```text
OPENAI_API_KEY=...
COFFEE_RECOMMENDER_API_BASE_URL=http://127.0.0.1:8000
COFFEE_RECOMMENDER_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
COFFEE_RECOMMENDER_DATABASE_URL=postgresql+psycopg://postgres:[password]@db.[project-ref].supabase.co:5432/postgres
```

Optional catalogue path overrides:

```text
COFFEE_RECOMMENDER_COFFEES_PATH=data/processed/coffees.csv
COFFEE_RECOMMENDER_SENSORY_PATH=data/processed/coffee_sensory_vectors.csv
COFFEE_RECOMMENDER_EMBEDDINGS_PATH=data/processed/coffee_embeddings.csv
```

## Running Locally

Backend:

```bash
uvicorn coffee_recommender.api:app --reload
```

Frontend:

```bash
cd frontend && npm install && npm run dev
```

Streamlit debug UI:

```bash
streamlit run app.py
```

## Database

The repo includes:

- `SQLAlchemy 2` models under `src/coffee_recommender/db/`
- `Alembic` migrations under `migrations/`
- an optional database-aware health check on `GET /health`

### Supabase setup

1. Create a Supabase project.
2. Open `Project Settings`.
3. Open the `Database` section.
4. Copy the Postgres connection string.
5. Put it into `.env` as `COFFEE_RECOMMENDER_DATABASE_URL`.

### Install backend dependencies

```bash
.venv/bin/python -m pip install -r requirements.txt
```

### Run the first migration

```bash
.venv/bin/alembic upgrade head
```

If `alembic` is already on your shell path, this also works:

```bash
alembic upgrade head
```

### Health endpoint behavior

`GET /health` returns both API and database status.

Example when the database is configured and reachable:

```json
{
  "status": "ok",
  "database": {
    "status": "ok"
  }
}
```

If `COFFEE_RECOMMENDER_DATABASE_URL` is not set:

```json
{
  "status": "ok",
  "database": {
    "status": "not_configured"
  }
}
```

## Data Pipeline

Build the local processed catalogue with:

```bash
.venv/bin/python -m coffee_recommender.process_data.build_dataset
```

Artifacts:

- `data/processed/coffees.csv`
- `data/processed/coffee_sensory_vectors.csv`
- `data/processed/coffee_embeddings.csv`

## Checks

```bash
.venv/bin/python -m pytest
.venv/bin/python -m mypy src app.py
cd frontend && npm run build
```
