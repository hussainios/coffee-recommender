# Repo Conventions

## Goals

This repo is moving from prototype code to a real application. The conventions below are meant to
keep that transition lightweight while still enforcing clear boundaries and maintainable types.

## Core Principles

- Use explicit types at API and persistence boundaries.
- Keep database models separate from API models.
- Keep domain logic separate from transport and persistence details.
- Prefer small, typed abstractions over free-form dictionaries.
- Use LLMs only where they add product value.

## Python Tooling

- `Pydantic` for API schemas and settings
- `SQLAlchemy 2` for database models
- `Alembic` for schema migrations
- `Ruff` for linting and formatting
- `Mypy` for static typing
- `Pytest` for tests

## Module Boundaries

### API models

Put request and response models in Pydantic classes.

Rules:
- avoid `dict[str, Any]` for core app payloads
- prefer explicit nested models
- use JSON-like fallback fields only when the shape is intentionally flexible

### Database models

Put persistence models under `coffee_recommender.db`.

Rules:
- model columns explicitly
- keep timestamps and foreign keys present on durable entities
- do not return raw ORM models directly from API routes

### Services

Services should orchestrate application logic.

Rules:
- accept typed inputs
- return typed outputs
- avoid embedding HTTP or ORM-specific logic deep in business logic

## Typing Rules

- prefer `str | None` only when null is a real business case
- prefer enums only when the value set is stable
- use flexible text fields for evolving source metadata such as process detail
- avoid `Any` unless there is a clear justification
- use `list[str]`, `dict[str, float]`, and typed models instead of unstructured payloads

## Persistence Rules

- Postgres is the system of record
- raw source text may be stored in Postgres in v1
- derived artifacts such as sensory profiles and embeddings should be versioned
- keep `user_id` on user-owned data from day one even in single-user mode

## LLM Rules

Use LLMs for:
- coffee data extraction from messy source text
- sensory profile derivation
- free-text review parsing

Do not use LLMs for:
- CRUD
- catalogue retrieval
- persistence
- ranking history storage
- basic filtering or sorting

## Migration Rules

- all durable schema changes go through Alembic
- avoid manual database drift
- keep migrations small and reviewable

## Testing Expectations

For backend changes:
- add or update unit tests when behavior changes
- keep parsing and recommendation logic covered
- add persistence tests as the database layer grows

## Style Expectations

- prefer ASCII unless the file already relies on Unicode
- keep functions focused
- add comments only when they clarify non-obvious logic
- avoid framework-heavy abstractions until the app truly needs them
