# V1 App Plan

## Goal

Evolve this repo from a local prototype into a real single-user iOS-first app with:

- a persisted coffee catalogue
- persisted review and recommendation history
- a mobile-shaped API
- a lightweight admin/import path for adding coffees
- a deployment path that stays cheap for 1 to 5 users

The first production user is a single user, but the data model should support multiple users without a major refactor.

## Product Decisions

- Primary client: `SwiftUI` iOS app
- Backend: `FastAPI`
- Database: `Supabase Postgres`
- Hosting: `Vercel`
- LLM provider: `OpenAI`
- Initial app mode: single user
- Future compatibility: multi-user by including `user_id` from day one

## Architecture

### System shape

1. `SwiftUI` iOS app for the primary user experience
2. `FastAPI` backend for catalogue, review, and recommendation APIs
3. `Supabase Postgres` as the source of truth for app data
4. `OpenAI` for bounded LLM tasks only

### Source of truth

Persist the following in Postgres:

- canonical coffee catalogue records
- source page text and extraction metadata
- derived sensory profiles
- derived embeddings
- user review events
- recommendation runs and results

Do not treat in-memory session state as durable application state.

## Domain Model

### Core entities

- `users`
- `catalogue_coffees`
- `coffee_sources`
- `sensory_profiles`
- `embeddings`
- `review_events`
- `recommendation_runs`
- `recommendation_items`

### Table intent

#### `users`

Purpose:
- supports single-user now
- avoids a future single-user-only schema

Suggested fields:
- `id`
- `email`
- `display_name`
- `created_at`
- `updated_at`

#### `catalogue_coffees`

Purpose:
- canonical coffee records shown in the app
- stable coffee identity independent of extraction reruns

Suggested fields:
- `id`
- `name`
- `roaster`
- `origin_country`
- `region`
- `producer`
- `farm`
- `process_primary`
- `process_detail`
- `variety_text`
- `roast_level`
- `tasting_notes_json`
- `description`
- `price_minor`
- `currency`
- `weight_g`
- `is_active`
- `created_at`
- `updated_at`

Notes:
- keep `process_primary` normalized
- keep `process_detail` for richer specialty coffee process language
- keep arrays as JSON for v1 where query needs are modest

#### `coffee_sources`

Purpose:
- stores provenance and raw source material used to build a catalogue coffee
- supports re-extraction and debugging

Suggested fields:
- `id`
- `coffee_id`
- `source_type`
- `source_url`
- `raw_text`
- `raw_html_path`
- `scraped_at`
- `parser_version`
- `extraction_model`
- `extraction_status`
- `notes`
- `created_at`

Notes:
- `raw_text` lives in Postgres for v1
- `raw_html_path` is optional and can remain null until object storage is needed

#### `sensory_profiles`

Purpose:
- stores derived structured sensory vectors
- supports regeneration when extraction prompts or models improve

Suggested fields:
- `id`
- `coffee_id`
- `profile_version`
- `acidity`
- `sweetness`
- `body`
- `bitterness`
- `fruitiness`
- `chocolate_nutty`
- `floral`
- `funky_fermented`
- `roasty`
- `clean_cup`
- `confidence`
- `evidence_json`
- `model_name`
- `created_at`

#### `embeddings`

Purpose:
- stores vectors independently from coffee facts
- supports re-embedding without mutating catalogue metadata

Suggested fields:
- `id`
- `coffee_id`
- `embedding_model`
- `vector_json`
- `created_at`

#### `review_events`

Purpose:
- stores user tasting/review actions over time
- acts as the durable user preference history

Suggested fields:
- `id`
- `user_id`
- `coffee_id`
- `review_text`
- `overall_score`
- `parsed_review_json`
- `created_at`

#### `recommendation_runs`

Purpose:
- stores each recommendation request as a durable artifact
- supports history, debugging, and future evaluation

Suggested fields:
- `id`
- `user_id`
- `seed_review_event_id`
- `algorithm_version`
- `created_at`

#### `recommendation_items`

Purpose:
- stores coffees returned in a given recommendation run

Suggested fields:
- `id`
- `recommendation_run_id`
- `coffee_id`
- `rank`
- `score`
- `debug_json`

## API Plan

Design APIs for the mobile app rather than the current prototype UI.

### User-facing endpoints

- `GET /me`
- `GET /coffees`
- `GET /coffees/{id}`
- `POST /reviews`
- `GET /reviews`
- `POST /recommendations`
- `GET /recommendations/{run_id}`

### Admin/import endpoints

These are for the maintainer, not the end user.

- `POST /admin/coffees`
- `POST /admin/coffees/import`
- `POST /admin/coffees/from-url`
- `POST /admin/coffees/{id}/reextract`

### API design principles

- persist review and recommendation history in the database
- recommendation generation can remain synchronous in v1
- keep request and response models explicit rather than using `dict[str, Any]` for core flows
- make room for a future auth layer even if v1 runs with a single user

## LLM Boundaries

Keep LLM usage only where it creates real product value.

### Keep LLMs for

- extracting structured facts from messy coffee source text
- deriving sensory profiles from metadata and description
- parsing free-text user reviews into structured preference signals

### Do not use LLMs for

- ordinary CRUD
- catalogue retrieval
- session persistence
- recommendation history storage
- basic filtering and sorting

### Initial model choice

- factual extraction: `gpt-5.4-mini`
- sensory extraction: `gpt-5.4-mini`
- review parsing: `gpt-5.4-mini`

`gpt-5.4-nano` can remain available later for cheaper secondary tasks if needed.

## Data and Ingestion Plan

### Current problems to fix

- parser misses rich fields present in source text
- process extraction is too brittle
- CSV serialization of enum lists is poor
- many catalogue fields are currently under-populated

### V1 ingestion approach

1. ingest raw source text
2. parse deterministic fields first
3. use the LLM to fill missing or ambiguous fields
4. generate sensory profile
5. generate embedding
6. validate and persist the final coffee record

### Provenance requirements

Track enough metadata to debug extraction later:

- parser version
- extraction model
- profile version
- scrape timestamp
- extraction status

## Frontend Plan

The first real client is an iOS app built with `SwiftUI`.

### Initial screens

1. Home
- recent coffees
- quick path to submit a review
- recent recommendations

2. Coffee detail
- factual metadata
- tasting notes
- optional debug/provenance hidden from normal users

3. Review screen
- select a coffee
- enter free-text review
- submit review

4. Recommendations screen
- ranked recommendations
- short match rationale

5. History
- prior reviews
- prior recommendation runs

### Non-goals for v1

- user-facing coffee creation
- polished admin CMS
- tablet-specific UX
- Android client

## Hosting Plan

### V1 stack

- backend API: `Vercel`
- database and auth: `Supabase`
- secrets: provider-managed environment variables

### Object storage

Not required on day one.

Use Postgres for:

- structured app data
- raw extracted page text

Add object storage later only if needed for:

- raw HTML archives
- screenshots
- uploaded files
- large historical scrape artifacts

## Implementation Phases

### Phase 1: Persistence foundation

1. add database configuration
2. add migrations
3. create the core tables
4. add a repository or service layer for persisted entities

### Phase 2: Backend refactor

1. replace in-memory review session flows with persisted review history
2. persist recommendation runs and items
3. redesign API models for app use cases
4. keep recommendation logic but move it behind persisted data access

### Phase 3: Ingestion hardening

1. improve deterministic parsing
2. add provenance/version fields
3. persist source text and extraction outputs in the database
4. switch extraction defaults to `gpt-5.4-mini`

### Phase 4: Mobile client

1. scaffold the `SwiftUI` app
2. implement catalogue browsing
3. implement review submission
4. implement recommendation display
5. implement review and recommendation history

### Phase 5: Deployment

1. provision Supabase project
2. provision Vercel project
3. set environment variables
4. deploy backend
5. connect iOS app to deployed API

## Things We Intentionally Ignore For V1

- scheduled scraping
- background queues
- raw HTML archival
- advanced multi-user UX
- polished internal tooling
- analytics pipeline
- Redis or caching layer
- push notifications

## Immediate Next Step

Implement the persistence foundation first:

1. choose the database library and migration tool
2. add the initial schema
3. refactor the backend to use persisted review and recommendation data
