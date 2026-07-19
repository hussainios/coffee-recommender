# Coffee Recommender

Coffee Recommender is an app project for turning coffee tasting feedback into personalized recommendations.

The current codebase started as a prototype for exploring a simple but opinionated question:

How can free-text tasting reactions like "too acidic", "clean but thin", or "more floral please"
be converted into structured signals that improve the next recommendation?

## What The Project Does

The project combines three layers:

1. a coffee catalogue built from scraped product pages
2. a recommendation model that scores coffees based on review feedback
3. application surfaces for reviewing coffees and returning ranked suggestions

At the moment the repo includes:

- a FastAPI backend
- a React prototype frontend
- a Streamlit debug interface
- a data-processing pipeline for catalogue extraction, sensory inference, and embeddings

The long-term direction is an iOS-first app with persisted users, reviews, and recommendations.

## Recommendation Model

The model is intentionally hybrid rather than purely LLM-driven.

### 1. Factual coffee metadata

Each coffee starts as a structured product record containing things like:

- name
- origin
- producer
- process
- variety
- roast level
- tasting notes

This layer is meant to represent source facts rather than subjective interpretation.

### 2. Derived sensory representation

Each coffee is then mapped into a normalized sensory vector, including dimensions such as:

- acidity
- sweetness
- body
- bitterness
- fruitiness
- chocolate or nutty character
- floral character
- funky or fermented character
- roastiness
- clean-cup character

These values are derived from product text and tasting-note evidence so that coffees can be
compared in a shared feature space.

### 3. Embedding representation

The system also creates an embedding for each coffee so that broader semantic similarity can be
used alongside the hand-shaped sensory dimensions.

This gives the model two useful views of a coffee:

- explicit interpretable dimensions
- dense semantic similarity

### 4. Review parsing

When a user writes a review, the system tries to convert natural-language feedback into structured
directional signals.

Examples:

- "less acidic" should decrease the preferred acidity direction
- "more floral" should increase floral preference
- "too bitter but otherwise nice" should mix a negative constraint with a positive overall signal

### 5. Ranking

The recommender then scores candidate coffees by comparing:

- the reviewed coffee
- the interpreted user preference shift
- the sensory and embedding space of the catalogue

The goal is not just "find similar coffees" but "move in the right direction from the last cup."

## Design Philosophy

The project is trying to keep a useful boundary between:

- deterministic application logic
- model-derived representations
- source data provenance

That boundary matters because this is not just a chatbot layer over coffee data. The aim is a real
application where data quality, repeatability, and explainability matter.

## Current Status

The repo is now moving from prototype code toward an app architecture with:

- typed schemas
- a database-backed persistence layer
- migration support
- clearer separation between app, data, and model concerns

## Developer Docs

Operational setup, local run instructions, database setup, and migration steps live outside the
README so this file can stay focused on what the project is.

See:

- [REPO_CONVENTIONS.md](./REPO_CONVENTIONS.md)
- [DEVELOPMENT.md](./DEVELOPMENT.md)
