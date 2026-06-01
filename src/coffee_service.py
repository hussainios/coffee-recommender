from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app_state import append_review_event, build_scoring_features
from landscape import CoffeeFeatures, ReviewEvent, load_feature_index, recommend_from_landscape
from parse_review import parse_review_event
from reviewed_coffee_url import ReviewedCoffeeFromUrl, normalise_source_url, prepare_reviewed_coffee_from_url


@dataclass(frozen=True)
class CatalogueData:
    coffees: pd.DataFrame
    features: dict[str, CoffeeFeatures]
    data_paths_key: tuple[str, str, str]


@dataclass(frozen=True)
class ReviewedCoffeeSelection:
    features: CoffeeFeatures
    metadata: dict[str, Any] | None
    sensory: dict[str, Any] | None
    is_temporary: bool


@dataclass(frozen=True)
class ReviewSubmissionResult:
    event: ReviewEvent
    recommendations: list[dict[str, Any]]
    scoring_features: dict[str, CoffeeFeatures]


def load_catalogue(
    coffees_path: str | Path,
    sensory_path: str | Path,
    embeddings_path: str | Path,
) -> CatalogueData:
    coffees = pd.read_csv(coffees_path)
    features = load_feature_index(coffees_path, sensory_path, embeddings_path)
    return CatalogueData(
        coffees=coffees,
        features=features,
        data_paths_key=(str(coffees_path), str(sensory_path), str(embeddings_path)),
    )


def build_coffee_options(coffees: pd.DataFrame) -> dict[str, str]:
    return {
        f"{row['name']} ({row['coffee_id']})": str(row["coffee_id"])
        for _, row in coffees.sort_values("name").iterrows()
    }


def select_catalogue_reviewed_coffee(
    catalogue: CatalogueData,
    coffee_id: str,
) -> ReviewedCoffeeSelection:
    reviewed_coffee = catalogue.features[coffee_id]
    metadata = catalogue.coffees.loc[
        catalogue.coffees["coffee_id"].astype(str) == coffee_id
    ].iloc[0].to_dict()
    return ReviewedCoffeeSelection(
        features=reviewed_coffee,
        metadata=metadata,
        sensory=None,
        is_temporary=False,
    )


def normalise_optional_url(url_value: str) -> str:
    return normalise_source_url(url_value) if url_value else ""


def selection_from_url_reviewed_coffee(
    reviewed: ReviewedCoffeeFromUrl,
) -> ReviewedCoffeeSelection:
    return ReviewedCoffeeSelection(
        features=reviewed.features,
        metadata=reviewed.coffee.model_dump(mode="json"),
        sensory=reviewed.sensory.model_dump(mode="json"),
        is_temporary=True,
    )


def get_cached_url_selection(
    url_value: str,
    cached: ReviewedCoffeeFromUrl | None,
    cached_source: str,
) -> ReviewedCoffeeSelection | None:
    if cached is None:
        return None

    try:
        normalized_url = normalise_optional_url(url_value)
    except ValueError:
        return None

    if normalized_url and normalized_url == cached_source:
        return selection_from_url_reviewed_coffee(cached)

    return None


def prepare_url_selection(url_value: str) -> ReviewedCoffeeFromUrl:
    return prepare_reviewed_coffee_from_url(url_value)


def submit_review(
    *,
    session_state: Any,
    review_text: str,
    reviewed_coffee: CoffeeFeatures | None,
    catalogue_features: dict[str, CoffeeFeatures],
    top_k: int,
    is_temporary: bool,
) -> ReviewSubmissionResult:
    if reviewed_coffee is None:
        raise ValueError("Select or process a reviewed coffee before running the recommender.")

    event = parse_review_event(review_text, reviewed_coffee)
    append_review_event(
        session_state,
        event,
        reviewed_coffee,
        is_temporary=is_temporary,
    )
    scoring_features = build_scoring_features(
        catalogue_features,
        session_state.reviewed_feature_overrides,
    )
    recommendations = recommend_from_landscape(scoring_features, session_state.review_events, top_k=top_k)
    return ReviewSubmissionResult(
        event=event,
        recommendations=recommendations,
        scoring_features=scoring_features,
    )
