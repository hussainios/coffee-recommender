from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .api_models import (
    CoffeeDetailPayload,
    CoffeeFeaturesPayload,
    RecommendationPayload,
    ReviewEventPayload,
    ReviewSessionPayload,
    coffee_features_to_payload,
    payload_to_coffee_features,
    payload_to_review_event,
    recommendation_to_payload,
    review_event_to_payload,
)
from .landscape import CoffeeFeatures, load_feature_index, recommend_from_landscape
from .parse_review import parse_review_event
from .review_session import append_review_to_session
from .reviewed_coffee_url import ReviewedCoffeeFromUrl, normalise_source_url, prepare_reviewed_coffee_from_url


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
    event: ReviewEventPayload
    recommendations: list[RecommendationPayload]
    review_session: ReviewSessionPayload
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


def build_coffee_detail_payload(
    selection: ReviewedCoffeeSelection,
) -> CoffeeDetailPayload:
    metadata = selection.metadata or {}
    tasting_notes = metadata.get("tasting_notes")
    if not isinstance(tasting_notes, list):
        tasting_notes = []

    return CoffeeDetailPayload(
        coffee_id=selection.features.coffee_id,
        name=selection.features.name,
        roaster=_optional_string(metadata.get("roaster")),
        origin_country=_optional_string(metadata.get("origin_country")),
        region=_optional_string(metadata.get("region")),
        producer=_optional_string(metadata.get("producer")),
        farm=_optional_string(metadata.get("farm")),
        process=_optional_string(metadata.get("process")),
        roast_level=_optional_string(metadata.get("roast_level")),
        tasting_notes=[str(note) for note in tasting_notes if str(note).strip()],
        description=_optional_string(metadata.get("description")),
        weight_g=_optional_int(metadata.get("weight_g")),
        price=_optional_float(metadata.get("price")),
        currency=_optional_string(metadata.get("currency")),
        source_url=_optional_string(metadata.get("source_url")),
        features=coffee_features_to_payload(selection.features),
    )


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
    review_session: ReviewSessionPayload,
    review_text: str,
    reviewed_coffee: CoffeeFeatures | None,
    catalogue_features: dict[str, CoffeeFeatures],
    top_k: int,
    is_external_url: bool,
) -> ReviewSubmissionResult:
    if reviewed_coffee is None:
        raise ValueError("Select or process a reviewed coffee before running the recommender.")

    event = review_event_to_payload(parse_review_event(review_text, reviewed_coffee))
    pending_session = append_review_to_session(
        review_session,
        event,
        coffee_features_to_payload(reviewed_coffee),
        is_external_url=is_external_url,
        recommendations=[],
    )
    scoring_features = build_scoring_features(
        catalogue_features,
        {
            coffee_id: payload_to_coffee_features(payload)
            for coffee_id, payload in pending_session.reviewed_feature_overrides.items()
        },
    )
    recommendations = [
        recommendation_to_payload(item)
        for item in recommend_from_landscape(
            scoring_features,
            [payload_to_review_event(review) for review in pending_session.review_events],
            top_k=top_k,
        )
    ]
    current_session = ReviewSessionPayload(
        review_events=pending_session.review_events,
        reviewed_feature_overrides=pending_session.reviewed_feature_overrides,
        last_event=event,
        last_recommendations=recommendations,
    )
    return ReviewSubmissionResult(
        event=event,
        recommendations=recommendations,
        review_session=current_session,
        scoring_features=scoring_features,
    )


def build_scoring_features(
    features: dict[str, CoffeeFeatures],
    reviewed_feature_overrides: dict[str, CoffeeFeatures],
) -> dict[str, CoffeeFeatures]:
    scoring_features = dict(features)
    scoring_features.update(reviewed_feature_overrides)
    return scoring_features


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
