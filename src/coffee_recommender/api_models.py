from __future__ import annotations

from typing import Literal
from typing import Any
from typing import cast
from datetime import datetime

from pydantic import BaseModel, Field

from .landscape import CoffeeFeatures, ReviewEvent


class ChangeRequestPayload(BaseModel):
    direction: str
    strength: float
    target_value: float | None = None
    adjustment: float | None = None


class AttributeOpinionPayload(BaseModel):
    sentiment: str
    strength: float


class ReviewEventPayload(BaseModel):
    coffee_id: str
    overall: float = 0.0
    change_requests: dict[str, ChangeRequestPayload] = Field(default_factory=dict)
    attribute_opinions: dict[str, AttributeOpinionPayload] = Field(default_factory=dict)


class CoffeeFeaturesPayload(BaseModel):
    coffee_id: str
    name: str | None = None
    sensory: dict[str, float]
    process: dict[str, float]
    embedding: list[float]


class RecommendationPayload(BaseModel):
    coffee_id: str
    name: str | None = None
    score: float
    temperature: float
    roaster: str | None = None
    origin_country: str | None = None
    producer: str | None = None
    process: str | None = None
    tasting_notes: list[str] = Field(default_factory=list)
    source_url: str | None = None
    debug: dict[str, Any] = Field(default_factory=dict)


class ReviewSessionPayload(BaseModel):
    review_events: list[ReviewEventPayload] = Field(default_factory=list)
    reviewed_feature_overrides: dict[str, CoffeeFeaturesPayload] = Field(default_factory=dict)
    last_event: ReviewEventPayload | None = None
    last_recommendations: list[RecommendationPayload] = Field(default_factory=list)


class CatalogueCoffeeSummary(BaseModel):
    coffee_id: str
    name: str | None = None
    roaster: str | None = None
    origin_country: str | None = None
    process: str | None = None
    tasting_notes: list[str] = Field(default_factory=list)


class CoffeeDetailPayload(BaseModel):
    coffee_id: str
    name: str | None = None
    roaster: str | None = None
    origin_country: str | None = None
    region: str | None = None
    producer: str | None = None
    farm: str | None = None
    process: str | None = None
    roast_level: str | None = None
    tasting_notes: list[str] = Field(default_factory=list)
    description: str | None = None
    weight_g: int | None = None
    price: float | None = None
    currency: str | None = None
    source_url: str | None = None
    features: CoffeeFeaturesPayload


class ReviewedCoffeeDetails(BaseModel):
    features: CoffeeFeaturesPayload
    metadata: dict[str, Any] | None = None
    sensory: dict[str, Any] | None = None
    source_type: Literal["catalogue", "external_url"]
    normalized_url: str | None = None


class ReviewHistoryItemPayload(BaseModel):
    review_id: int
    coffee_id: str
    review_text: str
    overall: float = 0.0
    created_at: datetime


class RecommendationRunPayload(BaseModel):
    run_id: int
    seed_review_event_id: int | None = None
    algorithm_version: str
    created_at: datetime
    recommendations: list[RecommendationPayload] = Field(default_factory=list)


class ProcessUrlRequest(BaseModel):
    url: str


class SubmitReviewRequest(BaseModel):
    review_text: str
    reviewed_coffee: ReviewedCoffeeDetails
    top_k: int = Field(default=5, ge=1, le=10)


class SubmitReviewResponse(BaseModel):
    event: ReviewEventPayload
    review_session: ReviewSessionPayload
    recommendations: list[RecommendationPayload]


class LandscapeResponse(BaseModel):
    figure: dict[str, Any] | None = None
    message: str | None = None


def coffee_features_to_payload(features: CoffeeFeatures) -> CoffeeFeaturesPayload:
    return CoffeeFeaturesPayload(
        coffee_id=features.coffee_id,
        name=features.name,
        sensory={key: float(value) for key, value in features.sensory.items()},
        process={key: float(value) for key, value in features.process.items()},
        embedding=[float(value) for value in features.embedding],
    )


def payload_to_coffee_features(payload: CoffeeFeaturesPayload) -> CoffeeFeatures:
    return CoffeeFeatures(
        coffee_id=payload.coffee_id,
        name=payload.name,
        sensory={key: float(value) for key, value in payload.sensory.items()},
        process={key: float(value) for key, value in payload.process.items()},
        embedding=[float(value) for value in payload.embedding],
    )


def review_event_to_payload(event: ReviewEvent) -> ReviewEventPayload:
    return ReviewEventPayload.model_validate(event)


def payload_to_review_event(payload: ReviewEventPayload) -> ReviewEvent:
    return cast(ReviewEvent, payload.model_dump(mode="python"))


def recommendation_to_payload(recommendation: dict[str, Any]) -> RecommendationPayload:
    return RecommendationPayload.model_validate(recommendation)
