from __future__ import annotations

from .api_models import (
    CoffeeFeaturesPayload,
    RecommendationPayload,
    ReviewEventPayload,
    ReviewSessionPayload,
)


def create_review_session() -> ReviewSessionPayload:
    return ReviewSessionPayload()


def append_review_to_session(
    review_session: ReviewSessionPayload,
    event: ReviewEventPayload,
    reviewed_coffee: CoffeeFeaturesPayload,
    *,
    is_temporary: bool,
    recommendations: list[RecommendationPayload],
) -> ReviewSessionPayload:
    reviewed_feature_overrides = dict(review_session.reviewed_feature_overrides)
    if is_temporary:
        reviewed_feature_overrides[reviewed_coffee.coffee_id] = reviewed_coffee

    return ReviewSessionPayload(
        review_events=[*review_session.review_events, event],
        reviewed_feature_overrides=reviewed_feature_overrides,
        last_event=event,
        last_recommendations=recommendations,
    )
