from __future__ import annotations

from typing import Any

from .landscape import CoffeeFeatures, ReviewEvent


DEFAULT_INPUT_MODE = "Catalogue coffee"


def _has_state_key(session_state: Any, key: str) -> bool:
    try:
        return key in session_state
    except TypeError:
        return hasattr(session_state, key)


def initialise_review_state(session_state: Any) -> None:
    if not _has_state_key(session_state, "last_event"):
        session_state.last_event = None
    if not _has_state_key(session_state, "last_recommendations"):
        session_state.last_recommendations = []
    if not _has_state_key(session_state, "review_events"):
        session_state.review_events = []
    if not _has_state_key(session_state, "reviewed_feature_overrides"):
        session_state.reviewed_feature_overrides = {}
    if not _has_state_key(session_state, "url_reviewed_coffee"):
        session_state.url_reviewed_coffee = None
    if not _has_state_key(session_state, "url_reviewed_source"):
        session_state.url_reviewed_source = ""
    if not _has_state_key(session_state, "input_mode"):
        session_state.input_mode = DEFAULT_INPUT_MODE
    if not _has_state_key(session_state, "data_paths_key"):
        session_state.data_paths_key = None


def reset_review_history(session_state: Any) -> None:
    session_state.review_events = []
    session_state.reviewed_feature_overrides = {}
    session_state.last_event = None
    session_state.last_recommendations = []


def build_scoring_features(
    features: dict[str, CoffeeFeatures],
    reviewed_feature_overrides: dict[str, CoffeeFeatures],
) -> dict[str, CoffeeFeatures]:
    scoring_features = dict(features)
    scoring_features.update(reviewed_feature_overrides)
    return scoring_features


def append_review_event(
    session_state: Any,
    event: ReviewEvent,
    reviewed_coffee: CoffeeFeatures,
    *,
    is_temporary: bool,
) -> None:
    session_state.review_events.append(event)
    session_state.last_event = event
    if is_temporary:
        session_state.reviewed_feature_overrides[reviewed_coffee.coffee_id] = reviewed_coffee


def reset_review_history_if_data_paths_changed(
    session_state: Any,
    data_paths_key: tuple[str, str, str],
) -> None:
    if session_state.data_paths_key is None:
        session_state.data_paths_key = data_paths_key
    elif session_state.data_paths_key != data_paths_key:
        session_state.data_paths_key = data_paths_key
        reset_review_history(session_state)
