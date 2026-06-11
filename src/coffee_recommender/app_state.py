from __future__ import annotations

from typing import Any

from .review_session import create_review_session


DEFAULT_INPUT_MODE = "Catalogue coffee"


def _has_state_key(session_state: Any, key: str) -> bool:
    try:
        return key in session_state
    except TypeError:
        return hasattr(session_state, key)


def initialise_review_state(session_state: Any) -> None:
    if not _has_state_key(session_state, "review_session"):
        session_state.review_session = create_review_session()
    if not _has_state_key(session_state, "url_reviewed_coffee"):
        session_state.url_reviewed_coffee = None
    if not _has_state_key(session_state, "url_reviewed_source"):
        session_state.url_reviewed_source = ""
    if not _has_state_key(session_state, "input_mode"):
        session_state.input_mode = DEFAULT_INPUT_MODE


def reset_review_history(session_state: Any) -> None:
    session_state.review_session = create_review_session()
