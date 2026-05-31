from __future__ import annotations

from typing import TypedDict


class UserProfile(TypedDict, total=False):
    acidity: float
    sweetness: float
    body: float
    bitterness: float
    fruitiness: float
    chocolate_nutty: float
    floral: float
    funky_fermented: float
    roasty: float
    process_washed: float
    process_natural: float
    process_honey: float
    process_anaerobic: float
    process_cofermented: float


PROFILE_DIMENSIONS: tuple[str, ...] = (
    "acidity",
    "sweetness",
    "body",
    "bitterness",
    "fruitiness",
    "chocolate_nutty",
    "floral",
    "funky_fermented",
    "roasty",
)

PROCESS_PREFERENCE_DIMENSIONS: tuple[str, ...] = (
    "process_washed",
    "process_natural",
    "process_honey",
    "process_anaerobic",
    "process_cofermented",
)


def create_empty_profile() -> UserProfile:
    profile = {dimension: 0.0 for dimension in PROFILE_DIMENSIONS}
    profile.update({dimension: 0.0 for dimension in PROCESS_PREFERENCE_DIMENSIONS})
    return profile


def clip(value: float, minimum: float = -1.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def update_profile(
    profile: UserProfile,
    delta: dict[str, float],
    learning_rate: float = 0.3,
) -> UserProfile:
    updated = dict(profile)

    for dimension in PROFILE_DIMENSIONS:
        current = float(profile.get(dimension, 0.0))
        change = float(delta.get(dimension, 0.0))
        updated[dimension] = clip(current + learning_rate * change)

    for dimension in PROCESS_PREFERENCE_DIMENSIONS:
        current = float(profile.get(dimension, 0.0))
        change = float(delta.get(dimension, 0.0))
        updated[dimension] = clip(current + learning_rate * change)

    return updated
