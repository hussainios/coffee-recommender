from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from profile import PROFILE_DIMENSIONS, PROCESS_PREFERENCE_DIMENSIONS, UserProfile


class Recommendation(dict):
    pass


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_frame(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _infer_sensory_from_metadata(row: pd.Series) -> dict[str, object]:
    process = str(row.get("process", "")).lower()
    roast = str(row.get("roast_level", "")).lower()
    notes = " ".join(
        str(value).lower()
        for value in (
            row.get("tasting_notes", ""),
            row.get("description", ""),
            row.get("variety", ""),
            row.get("name", ""),
        )
    )

    sensory = {dimension: 0.0 for dimension in PROFILE_DIMENSIONS}
    sensory["process"] = process

    if "natural" in process or "anaerobic" in process or "fermented" in process:
        sensory["funky_fermented"] = 0.7
        sensory["fruitiness"] = 0.5
        sensory["sweetness"] = 0.4

    if "washed" in process:
        sensory["clean_cup"] = 0.7
        sensory["floral"] = 0.2
        sensory["acidity"] = 0.4

    if "honey" in process:
        sensory["sweetness"] = max(sensory["sweetness"], 0.5)
        sensory["body"] = 0.3

    if "light" in roast:
        sensory["acidity"] = max(sensory["acidity"], 0.5)
        sensory["roasty"] = 0.0
    elif "medium" in roast:
        sensory["body"] = max(sensory["body"], 0.4)
    elif "dark" in roast:
        sensory["roasty"] = 0.8
        sensory["bitterness"] = 0.5

    if any(term in notes for term in ("chocolate", "cocoa", "praline", "hazelnut", "almond")):
        sensory["chocolate_nutty"] = 0.7
    if any(term in notes for term in ("sweet", "honey", "caramel", "nougat")):
        sensory["sweetness"] = max(sensory["sweetness"], 0.7)
    if any(term in notes for term in ("berry", "berries", "fruit", "citrus", "apple", "stone fruit", "tropical")):
        sensory["fruitiness"] = max(sensory["fruitiness"], 0.7)
    if any(term in notes for term in ("jasmine", "rose", "floral", "tea")):
        sensory["floral"] = max(sensory["floral"], 0.6)
    if any(term in notes for term in ("funk", "wine", "ferment", "boozy", "anaerobic")):
        sensory["funky_fermented"] = max(sensory["funky_fermented"], 0.8)

    return sensory


def _coerce_profile(profile: UserProfile | dict[str, float]) -> dict[str, float]:
    dimensions = (*PROFILE_DIMENSIONS, *PROCESS_PREFERENCE_DIMENSIONS)
    return {dimension: _coerce_float(profile.get(dimension, 0.0)) for dimension in dimensions}


def _coerce_sensory_row(row: pd.Series) -> dict[str, float]:
    return {dimension: _coerce_float(row.get(dimension, 0.0)) for dimension in PROFILE_DIMENSIONS}


def score_coffee(
    profile: UserProfile | dict[str, float],
    sensory: dict[str, object],
) -> tuple[float, dict[str, float]]:
    profile_values = _coerce_profile(profile)
    sensory_values = {
        dimension: _coerce_float(sensory.get(dimension, 0.0))
        for dimension in PROFILE_DIMENSIONS
    }
    process_values = {dimension: 0.0 for dimension in PROCESS_PREFERENCE_DIMENSIONS}

    process = str(sensory.get("process", "")).lower()
    if "washed" in process:
        process_values["process_washed"] = 1.0
    if "natural" in process:
        process_values["process_natural"] = 1.0
    if "honey" in process:
        process_values["process_honey"] = 1.0
    if "anaerobic" in process:
        process_values["process_anaerobic"] = 1.0
    if "cofermented" in process:
        process_values["process_cofermented"] = 1.0

    breakdown = {
        dimension: profile_values[dimension] * sensory_values[dimension]
        for dimension in PROFILE_DIMENSIONS
        if profile_values[dimension] != 0.0 and sensory_values[dimension] != 0.0
    }
    breakdown.update(
        {
            dimension: profile_values[dimension] * process_values[dimension]
            for dimension in PROCESS_PREFERENCE_DIMENSIONS
            if profile_values[dimension] != 0.0 and process_values[dimension] != 0.0
        }
    )
    score = round(sum(breakdown.values()), 6)
    return score, breakdown


def recommend_coffees(
    user_profile: UserProfile | dict[str, float],
    coffees_path: str | Path,
    sensory_path: str | Path | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    coffees = _load_frame(coffees_path)
    recommendations: list[dict[str, Any]] = []

    sensory_lookup: dict[str, dict[str, float]] = {}
    if sensory_path is not None and Path(sensory_path).exists():
        sensory = _load_frame(sensory_path)
        for _, row in sensory.iterrows():
            sensory_lookup[str(row.get("coffee_id"))] = _coerce_sensory_row(row)

    for _, row in coffees.iterrows():
        coffee_id = str(row.get("coffee_id"))
        sensory_values = sensory_lookup.get(coffee_id) or _infer_sensory_from_metadata(row)
        score, breakdown = score_coffee(user_profile, sensory_values)
        recommendations.append(
            {
                "coffee_id": coffee_id,
                "name": row.get("name"),
                "score": score,
                "score_breakdown": breakdown,
                "explanation": [
                    f"Matches your preference for {dimension.replace('_', ' ')}"
                    if value > 0
                    else f"Low penalty for {dimension.replace('_', ' ')}"
                    for dimension, value in breakdown.items()
                ],
            }
        )

    recommendations.sort(key=lambda item: item["score"], reverse=True)
    return recommendations[:top_k]
