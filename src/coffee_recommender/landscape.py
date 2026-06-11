from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

import pandas as pd

from .coffee_dimensions import PROCESS_DIMENSIONS, SENSORY_DIMENSIONS
from .schemas import CoffeeRecord, SensoryVector


ChangeDirection = Literal["higher", "lower"]
AttributeSentiment = Literal["liked", "disliked"]


class ChangeRequest(TypedDict, total=False):
    direction: ChangeDirection
    strength: float
    target_value: float
    adjustment: float


class AttributeOpinion(TypedDict, total=False):
    sentiment: AttributeSentiment
    strength: float


class ReviewEvent(TypedDict, total=False):
    coffee_id: str
    overall: float
    change_requests: dict[str, ChangeRequest]
    attribute_opinions: dict[str, AttributeOpinion]


@dataclass(frozen=True)
class CoffeeFeatures:
    coffee_id: str
    name: str | None
    sensory: dict[str, float]
    process: dict[str, float]
    embedding: list[float]


@dataclass(frozen=True)
class DistanceWeights:
    embedding: float = 0.55
    sensory: float = 0.35
    process: float = 0.10


@dataclass(frozen=True)
class LandscapeConfig:
    distance_weights: DistanceWeights = DistanceWeights()
    neighbor_rank: int | None = None
    target_kernel_at_neighbor: float = 0.6
    default_temperature: float = 0.25


def _clip(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _process_features(process: Any) -> dict[str, float]:
    text = str(process or "").lower()
    features = {dimension: 0.0 for dimension in PROCESS_DIMENSIONS}

    if "washed" in text:
        features["process_washed"] = 1.0
    if "natural" in text:
        features["process_natural"] = 1.0
    if "honey" in text:
        features["process_honey"] = 1.0
    if "anaerobic" in text or "carbonic" in text:
        features["process_anaerobic"] = 1.0
    if "cofermented" in text or "co-fermented" in text:
        features["process_cofermented"] = 1.0

    return features


def _sensory_from_row(row: pd.Series) -> dict[str, float]:
    return {
        dimension: _clip(_coerce_float(row.get(dimension, 0.0)))
        for dimension in SENSORY_DIMENSIONS
    }


def _embedding_from_row(row: pd.Series) -> list[float]:
    raw_embedding = row.get("embedding")
    if isinstance(raw_embedding, str):
        parsed = json.loads(raw_embedding)
    else:
        parsed = raw_embedding

    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"Invalid embedding for coffee_id: {row.get('coffee_id')}")

    return [float(value) for value in parsed]


def build_feature_index(
    coffees: pd.DataFrame,
    sensory: pd.DataFrame,
    embeddings: pd.DataFrame,
) -> dict[str, CoffeeFeatures]:
    sensory_lookup: dict[str, dict[str, float]] = {}
    for _, row in sensory.iterrows():
        sensory_lookup[str(row.get("coffee_id"))] = _sensory_from_row(row)

    embedding_lookup: dict[str, list[float]] = {}
    for _, row in embeddings.iterrows():
        embedding_lookup[str(row.get("coffee_id"))] = _embedding_from_row(row)

    features: dict[str, CoffeeFeatures] = {}
    for _, row in coffees.iterrows():
        coffee_id = str(row.get("coffee_id"))
        sensory_values = sensory_lookup.get(coffee_id)
        if sensory_values is None:
            raise KeyError(f"Missing sensory vector for coffee_id: {coffee_id}")
        embedding_values = embedding_lookup.get(coffee_id)
        if embedding_values is None:
            raise KeyError(f"Missing embedding for coffee_id: {coffee_id}")

        features[coffee_id] = CoffeeFeatures(
            coffee_id=coffee_id,
            name=row.get("name"),
            sensory=sensory_values,
            process=_process_features(row.get("process")),
            embedding=embedding_values,
        )

    return features


def build_single_coffee_features(
    coffee: CoffeeRecord,
    sensory: SensoryVector,
    embedding: list[float],
) -> CoffeeFeatures:
    sensory_payload = {
        dimension: _clip(_coerce_float(getattr(sensory, dimension, 0.0)))
        for dimension in SENSORY_DIMENSIONS
    }

    if len(embedding) == 0:
        raise ValueError(f"Missing embedding for coffee_id: {coffee.coffee_id}")

    return CoffeeFeatures(
        coffee_id=coffee.coffee_id,
        name=coffee.name,
        sensory=sensory_payload,
        process=_process_features(coffee.process.value if hasattr(coffee.process, "value") else coffee.process),
        embedding=[float(value) for value in embedding],
    )


def load_feature_index(
    coffees_path: str | Path,
    sensory_path: str | Path,
    embeddings_path: str | Path,
) -> dict[str, CoffeeFeatures]:
    coffees = pd.read_csv(coffees_path)
    sensory_file = Path(sensory_path)
    if not sensory_file.exists():
        raise FileNotFoundError(f"Sensory vector CSV not found: {sensory_file}")
    sensory = pd.read_csv(sensory_file)
    embeddings_file = Path(embeddings_path)
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Coffee embeddings CSV not found: {embeddings_file}")
    embeddings = pd.read_csv(embeddings_file)
    return build_feature_index(coffees, sensory, embeddings)


def _mean_squared_distance(
    left: dict[str, float],
    right: dict[str, float],
    dimensions: tuple[str, ...],
) -> float:
    if not dimensions:
        return 0.0
    total = sum((left.get(dimension, 0.0) - right.get(dimension, 0.0)) ** 2 for dimension in dimensions)
    return math.sqrt(total / len(dimensions))


def _cosine_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions must match.")

    dot = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0

    cosine_similarity = max(-1.0, min(1.0, dot / (left_norm * right_norm)))
    return (1.0 - cosine_similarity) / 2.0


def feature_distance(
    left: CoffeeFeatures,
    right: CoffeeFeatures,
    weights: DistanceWeights = DistanceWeights(),
) -> tuple[float, dict[str, float]]:
    embedding_distance = _cosine_distance(left.embedding, right.embedding)
    sensory_distance = _mean_squared_distance(left.sensory, right.sensory, SENSORY_DIMENSIONS)
    process_distance = _mean_squared_distance(left.process, right.process, PROCESS_DIMENSIONS)
    distance = (
        weights.embedding * embedding_distance
        + weights.sensory * sensory_distance
        + weights.process * process_distance
    )
    return distance, {
        "embedding": embedding_distance,
        "sensory": sensory_distance,
        "process": process_distance,
        "weighted": distance,
    }


def resolve_neighbor_rank(coffee_count: int) -> int:
    if coffee_count < 2:
        return 1
    return max(2, round(math.sqrt(coffee_count)))


def estimate_temperature(
    features: dict[str, CoffeeFeatures],
    weights: DistanceWeights = DistanceWeights(),
    neighbor_rank: int | None = None,
    target_kernel: float = 0.6,
    default_temperature: float = 0.25,
) -> float:
    if len(features) < 2 or not 0.0 < target_kernel < 1.0:
        return default_temperature

    kth_distances: list[float] = []
    values = list(features.values())
    requested_rank = resolve_neighbor_rank(len(values)) if neighbor_rank is None else neighbor_rank
    rank = max(1, min(requested_rank, len(values) - 1))

    for feature in values:
        distances = sorted(
            feature_distance(feature, other, weights)[0]
            for other in values
            if other.coffee_id != feature.coffee_id
        )
        if distances:
            kth_distances.append(distances[rank - 1])

    if not kth_distances:
        return default_temperature

    reference_distance = sorted(kth_distances)[len(kth_distances) // 2]
    if reference_distance <= 0.0:
        return default_temperature

    return -(reference_distance**2) / math.log(target_kernel)


def kernel(distance: float, temperature: float) -> float:
    if temperature <= 0.0:
        return 0.0
    return math.exp(-(distance**2) / temperature)


def _change_request_penalty(
    candidate: CoffeeFeatures,
    reviewed: CoffeeFeatures,
    attribute: str,
    change_request: ChangeRequest,
    similarity: float,
) -> float:
    if attribute not in SENSORY_DIMENSIONS:
        return 0.0

    direction = change_request.get("direction")
    if direction not in ("higher", "lower"):
        return 0.0

    strength = _clip(_coerce_float(change_request.get("strength", 0.0)))
    if strength == 0.0:
        return 0.0

    reviewed_value = reviewed.sensory.get(attribute, 0.0)
    candidate_value = candidate.sensory.get(attribute, 0.0)
    target_value = change_request.get("target_value")

    if target_value is None:
        adjustment = _coerce_float(change_request.get("adjustment", 0.15 * strength))
        target_value = reviewed_value - adjustment if direction == "lower" else reviewed_value + adjustment

    target = _clip(_coerce_float(target_value))
    if direction == "lower":
        excess = max(0.0, candidate_value - target)
    else:
        excess = max(0.0, target - candidate_value)

    change_span = max(abs(reviewed_value - target), 0.05)
    normalized_excess = min(1.0, excess / change_span)
    return strength * similarity * normalized_excess


def _attribute_opinion_adjustment(
    candidate: CoffeeFeatures,
    reviewed: CoffeeFeatures,
    attribute: str,
    opinion: AttributeOpinion,
    similarity: float,
) -> float:
    if attribute not in SENSORY_DIMENSIONS:
        return 0.0

    sentiment = opinion.get("sentiment")
    if sentiment not in ("liked", "disliked"):
        return 0.0

    strength = _clip(_coerce_float(opinion.get("strength", 0.0)))
    if strength == 0.0:
        return 0.0

    reviewed_value = _clip(_coerce_float(reviewed.sensory.get(attribute, 0.0)))
    candidate_value = _clip(_coerce_float(candidate.sensory.get(attribute, 0.0)))
    reviewed_extremity = abs(reviewed_value - 0.5) * 2.0
    extremity_scale = _clip((reviewed_extremity - 0.25) / 0.75)
    if extremity_scale == 0.0:
        return 0.0

    attribute_similarity = 1.0 - min(1.0, abs(candidate_value - reviewed_value))
    adjustment = strength * extremity_scale * similarity * attribute_similarity
    return adjustment if sentiment == "liked" else -adjustment


def score_candidate(
    candidate: CoffeeFeatures,
    reviews: list[ReviewEvent],
    features: dict[str, CoffeeFeatures],
    temperature: float,
    weights: DistanceWeights = DistanceWeights(),
) -> tuple[float, dict[str, Any]]:
    total = 0.0
    review_details: list[dict[str, Any]] = []

    for review in reviews:
        reviewed = features.get(str(review.get("coffee_id", "")))
        if reviewed is None:
            continue

        distance, distance_breakdown = feature_distance(candidate, reviewed, weights)
        similarity = kernel(distance, temperature)
        overall = max(-1.0, min(1.0, _coerce_float(review.get("overall", 0.0))))
        base_contribution = overall * similarity
        total += base_contribution

        change_request_details: dict[str, float] = {}
        for attribute, change_request in review.get("change_requests", {}).items():
            penalty = _change_request_penalty(candidate, reviewed, attribute, change_request, similarity)
            if penalty:
                change_request_details[attribute] = round(penalty, 6)
                total -= penalty

        attribute_opinion_details: dict[str, float] = {}
        for attribute, opinion in review.get("attribute_opinions", {}).items():
            adjustment = _attribute_opinion_adjustment(candidate, reviewed, attribute, opinion, similarity)
            if adjustment:
                attribute_opinion_details[attribute] = round(adjustment, 6)
                total += adjustment

        review_details.append(
            {
                "reviewed_coffee_id": reviewed.coffee_id,
                "reviewed_name": reviewed.name,
                "distance": round(distance, 6),
                "distance_breakdown": {key: round(value, 6) for key, value in distance_breakdown.items()},
                "kernel": round(similarity, 6),
                "overall": overall,
                "base_contribution": round(base_contribution, 6),
                "change_request_penalties": change_request_details,
                "attribute_opinion_adjustments": attribute_opinion_details,
            }
        )

    return round(total, 6), {
        "candidate": {
            "coffee_id": candidate.coffee_id,
            "name": candidate.name,
            "sensory": candidate.sensory,
            "process": candidate.process,
            "embedding_dimensions": len(candidate.embedding),
            "embedding_norm": round(math.sqrt(sum(value * value for value in candidate.embedding)), 6),
        },
        "reviews": review_details,
    }


def recommend_from_landscape(
    features: dict[str, CoffeeFeatures],
    reviews: list[ReviewEvent],
    top_k: int = 5,
    config: LandscapeConfig = LandscapeConfig(),
    exclude_reviewed: bool = True,
) -> list[dict[str, Any]]:
    reviewed_ids = {str(review.get("coffee_id")) for review in reviews}
    temperature = estimate_temperature(
        features,
        weights=config.distance_weights,
        neighbor_rank=config.neighbor_rank,
        target_kernel=config.target_kernel_at_neighbor,
        default_temperature=config.default_temperature,
    )

    recommendations: list[dict[str, Any]] = []
    for coffee_id, candidate in features.items():
        if exclude_reviewed and coffee_id in reviewed_ids:
            continue

        score, debug = score_candidate(
            candidate,
            reviews,
            features,
            temperature=temperature,
            weights=config.distance_weights,
        )
        recommendations.append(
            {
                "coffee_id": coffee_id,
                "name": candidate.name,
                "score": score,
                "temperature": round(temperature, 6),
                "debug": debug,
            }
        )

    recommendations.sort(key=lambda item: item["score"], reverse=True)
    return recommendations[:top_k]
