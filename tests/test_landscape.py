from __future__ import annotations

import math
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from landscape import (  # noqa: E402
    DistanceWeights,
    build_feature_index,
    estimate_temperature,
    load_feature_index,
    recommend_from_landscape,
    resolve_neighbor_rank,
)


def _coffee(coffee_id: str, process: str = "washed", name: str | None = None) -> dict[str, str]:
    return {
        "coffee_id": coffee_id,
        "name": name or coffee_id.replace("_", " ").title(),
        "process": process,
    }


def _sensory(
    coffee_id: str,
    acidity: float = 0.2,
    sweetness: float = 0.2,
    body: float = 0.4,
    fruitiness: float = 0.2,
    funky_fermented: float = 0.0,
    roasty: float = 0.1,
    clean_cup: float = 0.5,
) -> dict[str, float | str]:
    return {
        "coffee_id": coffee_id,
        "acidity": acidity,
        "sweetness": sweetness,
        "body": body,
        "bitterness": 0.1,
        "fruitiness": fruitiness,
        "chocolate_nutty": 0.1,
        "floral": 0.1,
        "funky_fermented": funky_fermented,
        "roasty": roasty,
        "clean_cup": clean_cup,
    }


def _embedding(coffee_id: str, values: list[float] | None = None) -> dict[str, str]:
    return {
        "coffee_id": coffee_id,
        "embedding": json.dumps(values or [1.0, 0.0, 0.0]),
    }


def _features(
    coffees: list[dict[str, str]],
    sensory: list[dict[str, float | str]],
    embeddings: list[dict[str, str]] | None = None,
):
    if embeddings is None:
        embeddings = [_embedding(str(coffee["coffee_id"])) for coffee in coffees]
    return build_feature_index(pd.DataFrame(coffees), pd.DataFrame(sensory), pd.DataFrame(embeddings))


class LandscapeTests(unittest.TestCase):
    def test_liked_coffee_boosts_similar_candidates(self) -> None:
        features = _features(
            [
                _coffee("liked", "natural"),
                _coffee("similar", "natural"),
                _coffee("unrelated", "washed"),
            ],
            [
                _sensory("liked", fruitiness=0.9, funky_fermented=0.8, clean_cup=0.2),
                _sensory("similar", fruitiness=0.85, funky_fermented=0.75, clean_cup=0.2),
                _sensory("unrelated", acidity=0.8, fruitiness=0.1, funky_fermented=0.0, clean_cup=0.9),
            ],
        )

        recommendations = recommend_from_landscape(
            features,
            [{"coffee_id": "liked", "overall": 1.0}],
            top_k=2,
        )

        self.assertEqual(recommendations[0]["coffee_id"], "similar")
        self.assertGreater(recommendations[0]["score"], recommendations[1]["score"])

    def test_recommendation_debug_includes_candidate_structure(self) -> None:
        features = _features(
            [_coffee("liked"), _coffee("candidate")],
            [_sensory("liked", acidity=0.8), _sensory("candidate", acidity=0.4)],
        )

        recommendations = recommend_from_landscape(
            features,
            [{"coffee_id": "liked", "overall": 1.0}],
            top_k=1,
        )

        candidate_debug = recommendations[0]["debug"]["candidate"]
        self.assertEqual(candidate_debug["coffee_id"], "candidate")
        self.assertIn("sensory", candidate_debug)
        self.assertIn("process", candidate_debug)
        self.assertEqual(candidate_debug["embedding_dimensions"], 3)
        self.assertGreater(candidate_debug["embedding_norm"], 0)

    def test_embedding_similarity_contributes_to_ranking(self) -> None:
        coffees = [
            _coffee("liked", "washed"),
            _coffee("semantic_neighbor", "natural"),
            _coffee("semantic_far", "natural"),
        ]
        sensory = [
            _sensory("liked", acidity=0.5, fruitiness=0.5),
            _sensory("semantic_neighbor", acidity=0.5, fruitiness=0.5),
            _sensory("semantic_far", acidity=0.5, fruitiness=0.5),
        ]
        features = _features(
            coffees,
            sensory,
            [
                _embedding("liked", [1.0, 0.0, 0.0]),
                _embedding("semantic_neighbor", [0.95, 0.05, 0.0]),
                _embedding("semantic_far", [0.0, 1.0, 0.0]),
            ],
        )

        recommendations = recommend_from_landscape(
            features,
            [{"coffee_id": "liked", "overall": 1.0}],
            top_k=2,
        )

        self.assertEqual(recommendations[0]["coffee_id"], "semantic_neighbor")
        breakdown = recommendations[0]["debug"]["reviews"][0]["distance_breakdown"]
        self.assertIn("embedding", breakdown)

    def test_disliked_coffee_suppresses_similar_candidates(self) -> None:
        features = _features(
            [
                _coffee("disliked", "anaerobic"),
                _coffee("similar_bad", "anaerobic"),
                _coffee("safe", "washed"),
            ],
            [
                _sensory("disliked", fruitiness=0.9, funky_fermented=0.9, clean_cup=0.1),
                _sensory("similar_bad", fruitiness=0.88, funky_fermented=0.85, clean_cup=0.1),
                _sensory("safe", acidity=0.3, fruitiness=0.2, funky_fermented=0.0, clean_cup=0.9),
            ],
        )

        recommendations = recommend_from_landscape(
            features,
            [{"coffee_id": "disliked", "overall": -1.0}],
            top_k=2,
        )

        self.assertEqual(recommendations[0]["coffee_id"], "safe")
        self.assertGreater(recommendations[0]["score"], recommendations[1]["score"])

    def test_two_liked_coffees_create_two_regions(self) -> None:
        features = _features(
            [
                _coffee("liked_funky", "natural"),
                _coffee("near_funky", "natural"),
                _coffee("liked_clean", "washed"),
                _coffee("near_clean", "washed"),
                _coffee("middle", "honey"),
            ],
            [
                _sensory("liked_funky", fruitiness=0.95, funky_fermented=0.9, clean_cup=0.1),
                _sensory("near_funky", fruitiness=0.9, funky_fermented=0.85, clean_cup=0.15),
                _sensory("liked_clean", acidity=0.8, fruitiness=0.55, funky_fermented=0.0, clean_cup=0.95),
                _sensory("near_clean", acidity=0.75, fruitiness=0.5, funky_fermented=0.0, clean_cup=0.9),
                _sensory("middle", acidity=0.45, fruitiness=0.55, funky_fermented=0.4, clean_cup=0.5),
            ],
        )

        recommendations = recommend_from_landscape(
            features,
            [
                {"coffee_id": "liked_funky", "overall": 1.0},
                {"coffee_id": "liked_clean", "overall": 1.0},
            ],
            top_k=3,
        )

        self.assertEqual(
            {recommendations[0]["coffee_id"], recommendations[1]["coffee_id"]},
            {"near_funky", "near_clean"},
        )
        self.assertEqual(recommendations[2]["coffee_id"], "middle")

    def test_liked_but_too_acidic_prefers_similar_lower_acidity(self) -> None:
        features = _features(
            [
                _coffee("reviewed", "washed"),
                _coffee("same_acidity", "washed"),
                _coffee("lower_acidity", "washed"),
                _coffee("unrelated_low_acid", "natural"),
            ],
            [
                _sensory("reviewed", acidity=0.8, fruitiness=0.8, clean_cup=0.8),
                _sensory("same_acidity", acidity=0.8, fruitiness=0.8, clean_cup=0.8),
                _sensory("lower_acidity", acidity=0.55, fruitiness=0.8, clean_cup=0.8),
                _sensory("unrelated_low_acid", acidity=0.1, fruitiness=0.1, funky_fermented=0.8, clean_cup=0.1),
            ],
        )

        recommendations = recommend_from_landscape(
            features,
            [
                {
                    "coffee_id": "reviewed",
                    "overall": 1.0,
                    "change_requests": {
                        "acidity": {
                            "direction": "lower",
                            "strength": 1.0,
                            "adjustment": 0.2,
                        }
                    },
                }
            ],
            top_k=3,
        )

        self.assertEqual(recommendations[0]["coffee_id"], "lower_acidity")
        self.assertGreater(
            recommendations[0]["score"],
            next(item["score"] for item in recommendations if item["coffee_id"] == "same_acidity"),
        )
        self.assertGreater(
            recommendations[0]["score"],
            next(item["score"] for item in recommendations if item["coffee_id"] == "unrelated_low_acid"),
        )

    def test_liked_low_attribute_opinion_boosts_low_value_candidates(self) -> None:
        features = _features(
            [
                _coffee("reviewed"),
                _coffee("low_roasty"),
                _coffee("high_roasty"),
            ],
            [
                _sensory("reviewed", roasty=0.2),
                _sensory("low_roasty", roasty=0.25),
                _sensory("high_roasty", roasty=0.8),
            ],
        )

        recommendations = recommend_from_landscape(
            features,
            [
                {
                    "coffee_id": "reviewed",
                    "overall": 0.0,
                    "attribute_opinions": {
                        "roasty": {"sentiment": "liked", "strength": 1.0},
                    },
                }
            ],
            top_k=2,
        )

        self.assertEqual(recommendations[0]["coffee_id"], "low_roasty")
        details = recommendations[0]["debug"]["reviews"][0]
        self.assertIn("roasty", details["attribute_opinion_adjustments"])

    def test_liked_high_attribute_opinion_boosts_high_value_candidates(self) -> None:
        features = _features(
            [_coffee("reviewed"), _coffee("low_roasty"), _coffee("high_roasty")],
            [
                _sensory("reviewed", roasty=0.8),
                _sensory("low_roasty", roasty=0.2),
                _sensory("high_roasty", roasty=0.75),
            ],
        )

        recommendations = recommend_from_landscape(
            features,
            [
                {
                    "coffee_id": "reviewed",
                    "overall": 0.0,
                    "attribute_opinions": {
                        "roasty": {"sentiment": "liked", "strength": 1.0},
                    },
                }
            ],
            top_k=2,
        )

        self.assertEqual(recommendations[0]["coffee_id"], "high_roasty")

    def test_disliked_low_attribute_opinion_suppresses_low_value_candidates(self) -> None:
        features = _features(
            [_coffee("reviewed"), _coffee("low_roasty"), _coffee("high_roasty")],
            [
                _sensory("reviewed", roasty=0.2),
                _sensory("low_roasty", roasty=0.25),
                _sensory("high_roasty", roasty=0.8),
            ],
        )

        recommendations = recommend_from_landscape(
            features,
            [
                {
                    "coffee_id": "reviewed",
                    "overall": 0.0,
                    "attribute_opinions": {
                        "roasty": {"sentiment": "disliked", "strength": 1.0},
                    },
                }
            ],
            top_k=2,
        )

        self.assertEqual(recommendations[0]["coffee_id"], "high_roasty")

    def test_disliked_high_attribute_opinion_suppresses_high_value_candidates(self) -> None:
        features = _features(
            [_coffee("reviewed"), _coffee("low_roasty"), _coffee("high_roasty")],
            [
                _sensory("reviewed", roasty=0.8),
                _sensory("low_roasty", roasty=0.2),
                _sensory("high_roasty", roasty=0.75),
            ],
        )

        recommendations = recommend_from_landscape(
            features,
            [
                {
                    "coffee_id": "reviewed",
                    "overall": 0.0,
                    "attribute_opinions": {
                        "roasty": {"sentiment": "disliked", "strength": 1.0},
                    },
                }
            ],
            top_k=2,
        )

        self.assertEqual(recommendations[0]["coffee_id"], "low_roasty")

    def test_neutral_attribute_opinion_has_no_adjustment(self) -> None:
        features = _features(
            [_coffee("reviewed"), _coffee("candidate")],
            [
                _sensory("reviewed", roasty=0.5),
                _sensory("candidate", roasty=0.5),
            ],
        )

        recommendations = recommend_from_landscape(
            features,
            [
                {
                    "coffee_id": "reviewed",
                    "overall": 0.0,
                    "attribute_opinions": {
                        "roasty": {"sentiment": "liked", "strength": 1.0},
                    },
                }
            ],
            top_k=1,
        )

        self.assertEqual(
            recommendations[0]["debug"]["reviews"][0]["attribute_opinion_adjustments"],
            {},
        )

    def test_extreme_attribute_opinion_has_stronger_adjustment_than_moderate(self) -> None:
        moderate_features = _features(
            [_coffee("reviewed"), _coffee("candidate")],
            [_sensory("reviewed", roasty=0.7), _sensory("candidate", roasty=0.7)],
        )
        extreme_features = _features(
            [_coffee("reviewed"), _coffee("candidate")],
            [_sensory("reviewed", roasty=0.9), _sensory("candidate", roasty=0.9)],
        )
        review: ReviewEvent = {
            "coffee_id": "reviewed",
            "overall": 0.0,
            "attribute_opinions": {
                "roasty": {"sentiment": "liked", "strength": 1.0},
            },
        }

        moderate = recommend_from_landscape(moderate_features, [review], top_k=1)
        extreme = recommend_from_landscape(extreme_features, [review], top_k=1)

        moderate_adjustment = moderate[0]["debug"]["reviews"][0]["attribute_opinion_adjustments"]["roasty"]
        extreme_adjustment = extreme[0]["debug"]["reviews"][0]["attribute_opinion_adjustments"]["roasty"]
        self.assertGreater(extreme_adjustment, moderate_adjustment)

    def test_reviewed_coffees_are_excluded_by_default(self) -> None:
        features = _features(
            [_coffee("reviewed"), _coffee("candidate")],
            [_sensory("reviewed"), _sensory("candidate")],
        )

        recommendations = recommend_from_landscape(
            features,
            [{"coffee_id": "reviewed", "overall": 1.0}],
            top_k=2,
        )

        self.assertEqual([item["coffee_id"] for item in recommendations], ["candidate"])

    def test_missing_sensory_vectors_fail_loudly(self) -> None:
        coffees = pd.DataFrame([_coffee("a"), _coffee("b")])
        sensory = pd.DataFrame([_sensory("a")])
        embeddings = pd.DataFrame([_embedding("a"), _embedding("b")])

        with self.assertRaisesRegex(KeyError, "Missing sensory vector for coffee_id: b"):
            build_feature_index(coffees, sensory, embeddings)

    def test_missing_embeddings_fail_loudly(self) -> None:
        coffees = pd.DataFrame([_coffee("a"), _coffee("b")])
        sensory = pd.DataFrame([_sensory("a"), _sensory("b")])
        embeddings = pd.DataFrame([_embedding("a")])

        with self.assertRaisesRegex(KeyError, "Missing embedding for coffee_id: b"):
            build_feature_index(coffees, sensory, embeddings)

    def test_load_feature_index_requires_sensory_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            coffees_path = Path(tmpdir) / "coffees.csv"
            sensory_path = Path(tmpdir) / "missing_sensory.csv"
            embeddings_path = Path(tmpdir) / "coffee_embeddings.csv"
            pd.DataFrame([_coffee("a")]).to_csv(coffees_path, index=False)
            pd.DataFrame([_embedding("a")]).to_csv(embeddings_path, index=False)

            with self.assertRaisesRegex(FileNotFoundError, "Sensory vector CSV not found"):
                load_feature_index(coffees_path, sensory_path, embeddings_path)

    def test_load_feature_index_requires_embeddings_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            coffees_path = Path(tmpdir) / "coffees.csv"
            sensory_path = Path(tmpdir) / "coffee_sensory_vectors.csv"
            embeddings_path = Path(tmpdir) / "missing_embeddings.csv"
            pd.DataFrame([_coffee("a")]).to_csv(coffees_path, index=False)
            pd.DataFrame([_sensory("a")]).to_csv(sensory_path, index=False)

            with self.assertRaisesRegex(FileNotFoundError, "Coffee embeddings CSV not found"):
                load_feature_index(coffees_path, sensory_path, embeddings_path)

    def test_temperature_uses_sqrt_catalogue_size_neighbor_rank(self) -> None:
        features = _features(
            [_coffee(f"coffee_{index}") for index in range(9)],
            [
                _sensory(f"coffee_{index}", acidity=index / 10, fruitiness=index / 10)
                for index in range(9)
            ],
        )

        self.assertEqual(resolve_neighbor_rank(1), 1)
        self.assertEqual(resolve_neighbor_rank(4), 2)
        self.assertEqual(resolve_neighbor_rank(23), 5)
        self.assertEqual(resolve_neighbor_rank(100), 10)

        automatic = estimate_temperature(features)
        explicit = estimate_temperature(features, neighbor_rank=max(2, round(math.sqrt(len(features)))))

        self.assertGreater(automatic, 0.0)
        self.assertAlmostEqual(automatic, explicit, places=12)


if __name__ == "__main__":
    unittest.main()
