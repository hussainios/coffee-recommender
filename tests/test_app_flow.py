from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import coffee_service  # noqa: E402
from app_state import append_review_event, build_scoring_features, reset_review_history  # noqa: E402
from coffee_service import (  # noqa: E402
    build_coffee_options,
    get_cached_url_selection,
    load_catalogue,
    select_catalogue_reviewed_coffee,
    submit_review,
)
from landscape import CoffeeFeatures, recommend_from_landscape  # noqa: E402
from reviewed_coffee_url import ReviewedCoffeeFromUrl  # noqa: E402
from schemas import CoffeeRecord, Process, SensoryVector  # noqa: E402
from unittest.mock import patch


def _coffee_row(coffee_id: str, name: str, process: str = "washed") -> dict[str, str]:
    return {
        "coffee_id": coffee_id,
        "name": name,
        "process": process,
    }


def _sensory_row(
    coffee_id: str,
    *,
    acidity: float = 0.5,
    sweetness: float = 0.5,
    body: float = 0.5,
    bitterness: float = 0.1,
    fruitiness: float = 0.5,
    chocolate_nutty: float = 0.1,
    floral: float = 0.1,
    funky_fermented: float = 0.0,
    roasty: float = 0.1,
    clean_cup: float = 0.8,
) -> dict[str, float | str]:
    return {
        "coffee_id": coffee_id,
        "acidity": acidity,
        "sweetness": sweetness,
        "body": body,
        "bitterness": bitterness,
        "fruitiness": fruitiness,
        "chocolate_nutty": chocolate_nutty,
        "floral": floral,
        "funky_fermented": funky_fermented,
        "roasty": roasty,
        "clean_cup": clean_cup,
    }


def _embedding_row(coffee_id: str, embedding: str) -> dict[str, str]:
    return {
        "coffee_id": coffee_id,
        "embedding": embedding,
    }


class _SessionState:
    def __init__(self) -> None:
        self.review_events = []
        self.reviewed_feature_overrides = {}
        self.last_event = None
        self.last_recommendations = []


class AppFlowTests(unittest.TestCase):
    def test_catalogue_review_flow_produces_ranked_recommendations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            coffees_path = Path(tmpdir) / "coffees.csv"
            sensory_path = Path(tmpdir) / "coffee_sensory_vectors.csv"
            embeddings_path = Path(tmpdir) / "coffee_embeddings.csv"

            pd.DataFrame(
                [
                    _coffee_row("reviewed", "Reviewed Coffee", "washed"),
                    _coffee_row("candidate", "Candidate Coffee", "washed"),
                    _coffee_row("far", "Far Coffee", "natural"),
                ]
            ).to_csv(coffees_path, index=False)
            pd.DataFrame(
                [
                    _sensory_row("reviewed", acidity=0.8, fruitiness=0.8),
                    _sensory_row("candidate", acidity=0.55, fruitiness=0.8),
                    _sensory_row("far", acidity=0.1, fruitiness=0.1, funky_fermented=0.8, clean_cup=0.1),
                ]
            ).to_csv(sensory_path, index=False)
            pd.DataFrame(
                [
                    _embedding_row("reviewed", "[1.0, 0.0, 0.0]"),
                    _embedding_row("candidate", "[0.98, 0.02, 0.0]"),
                    _embedding_row("far", "[0.0, 1.0, 0.0]"),
                ]
            ).to_csv(embeddings_path, index=False)

            catalogue = load_catalogue(coffees_path, sensory_path, embeddings_path)
            state = _SessionState()
            reviewed = catalogue.features["reviewed"]
            event = {
                "coffee_id": reviewed.coffee_id,
                "overall": 1.0,
                "change_requests": {
                    "acidity": {
                        "direction": "lower",
                        "strength": 1.0,
                        "adjustment": 0.2,
                    }
                },
                "attribute_opinions": {},
            }

            append_review_event(state, event, reviewed, is_temporary=False)
            recommendations = recommend_from_landscape(
                build_scoring_features(catalogue.features, state.reviewed_feature_overrides),
                state.review_events,
                top_k=2,
            )

            self.assertEqual(recommendations[0]["coffee_id"], "candidate")
            self.assertEqual(set(recommendations[0]), {"coffee_id", "name", "score", "temperature", "debug"})
            self.assertEqual(state.last_event, event)

    def test_service_selects_catalogue_metadata_and_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            coffees_path = Path(tmpdir) / "coffees.csv"
            sensory_path = Path(tmpdir) / "coffee_sensory_vectors.csv"
            embeddings_path = Path(tmpdir) / "coffee_embeddings.csv"

            pd.DataFrame(
                [
                    _coffee_row("b", "Beta Coffee", "natural"),
                    _coffee_row("a", "Alpha Coffee", "washed"),
                ]
            ).to_csv(coffees_path, index=False)
            pd.DataFrame([_sensory_row("a"), _sensory_row("b")]).to_csv(sensory_path, index=False)
            pd.DataFrame(
                [
                    _embedding_row("a", "[1.0, 0.0, 0.0]"),
                    _embedding_row("b", "[0.0, 1.0, 0.0]"),
                ]
            ).to_csv(embeddings_path, index=False)

            catalogue = load_catalogue(coffees_path, sensory_path, embeddings_path)
            options = build_coffee_options(catalogue.coffees)
            selected = select_catalogue_reviewed_coffee(catalogue, "a")

            self.assertEqual(list(options), ["Alpha Coffee (a)", "Beta Coffee (b)"])
            self.assertEqual(selected.features.coffee_id, "a")
            self.assertEqual(selected.metadata["name"], "Alpha Coffee")
            self.assertIsNone(selected.sensory)
            self.assertFalse(selected.is_temporary)

    def test_url_reviewed_coffee_can_influence_ranking_without_becoming_candidate(self) -> None:
        state = _SessionState()
        temporary = CoffeeFeatures(
            coffee_id="temporary-url",
            name="Temporary URL Coffee",
            sensory={
                "acidity": 0.8,
                "sweetness": 0.4,
                "body": 0.4,
                "bitterness": 0.1,
                "fruitiness": 0.8,
                "chocolate_nutty": 0.1,
                "floral": 0.2,
                "funky_fermented": 0.1,
                "roasty": 0.0,
                "clean_cup": 0.8,
            },
            process={
                "process_washed": 1.0,
                "process_natural": 0.0,
                "process_honey": 0.0,
                "process_anaerobic": 0.0,
                "process_cofermented": 0.0,
            },
            embedding=[1.0, 0.0, 0.0],
        )
        candidate = CoffeeFeatures(
            coffee_id="candidate",
            name="Candidate",
            sensory=temporary.sensory.copy(),
            process=temporary.process.copy(),
            embedding=[0.99, 0.01, 0.0],
        )
        far = CoffeeFeatures(
            coffee_id="far",
            name="Far",
            sensory={**temporary.sensory, "acidity": 0.1, "fruitiness": 0.1, "clean_cup": 0.1},
            process={
                "process_washed": 0.0,
                "process_natural": 1.0,
                "process_honey": 0.0,
                "process_anaerobic": 0.0,
                "process_cofermented": 0.0,
            },
            embedding=[0.0, 1.0, 0.0],
        )
        event = {
            "coffee_id": temporary.coffee_id,
            "overall": 1.0,
            "change_requests": {},
            "attribute_opinions": {},
        }

        append_review_event(state, event, temporary, is_temporary=True)
        recommendations = recommend_from_landscape(
            build_scoring_features(
                {
                    candidate.coffee_id: candidate,
                    far.coffee_id: far,
                },
                state.reviewed_feature_overrides,
            ),
            state.review_events,
            top_k=2,
        )

        self.assertEqual(recommendations[0]["coffee_id"], "candidate")
        self.assertNotIn(temporary.coffee_id, [item["coffee_id"] for item in recommendations])
        self.assertIn(temporary.coffee_id, state.reviewed_feature_overrides)

    def test_submit_review_orchestrates_parser_state_and_recommendations(self) -> None:
        state = _SessionState()
        reviewed = CoffeeFeatures(
            coffee_id="reviewed",
            name="Reviewed",
            sensory={
                "acidity": 0.8,
                "sweetness": 0.4,
                "body": 0.4,
                "bitterness": 0.1,
                "fruitiness": 0.8,
                "chocolate_nutty": 0.1,
                "floral": 0.2,
                "funky_fermented": 0.1,
                "roasty": 0.0,
                "clean_cup": 0.8,
            },
            process={
                "process_washed": 1.0,
                "process_natural": 0.0,
                "process_honey": 0.0,
                "process_anaerobic": 0.0,
                "process_cofermented": 0.0,
            },
            embedding=[1.0, 0.0, 0.0],
        )
        candidate = CoffeeFeatures(
            coffee_id="candidate",
            name="Candidate",
            sensory=reviewed.sensory.copy(),
            process=reviewed.process.copy(),
            embedding=[0.99, 0.01, 0.0],
        )
        event = {
            "coffee_id": reviewed.coffee_id,
            "overall": 1.0,
            "change_requests": {},
            "attribute_opinions": {},
        }

        with patch.object(coffee_service, "parse_review_event", return_value=event) as parse:
            result = submit_review(
                session_state=state,
                review_text="Loved it.",
                reviewed_coffee=reviewed,
                catalogue_features={candidate.coffee_id: candidate},
                top_k=1,
                is_temporary=True,
            )

        parse.assert_called_once_with("Loved it.", reviewed)
        self.assertEqual(result.event, event)
        self.assertEqual(state.review_events, [event])
        self.assertIn(reviewed.coffee_id, state.reviewed_feature_overrides)
        self.assertEqual(result.recommendations[0]["coffee_id"], "candidate")
        self.assertIn(reviewed.coffee_id, result.scoring_features)

    def test_cached_url_selection_requires_matching_normalized_source(self) -> None:
        sensory = SensoryVector(
            coffee_id="url-coffee",
            acidity=0.6,
            sweetness=0.7,
            body=0.5,
            bitterness=0.2,
            fruitiness=0.8,
            chocolate_nutty=0.2,
            floral=0.1,
            funky_fermented=0.0,
            roasty=0.1,
            clean_cup=0.9,
            confidence=0.8,
            evidence={
                "acidity": [],
                "sweetness": [],
                "body": [],
                "bitterness": [],
                "fruitiness": [],
                "chocolate_nutty": [],
                "floral": [],
                "funky_fermented": [],
                "roasty": [],
                "clean_cup": [],
            },
        )
        features = CoffeeFeatures(
            coffee_id="url-coffee",
            name="URL Coffee",
            sensory={dimension: 0.5 for dimension in sensory.evidence},
            process={
                "process_washed": 1.0,
                "process_natural": 0.0,
                "process_honey": 0.0,
                "process_anaerobic": 0.0,
                "process_cofermented": 0.0,
            },
            embedding=[1.0, 0.0, 0.0],
        )
        reviewed = ReviewedCoffeeFromUrl(
            url="https://example.com/coffee",
            coffee=CoffeeRecord(
                coffee_id="url-coffee",
                name="URL Coffee",
                process=Process.WASHED,
                source_url="https://example.com/coffee",
                source_file="url:https://example.com/coffee",
            ),
            sensory=sensory,
            features=features,
            extracted_text="Origin: Kenya. Process: Washed. Tasting notes: Citrus.",
        )

        selection = get_cached_url_selection(
            "https://example.com/coffee",
            reviewed,
            "https://example.com/coffee",
        )
        stale_selection = get_cached_url_selection(
            "https://example.com/other",
            reviewed,
            "https://example.com/coffee",
        )

        self.assertIsNotNone(selection)
        self.assertEqual(selection.features.coffee_id, "url-coffee")
        self.assertTrue(selection.is_temporary)
        self.assertIsNone(stale_selection)

    def test_reset_review_history_clears_temporary_overrides_and_recommendations(self) -> None:
        state = _SessionState()
        state.review_events = [{"coffee_id": "a"}]
        state.reviewed_feature_overrides = {"a": object()}
        state.last_event = {"coffee_id": "a"}
        state.last_recommendations = [{"coffee_id": "b"}]

        reset_review_history(state)

        self.assertEqual(state.review_events, [])
        self.assertEqual(state.reviewed_feature_overrides, {})
        self.assertIsNone(state.last_event)
        self.assertEqual(state.last_recommendations, [])


if __name__ == "__main__":
    unittest.main()
