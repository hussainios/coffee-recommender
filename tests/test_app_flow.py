from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from coffee_recommender import coffee_service
from coffee_recommender.api_models import SubmitReviewRequest
from coffee_recommender.application import ApplicationService
from coffee_recommender.coffee_service import (
    build_coffee_options,
    get_cached_url_selection,
    load_catalogue,
    select_catalogue_reviewed_coffee,
    submit_review,
)
from coffee_recommender.config import DataPaths
from coffee_recommender.review_session import create_review_session
from coffee_recommender.reviewed_coffee_url import ReviewedCoffeeFromUrl
from coffee_recommender.schemas import CoffeeRecord, Process, SensoryVector
from coffee_recommender.landscape import CoffeeFeatures


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


class AppFlowTests(unittest.TestCase):
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

    def test_submit_review_returns_updated_session_and_recommendations(self) -> None:
        session = create_review_session()
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
                review_session=session,
                review_text="Loved it.",
                reviewed_coffee=reviewed,
                catalogue_features={candidate.coffee_id: candidate},
                top_k=1,
                is_external_url=True,
            )

        parse.assert_called_once_with("Loved it.", reviewed)
        self.assertEqual(result.event.coffee_id, "reviewed")
        self.assertEqual(result.review_session.review_events[0].coffee_id, "reviewed")
        self.assertIn(reviewed.coffee_id, result.review_session.reviewed_feature_overrides)
        self.assertEqual(result.recommendations[0].coffee_id, "candidate")
        self.assertIn(reviewed.coffee_id, result.scoring_features)

    def test_application_service_processes_submit_review_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            coffees_path = Path(tmpdir) / "coffees.csv"
            sensory_path = Path(tmpdir) / "coffee_sensory_vectors.csv"
            embeddings_path = Path(tmpdir) / "coffee_embeddings.csv"

            pd.DataFrame(
                [
                    _coffee_row("reviewed", "Reviewed Coffee", "washed"),
                    _coffee_row("candidate", "Candidate Coffee", "washed"),
                ]
            ).to_csv(coffees_path, index=False)
            pd.DataFrame(
                [
                    _sensory_row("reviewed", acidity=0.8, fruitiness=0.8),
                    _sensory_row("candidate", acidity=0.55, fruitiness=0.8),
                ]
            ).to_csv(sensory_path, index=False)
            pd.DataFrame(
                [
                    _embedding_row("reviewed", "[1.0, 0.0, 0.0]"),
                    _embedding_row("candidate", "[0.98, 0.02, 0.0]"),
                ]
            ).to_csv(embeddings_path, index=False)

            service = ApplicationService(
                data_paths=DataPaths(
                    coffees_path=coffees_path,
                    sensory_path=sensory_path,
                    embeddings_path=embeddings_path,
                )
            )
            reviewed = service.get_catalogue_reviewed_coffee("reviewed")

            with patch.object(
                coffee_service,
                "parse_review_event",
                return_value={
                    "coffee_id": "reviewed",
                    "overall": 1.0,
                    "change_requests": {},
                    "attribute_opinions": {},
                },
            ):
                response = service.submit_review(
                    SubmitReviewRequest(
                        review_text="Loved it.",
                        reviewed_coffee=reviewed,
                        top_k=1,
                    )
                )

            self.assertEqual(response.event.coffee_id, "reviewed")
            self.assertEqual(response.review_session.last_recommendations[0].coffee_id, "candidate")

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


if __name__ == "__main__":
    unittest.main()
