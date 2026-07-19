from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from coffee_recommender.application import ApplicationService
from coffee_recommender.api_models import (
    RecommendationRunPayload,
    ReviewHistoryItemPayload,
    ReviewSessionPayload,
    SubmitReviewRequest,
)
from coffee_recommender.config import DataPaths
from coffee_recommender.db.review_history import ReviewHistoryStore


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
    fruitiness: float = 0.5,
) -> dict[str, float | str]:
    return {
        "coffee_id": coffee_id,
        "acidity": acidity,
        "sweetness": 0.5,
        "body": 0.5,
        "bitterness": 0.1,
        "fruitiness": fruitiness,
        "chocolate_nutty": 0.1,
        "floral": 0.1,
        "funky_fermented": 0.0,
        "roasty": 0.1,
        "clean_cup": 0.8,
    }


def _embedding_row(coffee_id: str, embedding: str) -> dict[str, str]:
    return {
        "coffee_id": coffee_id,
        "embedding": embedding,
    }


class FakeReviewHistoryStore(ReviewHistoryStore):
    def __init__(self) -> None:
        self.session = ReviewSessionPayload()
        self.persisted_submissions = 0
        self.runs: list[RecommendationRunPayload] = []
        self.reviews: list[ReviewHistoryItemPayload] = []

    def get_review_session(self) -> ReviewSessionPayload:
        return self.session

    def clear_review_session(self) -> ReviewSessionPayload:
        self.session = ReviewSessionPayload()
        return self.session

    def persist_review_submission(
        self,
        *,
        review_text: str,
        reviewed_coffee,
        event,
        recommendations,
        algorithm_version: str,
    ) -> ReviewSessionPayload:
        self.persisted_submissions += 1
        review_item = ReviewHistoryItemPayload(
            review_id=self.persisted_submissions,
            coffee_id=event.coffee_id,
            review_text=review_text,
            overall=event.overall,
            created_at="2026-07-19T00:00:00Z",
        )
        run = RecommendationRunPayload(
            run_id=self.persisted_submissions,
            seed_review_event_id=self.persisted_submissions,
            algorithm_version=algorithm_version,
            created_at="2026-07-19T00:00:00Z",
            recommendations=recommendations,
        )
        self.reviews = [review_item]
        self.runs = [run]
        self.session = ReviewSessionPayload(
            review_events=[event],
            reviewed_feature_overrides={},
            last_event=event,
            last_recommendations=recommendations,
        )
        return self.session

    def list_reviews(self) -> list[ReviewHistoryItemPayload]:
        return self.reviews

    def list_recommendation_runs(self) -> list[RecommendationRunPayload]:
        return self.runs

    def get_recommendation_run(self, run_id: int) -> RecommendationRunPayload:
        for run in self.runs:
            if run.run_id == run_id:
                return run
        raise KeyError(f"Recommendation run not found: {run_id}")


class ApplicationPersistenceTests(unittest.TestCase):
    def test_submit_review_uses_review_history_store_when_configured(self) -> None:
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

            store = FakeReviewHistoryStore()
            service = ApplicationService(
                data_paths=DataPaths(
                    coffees_path=coffees_path,
                    sensory_path=sensory_path,
                    embeddings_path=embeddings_path,
                ),
                _review_history_store=store,
            )
            reviewed = service.get_catalogue_reviewed_coffee("reviewed")

            with patch(
                "coffee_recommender.coffee_service.parse_review_event",
                return_value={
                    "coffee_id": "reviewed",
                    "overall": 1.0,
                    "change_requests": {},
                    "attribute_opinions": {},
                },
            ):
                response = service.submit_review(
                    request=SubmitReviewRequest(
                        review_text="Loved it.",
                        reviewed_coffee=reviewed,
                        top_k=1,
                    )
                )

            self.assertEqual(store.persisted_submissions, 1)
            self.assertEqual(response.review_session.last_event.coffee_id, "reviewed")
            self.assertEqual(response.review_session.last_recommendations[0].coffee_id, "candidate")
            self.assertEqual(service.get_review_session().last_event.coffee_id, "reviewed")


if __name__ == "__main__":
    unittest.main()
