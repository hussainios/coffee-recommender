from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from coffee_recommender.api import create_app
from coffee_recommender.application import ApplicationService
from coffee_recommender import application, coffee_service
from coffee_recommender.config import DataPaths


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


class ApiTests(unittest.TestCase):
    def test_catalogue_and_review_endpoints_work_end_to_end(self) -> None:
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
                    _sensory_row("far", acidity=0.1, fruitiness=0.1),
                ]
            ).to_csv(sensory_path, index=False)
            pd.DataFrame(
                [
                    _embedding_row("reviewed", "[1.0, 0.0, 0.0]"),
                    _embedding_row("candidate", "[0.98, 0.02, 0.0]"),
                    _embedding_row("far", "[0.0, 1.0, 0.0]"),
                ]
            ).to_csv(embeddings_path, index=False)

            service = ApplicationService(
                data_paths=DataPaths(
                    coffees_path=coffees_path,
                    sensory_path=sensory_path,
                    embeddings_path=embeddings_path,
                )
            )
            client = TestClient(create_app(service))

            coffees = client.get("/catalogue/coffees")
            self.assertEqual(coffees.status_code, 200)
            self.assertEqual(coffees.json()[0]["coffee_id"], "candidate")
            self.assertEqual(set(coffees.json()[0]), {"coffee_id", "name"})

            session = client.get("/review-session")
            self.assertEqual(session.status_code, 200)
            self.assertEqual(session.json()["review_events"], [])

            reviewed = client.get("/reviewed-coffees/catalogue/reviewed")
            self.assertEqual(reviewed.status_code, 200)
            self.assertEqual(reviewed.json()["source_type"], "catalogue")

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
                submitted = client.post(
                    "/reviews/submit",
                    json={
                        "review_text": "Loved it.",
                        "reviewed_coffee": reviewed.json(),
                        "top_k": 1,
                    },
                )

            self.assertEqual(submitted.status_code, 200)
            payload = submitted.json()
            self.assertEqual(payload["event"]["coffee_id"], "reviewed")
            self.assertEqual(payload["review_session"]["last_recommendations"][0]["coffee_id"], "candidate")

            current_session = client.get("/review-session")
            self.assertEqual(current_session.status_code, 200)
            self.assertEqual(current_session.json()["last_event"]["coffee_id"], "reviewed")

            landscape = client.get("/review-session/landscape?show_surface=true")
            self.assertEqual(landscape.status_code, 200)
            self.assertIn("figure", landscape.json())

            cleared = client.delete("/review-session")
            self.assertEqual(cleared.status_code, 200)
            self.assertEqual(cleared.json()["review_events"], [])

    def test_url_endpoint_translates_value_errors(self) -> None:
        client = TestClient(create_app(ApplicationService(
            data_paths=DataPaths(
                coffees_path=Path("data/processed/coffees.csv"),
                sensory_path=Path("data/processed/coffee_sensory_vectors.csv"),
                embeddings_path=Path("data/processed/coffee_embeddings.csv"),
            )
        )))

        with patch.object(application, "prepare_url_selection", side_effect=ValueError("Bad URL")):
            response = client.post("/reviewed-coffees/from-url", json={"url": "notaurl"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Bad URL")

    def test_health_includes_cors_headers_for_frontend_dev_origin(self) -> None:
        client = TestClient(create_app())

        response = client.get(
            "/health",
            headers={"Origin": "http://localhost:5173"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "http://localhost:5173")


if __name__ == "__main__":
    unittest.main()
