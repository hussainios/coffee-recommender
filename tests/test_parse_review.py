from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import parse_review  # noqa: E402
import openai_client  # noqa: E402
from landscape import CoffeeFeatures, recommend_from_landscape  # noqa: E402
from parse_review import parse_review_event  # noqa: E402


class _Responses:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        return SimpleNamespace(output_text=json.dumps(self.payload))


class _Client:
    def __init__(self, payload: dict[str, object]) -> None:
        self.responses = _Responses(payload)


def _coffee(
    coffee_id: str = "reviewed",
    acidity: float = 0.8,
    roasty: float = 0.1,
    embedding: list[float] | None = None,
) -> CoffeeFeatures:
    return CoffeeFeatures(
        coffee_id=coffee_id,
        name=coffee_id.replace("_", " ").title(),
        sensory={
            "acidity": acidity,
            "sweetness": 0.4,
            "body": 0.4,
            "bitterness": 0.1,
            "fruitiness": 0.8,
            "chocolate_nutty": 0.1,
            "floral": 0.2,
            "funky_fermented": 0.2,
            "roasty": roasty,
            "clean_cup": 0.8,
        },
        process={
            "process_washed": 1.0,
            "process_natural": 0.0,
            "process_honey": 0.0,
            "process_anaerobic": 0.0,
            "process_cofermented": 0.0,
        },
        embedding=embedding or [1.0, 0.0, 0.0],
    )


class ParseReviewEventTests(unittest.TestCase):
    def setUp(self) -> None:
        openai_client.reset_openai_client_cache()

    def test_positive_review_with_mild_acidity_correction(self) -> None:
        payload = {
            "overall": 0.75,
            "change_requests": [
                {"attribute": "acidity", "direction": "lower", "strength": 0.35}
            ],
            "attribute_opinions": [],
        }

        with patch.object(openai_client, "get_openai_client", return_value=_Client(payload)):
            event = parse_review_event("I liked this, but it was a little too acidic.", _coffee())

        self.assertEqual(event["coffee_id"], "reviewed")
        self.assertGreater(event["overall"], 0)
        self.assertEqual(event["change_requests"]["acidity"]["direction"], "lower")
        self.assertAlmostEqual(event["change_requests"]["acidity"]["strength"], 0.35)

    def test_review_parser_uses_zero_temperature_by_default(self) -> None:
        payload = {
            "overall": 0.0,
            "change_requests": [],
            "attribute_opinions": [],
        }
        fake_client = _Client(payload)

        with patch.object(openai_client, "get_openai_client", return_value=fake_client):
            parse_review_event("It was fine.", _coffee())

        self.assertEqual(fake_client.responses.kwargs["temperature"], 0.0)

    def test_negative_review_with_funky_and_bitter_change_requests(self) -> None:
        payload = {
            "overall": -0.8,
            "change_requests": [
                {"attribute": "funky_fermented", "direction": "lower", "strength": 0.8},
                {"attribute": "bitterness", "direction": "lower", "strength": 0.7},
            ],
            "attribute_opinions": [],
        }

        with patch.object(openai_client, "get_openai_client", return_value=_Client(payload)):
            event = parse_review_event("I hated this, too funky and bitter.", _coffee())

        self.assertLess(event["overall"], 0)
        self.assertEqual(event["change_requests"]["funky_fermented"]["direction"], "lower")
        self.assertEqual(event["change_requests"]["bitterness"]["direction"], "lower")

    def test_wanted_more_sweetness_returns_higher_change_request(self) -> None:
        payload = {
            "overall": 0.4,
            "change_requests": [
                {"attribute": "sweetness", "direction": "higher", "strength": 0.6}
            ],
            "attribute_opinions": [],
        }

        with patch.object(openai_client, "get_openai_client", return_value=_Client(payload)):
            event = parse_review_event("Nice, but I wanted more sweetness.", _coffee())

        self.assertEqual(event["change_requests"]["sweetness"]["direction"], "higher")
        self.assertAlmostEqual(event["change_requests"]["sweetness"]["strength"], 0.6)

    def test_unsupported_feedback_is_ignored_before_event_construction(self) -> None:
        payload = {
            "overall": 0.5,
            "change_requests": [
                {"attribute": "price", "direction": "lower", "strength": 1.0},
                {"attribute": "acidity", "direction": "sideways", "strength": 1.0},
                {"attribute": "sweetness", "direction": "higher", "strength": 0.0},
            ],
            "attribute_opinions": [
                {"attribute": "price", "sentiment": "liked", "strength": 1.0},
                {"attribute": "body", "sentiment": "meh", "strength": 1.0},
                {"attribute": "roasty", "sentiment": "liked", "strength": 0.0},
            ],
        }

        with patch.object(openai_client, "get_openai_client", return_value=_Client(payload)):
            event = parse_review_event("Good but cheaper please, somehow.", _coffee())

        self.assertEqual(event["change_requests"], {})
        self.assertEqual(event["attribute_opinions"], {})

    def test_liked_roast_level_returns_attribute_opinion(self) -> None:
        payload = {
            "overall": -0.8,
            "change_requests": [],
            "attribute_opinions": [
                {"attribute": "roasty", "sentiment": "liked", "strength": 0.4}
            ],
        }

        with patch.object(openai_client, "get_openai_client", return_value=_Client(payload)):
            event = parse_review_event("I hated this coffee, but I liked the roast level.", _coffee(roasty=0.2))

        self.assertEqual(event["attribute_opinions"]["roasty"]["sentiment"], "liked")
        self.assertAlmostEqual(event["attribute_opinions"]["roasty"]["strength"], 0.4)

    def test_disliked_body_returns_attribute_opinion(self) -> None:
        payload = {
            "overall": 0.2,
            "change_requests": [],
            "attribute_opinions": [
                {"attribute": "body", "sentiment": "disliked", "strength": 0.5}
            ],
        }

        with patch.object(openai_client, "get_openai_client", return_value=_Client(payload)):
            event = parse_review_event("I liked this coffee overall, but disliked the body.", _coffee())

        self.assertEqual(event["attribute_opinions"]["body"]["sentiment"], "disliked")

    def test_mixed_sentence_can_return_change_request_and_attribute_opinion(self) -> None:
        payload = {
            "overall": 0.4,
            "change_requests": [
                {"attribute": "sweetness", "direction": "higher", "strength": 0.3}
            ],
            "attribute_opinions": [
                {"attribute": "sweetness", "sentiment": "liked", "strength": 0.4}
            ],
        }

        with patch.object(openai_client, "get_openai_client", return_value=_Client(payload)):
            event = parse_review_event("The sweetness was nice, but I wanted it a bit sweeter.", _coffee())

        self.assertEqual(event["change_requests"]["sweetness"]["direction"], "higher")
        self.assertEqual(event["attribute_opinions"]["sweetness"]["sentiment"], "liked")

    def test_missing_api_key_raises_clear_error(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(openai_client, "load_dotenv", return_value=False),
        ):
            openai_client.reset_openai_client_cache()

            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY is required for review parsing"):
                openai_client.get_openai_client("review parsing")

    def test_parsed_event_works_with_landscape_recommendations(self) -> None:
        payload = {
            "overall": 1.0,
            "change_requests": [
                {"attribute": "acidity", "direction": "lower", "strength": 1.0}
            ],
            "attribute_opinions": [],
        }
        reviewed = _coffee("reviewed")
        same_acidity = _coffee("same_acidity")
        lower_acidity = _coffee("lower_acidity", acidity=0.45, embedding=[0.98, 0.02, 0.0])
        features = {
            reviewed.coffee_id: reviewed,
            same_acidity.coffee_id: same_acidity,
            lower_acidity.coffee_id: lower_acidity,
        }

        with patch.object(openai_client, "get_openai_client", return_value=_Client(payload)):
            event = parse_review_event("Loved this but it was too acidic.", reviewed)

        recommendations = recommend_from_landscape(features, [event], top_k=2)

        self.assertEqual(recommendations[0]["coffee_id"], "lower_acidity")


if __name__ == "__main__":
    unittest.main()
