from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from coffee_recommender import openai_client, parse_review
from coffee_recommender.landscape import CoffeeFeatures
from coffee_recommender.process_data import embed_coffee, extract_sensory
from coffee_recommender.schemas import CoffeeRecord, Process, RoastLevel


class _Responses:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        return SimpleNamespace(output_text=json.dumps(self.payload))


class _ResponseClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.responses = _Responses(payload)


class _Embeddings:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=vector) for vector in self.vectors]
        )


class _EmbeddingClient:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.embeddings = _Embeddings(vectors)


def _coffee_features() -> CoffeeFeatures:
    return CoffeeFeatures(
        coffee_id="reviewed",
        name="Reviewed Coffee",
        sensory={
            "acidity": 0.8,
            "sweetness": 0.4,
            "body": 0.4,
            "bitterness": 0.1,
            "fruitiness": 0.8,
            "chocolate_nutty": 0.1,
            "floral": 0.2,
            "funky_fermented": 0.2,
            "roasty": 0.1,
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


def _coffee_record() -> CoffeeRecord:
    return CoffeeRecord(
        coffee_id="coffee-1",
        name="Kenya Example",
        roaster="April",
        origin_country="Kenya",
        region="Nyeri",
        producer="Producer A",
        process=Process.WASHED,
        variety=["sl28", "sl34"],
        roast_level=RoastLevel.LIGHT,
        tasting_notes=["blackberry", "citrus"],
        description="Bright and sweet filter coffee.",
        source_file="sample.txt",
    )


class OpenAIBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        openai_client.reset_openai_client_cache()

    def test_parse_review_event_sends_schema_and_default_temperature(self) -> None:
        payload = {
            "overall": 0.5,
            "change_requests": [{"attribute": "acidity", "direction": "lower", "strength": 0.3}],
            "attribute_opinions": [],
        }
        client = _ResponseClient(payload)

        with patch.object(openai_client, "get_openai_client", return_value=client) as get_client:
            event = parse_review.parse_review_event("Nice, but a little too acidic.", _coffee_features())

        get_client.assert_called_once_with("review parsing")
        self.assertEqual(event["change_requests"]["acidity"]["direction"], "lower")
        self.assertEqual(client.responses.kwargs["model"], openai_client.DEFAULT_CHAT_MODEL)
        self.assertEqual(client.responses.kwargs["temperature"], 0.0)
        self.assertEqual(
            client.responses.kwargs["text"]["format"]["name"],
            "coffee_review_event",
        )
        self.assertTrue(client.responses.kwargs["text"]["format"]["strict"])

    def test_extract_sensory_vector_uses_schema_and_default_temperature(self) -> None:
        payload = {
            "acidity": 0.7,
            "sweetness": 0.6,
            "body": 0.5,
            "bitterness": 0.1,
            "fruitiness": 0.8,
            "chocolate_nutty": 0.2,
            "floral": 0.3,
            "funky_fermented": 0.0,
            "roasty": 0.1,
            "clean_cup": 0.9,
            "confidence": 0.8,
            "evidence": {
                "acidity": ["bright"],
                "sweetness": ["sweet"],
                "body": [],
                "bitterness": [],
                "fruitiness": ["berry"],
                "chocolate_nutty": [],
                "floral": [],
                "funky_fermented": [],
                "roasty": [],
                "clean_cup": ["clean"],
            },
        }
        client = _ResponseClient(payload)

        with patch.object(openai_client, "get_openai_client", return_value=client) as get_client:
            sensory = extract_sensory.extract_sensory_vector_llm(_coffee_record())

        get_client.assert_called_once_with("sensory extraction")
        self.assertEqual(sensory.coffee_id, "coffee-1")
        self.assertEqual(client.responses.kwargs["model"], openai_client.DEFAULT_CHAT_MODEL)
        self.assertEqual(client.responses.kwargs["temperature"], 0.0)
        self.assertEqual(
            client.responses.kwargs["text"]["format"]["name"],
            "coffee_sensory_vector",
        )
        self.assertTrue(client.responses.kwargs["text"]["format"]["strict"])

    def test_embed_texts_uses_expected_model_and_input_payload(self) -> None:
        client = _EmbeddingClient([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])

        with patch.object(openai_client, "get_openai_client", return_value=client) as get_client:
            vectors = embed_coffee.embed_texts(["coffee a", "coffee b"])

        get_client.assert_called_once_with("coffee embeddings")
        self.assertEqual(vectors, [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])
        self.assertEqual(client.embeddings.kwargs["model"], openai_client.DEFAULT_EMBEDDING_MODEL)
        self.assertEqual(client.embeddings.kwargs["input"], ["coffee a", "coffee b"])

    def test_embed_coffee_record_builds_expected_text_payload(self) -> None:
        client = _EmbeddingClient([[0.11, 0.22, 0.33]])

        with patch.object(openai_client, "get_openai_client", return_value=client):
            vector = embed_coffee.embed_coffee_record(_coffee_record())

        self.assertEqual(vector, [0.11, 0.22, 0.33])
        self.assertEqual(client.embeddings.kwargs["model"], openai_client.DEFAULT_EMBEDDING_MODEL)
        self.assertIn("Name: Kenya Example", client.embeddings.kwargs["input"][0])
        self.assertIn("Process: washed", client.embeddings.kwargs["input"][0])
        self.assertIn("Tasting notes: blackberry, citrus", client.embeddings.kwargs["input"][0])

    def test_openai_gateway_missing_key_includes_purpose(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(openai_client, "load_dotenv", return_value=False),
        ):
            openai_client.reset_openai_client_cache()

            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY is required for review parsing"):
                openai_client.get_openai_client("review parsing")

    def test_openai_gateway_reuses_cached_client(self) -> None:
        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True),
            patch.object(openai_client, "OpenAI", side_effect=[object()]) as openai,
        ):
            openai_client.reset_openai_client_cache()

            first = openai_client.get_openai_client("review parsing")
            second = openai_client.get_openai_client("coffee embeddings")

        self.assertIs(first, second)
        openai.assert_called_once_with(api_key="test-key")


if __name__ == "__main__":
    unittest.main()
