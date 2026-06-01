from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PROCESS_DATA = SRC / "process_data"

for path in (SRC, PROCESS_DATA):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import parse_review  # noqa: E402
from landscape import CoffeeFeatures  # noqa: E402
import extract_sensory  # noqa: E402
import embed_coffee  # noqa: E402
from schemas import CoffeeRecord, Process, RoastLevel  # noqa: E402


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
        parse_review.client = None
        extract_sensory.client = None
        embed_coffee.client = None

    def test_parse_review_event_sends_schema_and_default_temperature(self) -> None:
        payload = {
            "overall": 0.5,
            "change_requests": [{"attribute": "acidity", "direction": "lower", "strength": 0.3}],
            "attribute_opinions": [],
        }
        client = _ResponseClient(payload)

        with patch.object(parse_review, "get_client", return_value=client):
            event = parse_review.parse_review_event("Nice, but a little too acidic.", _coffee_features())

        self.assertEqual(event["change_requests"]["acidity"]["direction"], "lower")
        self.assertEqual(client.responses.kwargs["model"], "gpt-5.4-nano")
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

        with patch.object(extract_sensory, "get_client", return_value=client):
            sensory = extract_sensory.extract_sensory_vector_llm(_coffee_record())

        self.assertEqual(sensory.coffee_id, "coffee-1")
        self.assertEqual(client.responses.kwargs["model"], "gpt-5.4-nano")
        self.assertEqual(client.responses.kwargs["temperature"], 0.0)
        self.assertEqual(
            client.responses.kwargs["text"]["format"]["name"],
            "coffee_sensory_vector",
        )
        self.assertTrue(client.responses.kwargs["text"]["format"]["strict"])

    def test_embed_texts_uses_expected_model_and_input_payload(self) -> None:
        client = _EmbeddingClient([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])

        with patch.object(embed_coffee, "get_client", return_value=client):
            vectors = embed_coffee.embed_texts(["coffee a", "coffee b"])

        self.assertEqual(vectors, [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])
        self.assertEqual(client.embeddings.kwargs["model"], "text-embedding-3-small")
        self.assertEqual(client.embeddings.kwargs["input"], ["coffee a", "coffee b"])

    def test_embed_coffee_record_builds_expected_text_payload(self) -> None:
        client = _EmbeddingClient([[0.11, 0.22, 0.33]])

        with patch.object(embed_coffee, "get_client", return_value=client):
            vector = embed_coffee.embed_coffee_record(_coffee_record())

        self.assertEqual(vector, [0.11, 0.22, 0.33])
        self.assertEqual(client.embeddings.kwargs["model"], "text-embedding-3-small")
        self.assertIn("Name: Kenya Example", client.embeddings.kwargs["input"][0])
        self.assertIn("Process: washed", client.embeddings.kwargs["input"][0])
        self.assertIn("Tasting notes: blackberry, citrus", client.embeddings.kwargs["input"][0])


if __name__ == "__main__":
    unittest.main()
