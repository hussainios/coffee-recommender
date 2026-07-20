from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from coffee_recommender import openai_client, reviewed_coffee_url
from coffee_recommender.landscape import CoffeeFeatures, recommend_from_landscape
from coffee_recommender.process_data.embed_coffee import build_embedding_text_from_record
from coffee_recommender.reviewed_coffee_url import (
    extract_product_text_from_html,
    fetch_url_html,
    prepare_reviewed_coffee_from_url,
)
from coffee_recommender.schemas import BrewMethod, CoffeeRecord, Process, RoastLevel, SensoryVector

ROOT = Path(__file__).resolve().parents[1]


class ReviewedCoffeeUrlTests(unittest.TestCase):
    def _fake_coffee(
        self,
        *,
        name: str = "Diego Bermudez - Chocolate Strudel",
        process: Process = Process.WASHED,
        source_url: str = "https://example.com/coffee",
    ) -> CoffeeRecord:
        return CoffeeRecord(
            coffee_id="diego-bermudez-chocolate-strudel-123456789abc",
            name=name,
            roaster="Native",
            origin_country="Colombia",
            region="Huila",
            producer="Diego Bermudez",
            farm=None,
            process=process,
            variety=["castillo", "caturra"],
            roast_level=RoastLevel.UNKNOWN,
            tasting_notes=["cherry", "chocolate", "brown sugar"],
            description="Filter coffee roasted for clarity and sweetness.",
            price=18.5,
            currency="GBP",
            weight_g=250,
            brew_methods=[BrewMethod.FILTER],
            source_url=source_url,
            source_file=f"url:{source_url}",
        )

    def _fake_sensory(self, **overrides: float | str) -> SensoryVector:
        payload = {
            "coffee_id": "placeholder",
            "acidity": 0.6,
            "sweetness": 0.7,
            "body": 0.5,
            "bitterness": 0.2,
            "fruitiness": 0.8,
            "chocolate_nutty": 0.6,
            "floral": 0.1,
            "funky_fermented": 0.0,
            "roasty": 0.1,
            "clean_cup": 0.8,
            "confidence": 0.9,
        }
        payload.update(overrides)
        payload["evidence"] = {
            dimension: []
            for dimension in (
                "acidity",
                "sweetness",
                "body",
                "bitterness",
                "fruitiness",
                "chocolate_nutty",
                "floral",
                "funky_fermented",
                "roasty",
                "clean_cup",
            )
        }
        return SensoryVector(**payload)

    def test_extract_product_text_from_html_rejects_thin_pages(self) -> None:
        html = "<html><body><h1>Hi</h1><p>Too short.</p></body></html>"

        with self.assertRaisesRegex(ValueError, "enough visible text"):
            extract_product_text_from_html(html)

    def test_fetch_url_html_rejects_non_http_urls(self) -> None:
        with self.assertRaisesRegex(ValueError, "http or https"):
            fetch_url_html("ftp://example.com/coffee")

    def test_prepare_reviewed_coffee_from_url_builds_temporary_features(self) -> None:
        html = """
        <html>
          <body>
            <h1>Diego Bermudez - Chocolate Strudel</h1>
            <p>Origin: Colombia</p>
            <p>Producer: Diego Bermudez</p>
            <p>Process: Washed</p>
            <p>Tasting Notes: Cherry, Chocolate, Brown sugar</p>
            <p>Filter coffee roasted for clarity and sweetness.</p>
            <p>Available in 250g bags for £18.50.</p>
          </body>
        </html>
        """
        processed_dir = ROOT / "data" / "processed"
        before_files = sorted(path.name for path in processed_dir.iterdir())
        fake_sensory = self._fake_sensory()

        with (
            patch.object(reviewed_coffee_url, "fetch_url_html", return_value=("https://example.com/coffee", html)),
            patch.object(reviewed_coffee_url, "parse_metadata_text", return_value=self._fake_coffee()),
            patch.object(reviewed_coffee_url, "extract_sensory_vector_llm", return_value=fake_sensory),
            patch.object(reviewed_coffee_url, "embed_coffee_record", return_value=[0.1, 0.2, 0.3]),
        ):
            prepared = prepare_reviewed_coffee_from_url("https://example.com/coffee")

        after_files = sorted(path.name for path in processed_dir.iterdir())

        self.assertEqual(before_files, after_files)
        self.assertEqual(prepared.url, "https://example.com/coffee")
        self.assertEqual(str(prepared.coffee.source_url), "https://example.com/coffee")
        self.assertEqual(prepared.coffee.source_file, "url:https://example.com/coffee")
        self.assertEqual(prepared.coffee.process, Process.WASHED)
        self.assertEqual(prepared.features.coffee_id, prepared.coffee.coffee_id)
        self.assertEqual(prepared.features.embedding, [0.1, 0.2, 0.3])

    def test_url_record_embedding_text_handles_list_fields(self) -> None:
        html = """
        <html>
          <body>
            <h1>Diego Bermudez - Chocolate Strudel</h1>
            <p>Origin: Colombia</p>
            <p>Producer: Diego Bermudez</p>
            <p>Process: Washed</p>
            <p>Variety: Castillo, Caturra</p>
            <p>Tasting Notes: Cherry, Chocolate, Brown sugar</p>
            <p>Filter coffee roasted for clarity and sweetness.</p>
          </body>
        </html>
        """

        with (
            patch.object(reviewed_coffee_url, "fetch_url_html", return_value=("https://example.com/coffee", html)),
            patch.object(reviewed_coffee_url, "parse_metadata_text", return_value=self._fake_coffee()),
            patch.object(reviewed_coffee_url, "extract_sensory_vector_llm", return_value=self._fake_sensory()),
            patch.object(reviewed_coffee_url, "embed_coffee_record", return_value=[0.1, 0.2, 0.3]),
        ):
            prepared = prepare_reviewed_coffee_from_url("https://example.com/coffee")

        embedding_text = build_embedding_text_from_record(prepared.coffee)

        self.assertIn("Variety: castillo, caturra", embedding_text)
        self.assertIn("Tasting notes: cherry, chocolate, brown sugar", embedding_text)

    def test_build_reviewed_coffee_from_url_returns_features_only(self) -> None:
        html = """
        <html><body>
        <h1>Example Coffee</h1>
        <p>Origin: Kenya</p>
        <p>Process: Washed</p>
        <p>Tasting Notes: Blackberry, Citrus, Tea</p>
        <p>This coffee is sweet, bright, and made for filter brewing with lots of clarity.</p>
        </body></html>
        """
        fake_sensory = self._fake_sensory(
            acidity=0.7,
            sweetness=0.6,
            body=0.4,
            bitterness=0.1,
            chocolate_nutty=0.1,
            floral=0.4,
            roasty=0.0,
            clean_cup=0.9,
            confidence=0.8,
        )

        with (
            patch.object(reviewed_coffee_url, "fetch_url_html", return_value=("https://example.com/kenya", html)),
            patch.object(
                reviewed_coffee_url,
                "parse_metadata_text",
                return_value=self._fake_coffee(name="Example Coffee", process=Process.WASHED, source_url="https://example.com/kenya"),
            ),
            patch.object(reviewed_coffee_url, "extract_sensory_vector_llm", return_value=fake_sensory),
            patch.object(reviewed_coffee_url, "embed_coffee_record", return_value=[0.3, 0.2, 0.1]),
        ):
            features = prepare_reviewed_coffee_from_url("https://example.com/kenya").features

        self.assertEqual(features.name, "Example Coffee")
        self.assertEqual(features.process["process_washed"], 1.0)

    def test_temporary_reviewed_coffee_can_shape_ranking_without_becoming_candidate(self) -> None:
        temp_reviewed = CoffeeFeatures(
            coffee_id="temp-reviewed",
            name="Temp Reviewed",
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
            sensory=temp_reviewed.sensory.copy(),
            process=temp_reviewed.process.copy(),
            embedding=[0.99, 0.01, 0.0],
        )
        far = CoffeeFeatures(
            coffee_id="far",
            name="Far",
            sensory={**temp_reviewed.sensory, "acidity": 0.1, "fruitiness": 0.1, "clean_cup": 0.1},
            process={
                "process_washed": 0.0,
                "process_natural": 1.0,
                "process_honey": 0.0,
                "process_anaerobic": 0.0,
                "process_cofermented": 0.0,
            },
            embedding=[0.0, 1.0, 0.0],
        )

        recommendations = recommend_from_landscape(
            {
                temp_reviewed.coffee_id: temp_reviewed,
                candidate.coffee_id: candidate,
                far.coffee_id: far,
            },
            [{"coffee_id": temp_reviewed.coffee_id, "overall": 1.0}],
            top_k=2,
        )

        self.assertEqual(recommendations[0]["coffee_id"], "candidate")
        self.assertNotIn(temp_reviewed.coffee_id, [item["coffee_id"] for item in recommendations])

    def test_url_flow_uses_zero_temperature_for_sensory_extraction(self) -> None:
        html = """
        <html><body>
        <h1>Example Coffee</h1>
        <p>Origin: Kenya</p>
        <p>Process: Washed</p>
        <p>Tasting Notes: Blackberry, Citrus, Tea</p>
        <p>This coffee is sweet, bright, and made for filter brewing with lots of clarity.</p>
        </body></html>
        """

        with (
            patch.object(reviewed_coffee_url, "fetch_url_html", return_value=("https://example.com/kenya", html)),
            patch.object(
                reviewed_coffee_url,
                "parse_metadata_text",
                return_value=self._fake_coffee(name="Example Coffee", process=Process.WASHED, source_url="https://example.com/kenya"),
            ),
            patch.object(reviewed_coffee_url, "extract_sensory_vector_llm", return_value=self._fake_sensory()) as extract,
            patch.object(reviewed_coffee_url, "embed_coffee_record", return_value=[0.3, 0.2, 0.1]),
        ):
            prepare_reviewed_coffee_from_url("https://example.com/kenya")

        self.assertEqual(extract.call_args.kwargs["model"], openai_client.DEFAULT_CHAT_MODEL)
        self.assertEqual(extract.call_args.kwargs["temperature"], 0.0)


if __name__ == "__main__":
    unittest.main()
