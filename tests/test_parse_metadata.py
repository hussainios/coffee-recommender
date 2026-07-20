from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from coffee_recommender.process_data.parse_metadata import (
    parse_brew_methods,
    parse_metadata_text,
    parse_metadata,
    parse_farm,
    parse_name,
    parse_price,
    parse_process,
    parse_region,
    parse_roaster,
    parse_tasting_notes,
    parse_weight_g,
)
from coffee_recommender.schemas import BrewMethod, Process


class ParseMetadataTests(unittest.TestCase):
    def _metadata_response(self, **overrides: object) -> SimpleNamespace:
        payload = {
            "name": "Rigoberto Sanchez Pink Bourbon Washed",
            "roaster": "Shoebox Coffee",
            "origin_country": "Colombia",
            "region": "Pitalito, Huila",
            "producer": "Rigoberto Sanchez",
            "farm": None,
            "process": "washed",
            "variety": ["pink bourbon"],
            "roast_level": "unknown",
            "tasting_notes": ["floral", "stone fruit"],
            "description": "Clean and floral filter coffee.",
            "price": 12.5,
            "currency": "GBP",
            "weight_g": 125,
            "brew_methods": ["filter"],
        }
        payload.update(overrides)
        return SimpleNamespace(output_text=__import__("json").dumps(payload))

    def _mock_openai_client(self, response: SimpleNamespace) -> SimpleNamespace:
        return SimpleNamespace(
            responses=SimpleNamespace(
                create=lambda **kwargs: response,
            )
        )

    def test_parse_tasting_notes_handles_bullet_separators(self) -> None:
        text = "Tasting Notes: Mixed berries 🍓 • Lemon syrup 🍋 • Rose 🌹"

        self.assertEqual(
            parse_tasting_notes(text),
            ["mixed berries 🍓", "lemon syrup 🍋", "rose 🌹"],
        )

    def test_parse_name_uses_first_product_title_line(self) -> None:
        text = "\n".join(
            [
                "By Native",
                "Diego Bermudez - Chocolate Strudel",
                "Origin - Colombia",
                "Description",
                "Chocolate strudel is the base level from Native.",
            ]
        )

        self.assertEqual(
            parse_name(text, Path("native-diego-chocolatestrudel.txt")),
            "Diego Bermudez - Chocolate Strudel",
        )

    def test_parse_process_prefers_labelled_value(self) -> None:
        text = "Process: Washed (12 hrs held in cherry, depulped, 48 hrs dry fermentation)"

        self.assertEqual(parse_process(text), Process.WASHED)

    def test_parse_process_supports_process_method_label(self) -> None:
        text = "Process Method - Anaerobic Natural"

        self.assertEqual(parse_process(text), Process.ANAEROBIC_NATURAL)

    def test_parse_roaster_supports_by_prefix(self) -> None:
        text = "\n".join(
            [
                "By Onyx Coffee",
                "Kenya Kevote",
                "Origin - Kenya",
            ]
        )

        self.assertEqual(parse_roaster(text), "Onyx Coffee")

    def test_parse_region_and_farm_support_bullets_and_origin_fallback(self) -> None:
        text = "\n".join(
            [
                "Origin - Panama, Hacienda La Esmeralda",
                "- Region: Boquete, Chiriqui",
            ]
        )

        self.assertEqual(parse_region(text), "Boquete, Chiriqui")
        self.assertEqual(parse_farm(text), "Hacienda La Esmeralda")

    def test_parse_tasting_notes_supports_tastes_like_heading(self) -> None:
        text = "Tastes Like — Elderflower • Matcha Lemonade • Ruby Grapefruit"

        self.assertEqual(
            parse_tasting_notes(text),
            ["elderflower", "matcha lemonade", "ruby grapefruit"],
        )

    def test_parse_price_and_weight_are_numeric(self) -> None:
        text = "£14.00 / 250g"

        self.assertEqual(parse_price(text), 14.0)
        self.assertEqual(parse_weight_g(text), 250)

    def test_parse_brew_methods_detects_multiple_methods(self) -> None:
        text = "Recommended for filter and V60. Filter or Espresso? Both!"

        self.assertEqual(
            parse_brew_methods(text),
            [BrewMethod.ESPRESSO, BrewMethod.FILTER, BrewMethod.V60],
        )

    def test_parse_metadata_from_real_sample_file(self) -> None:
        file_path = Path(
            "data/raw/sigmacoffee.co.uk_products_shoebox-rigoberto-sanchez-pink-bourbon-washed-colombia.txt"
        )

        with patch(
            "coffee_recommender.process_data.extract_metadata.openai_client.get_openai_client",
            return_value=self._mock_openai_client(
                self._metadata_response(
                    name="Rigoberto Sanchez Pink Bourbon Washed",
                    roaster="Shoebox Coffee",
                    origin_country="Colombia",
                    region="Pitalito, Huila",
                    producer="Rigoberto Sanchez",
                    process="washed",
                    variety=["pink bourbon"],
                    weight_g=125,
                    brew_methods=["filter"],
                )
            ),
        ):
            coffee = parse_metadata(file_path)

        self.assertEqual(coffee.name, "Rigoberto Sanchez Pink Bourbon Washed")
        self.assertEqual(coffee.origin_country, "Colombia")
        self.assertEqual(coffee.region, "Pitalito, Huila")
        self.assertEqual(coffee.producer, "Rigoberto Sanchez")
        self.assertEqual(coffee.process, Process.WASHED)
        self.assertEqual(coffee.variety, ["pink bourbon"])
        self.assertEqual(coffee.weight_g, 125)
        self.assertEqual(coffee.brew_methods, [BrewMethod.FILTER])

    def test_parse_metadata_text_uses_llm_output_for_structured_fields(self) -> None:
        text = "\n".join(
            [
                "Ethiopia - Daye Bensa / Washed 74158 | La Tostadora",
                "Yuzu, Nectarine & Assam Tea",
                "Farmer: Daye Bensa",
            ]
        )

        with patch(
            "coffee_recommender.process_data.extract_metadata.openai_client.get_openai_client",
            return_value=self._mock_openai_client(
                self._metadata_response(
                    name="Ethiopia - Daye Bensa / Washed 74158",
                    roaster="La Tostadora",
                    origin_country="Ethiopia",
                    region="Sidama / Bensa / Arbegona Village",
                    producer="Daye Bensa",
                    process="washed",
                    variety=["74158"],
                    tasting_notes=["yuzu", "nectarine", "assam tea"],
                    description="High-altitude washed Ethiopia from Arbegona.",
                    price=13.0,
                    weight_g=None,
                )
            ),
        ):
            coffee = parse_metadata_text(text, source="sample.txt")

        self.assertEqual(coffee.roaster, "La Tostadora")
        self.assertEqual(coffee.producer, "Daye Bensa")
        self.assertEqual(coffee.tasting_notes, ["yuzu", "nectarine", "assam tea"])

    def test_parse_metadata_text_matches_file_backed_parser(self) -> None:
        file_path = Path(
            "data/raw/sigmacoffee.co.uk_products_shoebox-rigoberto-sanchez-pink-bourbon-washed-colombia.txt"
        )
        text = file_path.read_text(encoding="utf-8", errors="ignore")

        with patch(
            "coffee_recommender.process_data.extract_metadata.openai_client.get_openai_client",
            return_value=self._mock_openai_client(self._metadata_response()),
        ):
            from_file = parse_metadata(file_path)
            from_text = parse_metadata_text(text, source=str(file_path))

        self.assertEqual(from_text.model_dump(), from_file.model_dump())

    def test_parse_metadata_text_supports_url_sources(self) -> None:
        text = "\n".join(
            [
                "By Native",
                "Diego Bermudez - Chocolate Strudel",
                "Origin: Colombia",
                "Process: Washed",
                "Tasting Notes: Cherry, chocolate",
            ]
        )

        with patch(
            "coffee_recommender.process_data.extract_metadata.openai_client.get_openai_client",
            return_value=self._mock_openai_client(
                self._metadata_response(
                    name="Diego Bermudez - Chocolate Strudel",
                    roaster="Native",
                    origin_country="Colombia",
                    producer="Diego Bermudez",
                    process="washed",
                    tasting_notes=["cherry", "chocolate"],
                )
            ),
        ):
            coffee = parse_metadata_text(
                text,
                source="url:https://example.com/coffee",
                source_url="https://example.com/coffee",
            )

        self.assertEqual(coffee.source_file, "url:https://example.com/coffee")
        self.assertEqual(str(coffee.source_url), "https://example.com/coffee")
        self.assertTrue(coffee.coffee_id.startswith("diego-bermudez-chocolate-strudel-"))


if __name__ == "__main__":
    unittest.main()
