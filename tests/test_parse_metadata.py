from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from process_data.parse_metadata import (  # noqa: E402
    parse_brew_methods,
    parse_metadata_text,
    parse_metadata,
    parse_name,
    parse_price,
    parse_process,
    parse_tasting_notes,
    parse_weight_g,
)
from schemas import BrewMethod, Process  # noqa: E402


class ParseMetadataTests(unittest.TestCase):
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

        coffee = parse_metadata(file_path)

        self.assertEqual(coffee.name, "Rigoberto Sánchez Pink Bourbon Washed | Colombia")
        self.assertEqual(coffee.origin_country, "Colombia")
        self.assertIsNone(coffee.region)
        self.assertEqual(coffee.producer, "Rigoberto Sánchez")
        self.assertEqual(coffee.process, Process.WASHED)
        self.assertEqual(coffee.variety, ["pink bourbon"])
        self.assertEqual(coffee.weight_g, 125)
        self.assertEqual(coffee.brew_methods, [BrewMethod.FILTER])

    def test_parse_metadata_text_matches_file_backed_parser(self) -> None:
        file_path = Path(
            "data/raw/sigmacoffee.co.uk_products_shoebox-rigoberto-sanchez-pink-bourbon-washed-colombia.txt"
        )
        text = file_path.read_text(encoding="utf-8", errors="ignore")

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
