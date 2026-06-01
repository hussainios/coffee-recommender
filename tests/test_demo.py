from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import demo  # noqa: E402
from landscape import CoffeeFeatures  # noqa: E402


def _coffee(coffee_id: str) -> CoffeeFeatures:
    return CoffeeFeatures(
        coffee_id=coffee_id,
        name=coffee_id.replace("_", " ").title(),
        sensory={
            "acidity": 0.5,
            "sweetness": 0.5,
            "body": 0.5,
            "bitterness": 0.1,
            "fruitiness": 0.5,
            "chocolate_nutty": 0.1,
            "floral": 0.1,
            "funky_fermented": 0.0,
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


class DemoTests(unittest.TestCase):
    def test_main_loads_feature_index_with_embeddings_path(self) -> None:
        reviewed = _coffee("reviewed")
        candidate = _coffee("candidate")
        event = {"coffee_id": reviewed.coffee_id, "overall": 1.0}

        with (
            patch.object(
                demo,
                "load_feature_index",
                return_value={reviewed.coffee_id: reviewed, candidate.coffee_id: candidate},
            ) as load_feature_index,
            patch.object(demo, "parse_review_event", return_value=event),
            patch.object(demo, "recommend_from_landscape", return_value=[]),
            patch("builtins.print"),
        ):
            demo.main(
                review="Loved it.",
                reviewed_coffee_id="reviewed",
                coffees_path=Path("coffees.csv"),
                sensory_path=Path("sensory.csv"),
                embeddings_path=Path("embeddings.csv"),
                top_k=5,
            )

        load_feature_index.assert_called_once_with(
            Path("coffees.csv"),
            Path("sensory.csv"),
            Path("embeddings.csv"),
        )


if __name__ == "__main__":
    unittest.main()
