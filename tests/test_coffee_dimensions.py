from __future__ import annotations

import unittest

from coffee_recommender.coffee_dimensions import PROCESS_DIMENSIONS, SENSORY_DIMENSIONS
from coffee_recommender.process_data.extract_sensory import SENSORY_JSON_SCHEMA


class CoffeeDimensionsTests(unittest.TestCase):
    def test_sensory_extraction_schema_uses_shared_dimensions(self) -> None:
        evidence_schema = SENSORY_JSON_SCHEMA["properties"]["evidence"]

        self.assertEqual(
            tuple(SENSORY_JSON_SCHEMA["properties"][dimension]["type"] for dimension in SENSORY_DIMENSIONS),
            tuple("number" for _ in SENSORY_DIMENSIONS),
        )
        self.assertEqual(tuple(evidence_schema["required"]), SENSORY_DIMENSIONS)

    def test_process_dimensions_preserve_expected_order(self) -> None:
        self.assertEqual(
            PROCESS_DIMENSIONS,
            (
                "process_washed",
                "process_natural",
                "process_honey",
                "process_anaerobic",
                "process_cofermented",
            ),
        )


if __name__ == "__main__":
    unittest.main()
