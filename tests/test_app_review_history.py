from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app_state import (  # noqa: E402
    append_review_event,
    build_scoring_features,
    initialise_review_state,
    reset_review_history,
    reset_review_history_if_data_paths_changed,
)
from landscape import CoffeeFeatures, ReviewEvent  # noqa: E402


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


class ReviewHistoryHelperTests(unittest.TestCase):
    def test_initialise_review_state_adds_missing_defaults(self) -> None:
        state = SimpleNamespace()

        initialise_review_state(state)

        self.assertIsNone(state.last_event)
        self.assertEqual(state.last_recommendations, [])
        self.assertEqual(state.review_events, [])
        self.assertEqual(state.reviewed_feature_overrides, {})
        self.assertIsNone(state.url_reviewed_coffee)
        self.assertEqual(state.url_reviewed_source, "")
        self.assertEqual(state.input_mode, "Catalogue coffee")
        self.assertIsNone(state.data_paths_key)

    def test_append_catalogue_review_updates_events_without_override(self) -> None:
        state = SimpleNamespace(
            review_events=[],
            reviewed_feature_overrides={},
            last_event=None,
            last_recommendations=[],
        )
        event: ReviewEvent = {"coffee_id": "catalogue", "overall": 1.0}

        append_review_event(state, event, _coffee("catalogue"), is_temporary=False)

        self.assertEqual(state.review_events, [event])
        self.assertEqual(state.last_event, event)
        self.assertEqual(state.reviewed_feature_overrides, {})

    def test_append_url_review_stores_feature_override(self) -> None:
        state = SimpleNamespace(
            review_events=[],
            reviewed_feature_overrides={},
            last_event=None,
            last_recommendations=[],
        )
        coffee = _coffee("temporary-url")
        event: ReviewEvent = {"coffee_id": coffee.coffee_id, "overall": -0.5}

        append_review_event(state, event, coffee, is_temporary=True)

        self.assertEqual(state.review_events, [event])
        self.assertEqual(state.reviewed_feature_overrides[coffee.coffee_id], coffee)

    def test_build_scoring_features_merges_overrides(self) -> None:
        catalogue = _coffee("catalogue")
        temporary = _coffee("temporary")

        features = build_scoring_features(
            {catalogue.coffee_id: catalogue},
            {temporary.coffee_id: temporary},
        )

        self.assertEqual(features["catalogue"], catalogue)
        self.assertEqual(features["temporary"], temporary)

    def test_reset_review_history_clears_review_state(self) -> None:
        state = SimpleNamespace(
            review_events=[{"coffee_id": "a"}],
            reviewed_feature_overrides={"a": _coffee("a")},
            last_event={"coffee_id": "a"},
            last_recommendations=[{"coffee_id": "b"}],
        )

        reset_review_history(state)

        self.assertEqual(state.review_events, [])
        self.assertEqual(state.reviewed_feature_overrides, {})
        self.assertIsNone(state.last_event)
        self.assertEqual(state.last_recommendations, [])

    def test_reset_review_history_if_data_paths_changed_preserves_or_resets_state(self) -> None:
        state = SimpleNamespace(
            data_paths_key=None,
            review_events=[{"coffee_id": "a"}],
            reviewed_feature_overrides={"a": _coffee("a")},
            last_event={"coffee_id": "a"},
            last_recommendations=[{"coffee_id": "b"}],
        )

        reset_review_history_if_data_paths_changed(state, ("coffees.csv", "sensory.csv", "embeddings.csv"))

        self.assertEqual(state.data_paths_key, ("coffees.csv", "sensory.csv", "embeddings.csv"))
        self.assertEqual(state.review_events, [{"coffee_id": "a"}])

        reset_review_history_if_data_paths_changed(state, ("new.csv", "sensory.csv", "embeddings.csv"))

        self.assertEqual(state.data_paths_key, ("new.csv", "sensory.csv", "embeddings.csv"))
        self.assertEqual(state.review_events, [])
        self.assertEqual(state.reviewed_feature_overrides, {})


if __name__ == "__main__":
    unittest.main()
