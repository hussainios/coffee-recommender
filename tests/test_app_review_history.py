from __future__ import annotations

import unittest
from types import SimpleNamespace

from coffee_recommender.api_models import (
    CoffeeFeaturesPayload,
    RecommendationPayload,
    ReviewEventPayload,
)
from coffee_recommender.app_state import initialise_review_state, reset_review_history
from coffee_recommender.review_session import append_review_to_session, create_review_session


def _coffee(coffee_id: str) -> CoffeeFeaturesPayload:
    return CoffeeFeaturesPayload(
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
    def test_initialise_review_state_adds_review_session_defaults(self) -> None:
        state = SimpleNamespace()

        initialise_review_state(state)

        self.assertEqual(state.review_session.review_events, [])
        self.assertEqual(state.review_session.reviewed_feature_overrides, {})
        self.assertIsNone(state.url_reviewed_coffee)
        self.assertEqual(state.url_reviewed_source, "")
        self.assertEqual(state.input_mode, "Catalogue coffee")

    def test_append_catalogue_review_updates_session_without_override(self) -> None:
        session = create_review_session()
        event = ReviewEventPayload(coffee_id="catalogue", overall=1.0)
        recommendations = [RecommendationPayload(coffee_id="candidate", name="Candidate", score=0.9, temperature=0.2)]

        updated = append_review_to_session(
            session,
            event,
            _coffee("catalogue"),
            is_temporary=False,
            recommendations=recommendations,
        )

        self.assertEqual(updated.review_events, [event])
        self.assertEqual(updated.last_event, event)
        self.assertEqual(updated.reviewed_feature_overrides, {})
        self.assertEqual(updated.last_recommendations, recommendations)

    def test_append_url_review_stores_feature_override(self) -> None:
        session = create_review_session()
        coffee = _coffee("temporary-url")
        event = ReviewEventPayload(coffee_id=coffee.coffee_id, overall=-0.5)

        updated = append_review_to_session(
            session,
            event,
            coffee,
            is_temporary=True,
            recommendations=[],
        )

        self.assertEqual(updated.review_events, [event])
        self.assertEqual(updated.reviewed_feature_overrides[coffee.coffee_id], coffee)

    def test_reset_review_history_replaces_session_payload(self) -> None:
        state = SimpleNamespace()
        initialise_review_state(state)
        state.review_session = append_review_to_session(
            state.review_session,
            ReviewEventPayload(coffee_id="a", overall=1.0),
            _coffee("a"),
            is_temporary=True,
            recommendations=[RecommendationPayload(coffee_id="b", name="B", score=0.5, temperature=0.2)],
        )

        reset_review_history(state)

        self.assertEqual(state.review_session.review_events, [])
        self.assertEqual(state.review_session.reviewed_feature_overrides, {})
        self.assertIsNone(state.review_session.last_event)
        self.assertEqual(state.review_session.last_recommendations, [])


if __name__ == "__main__":
    unittest.main()
