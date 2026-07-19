from __future__ import annotations

import json
import math
from ast import literal_eval
from dataclasses import dataclass, field

import plotly.io as pio

from .api_models import (
    CatalogueCoffeeSummary,
    LandscapeResponse,
    RecommendationRunPayload,
    ReviewHistoryItemPayload,
    ReviewedCoffeeDetails,
    ReviewSessionPayload,
    SubmitReviewRequest,
    SubmitReviewResponse,
    coffee_features_to_payload,
    payload_to_coffee_features,
    payload_to_review_event,
)
from .catalogue_store import CatalogueStore, CsvCatalogueStore, SqlAlchemyCatalogueStore
from .coffee_service import (
    CatalogueData,
    build_coffee_options,
    build_scoring_features,
    prepare_url_selection,
    select_catalogue_reviewed_coffee,
    selection_from_url_reviewed_coffee,
    submit_review,
)
from .config import DataPaths, get_data_paths, get_optional_database_url
from .db import SqlAlchemyReviewHistoryStore, create_session_factory
from .db.review_history import ReviewHistoryStore
from .review_session import create_review_session
from .visualize_landscape import build_projected_score_landscape_figure

RECOMMENDATION_ALGORITHM_VERSION = "landscape_v1"


@dataclass
class ApplicationService:
    data_paths: DataPaths
    _catalogue: CatalogueData | None = None
    _review_session: ReviewSessionPayload = field(default_factory=create_review_session)
    _review_history_store: ReviewHistoryStore | None = None
    _catalogue_store: CatalogueStore | None = None

    def load_catalogue(self) -> CatalogueData:
        if self._catalogue is None:
            if self._catalogue_store is None:
                self._catalogue_store = CsvCatalogueStore(self.data_paths)
            self._catalogue = self._catalogue_store.load_catalogue()
        return self._catalogue

    def list_catalogue_coffees(self) -> list[CatalogueCoffeeSummary]:
        catalogue = self.load_catalogue()
        options = build_coffee_options(catalogue.coffees)
        return [
            CatalogueCoffeeSummary(
                coffee_id=coffee_id,
                name=label.rsplit(" (", 1)[0],
            )
            for label, coffee_id in options.items()
        ]

    def list_reviews(self) -> list[ReviewHistoryItemPayload]:
        if self._review_history_store is None:
            return []
        return self._review_history_store.list_reviews()

    def list_recommendation_runs(self) -> list[RecommendationRunPayload]:
        if self._review_history_store is None:
            return []
        return self._review_history_store.list_recommendation_runs()

    def get_recommendation_run(self, run_id: int) -> RecommendationRunPayload:
        if self._review_history_store is None:
            raise KeyError(f"Recommendation run not found: {run_id}")
        return self._review_history_store.get_recommendation_run(run_id)

    def get_review_session(self) -> ReviewSessionPayload:
        if self._review_history_store is not None:
            return self._review_history_store.get_review_session()
        return self._review_session

    def clear_review_session(self) -> ReviewSessionPayload:
        if self._review_history_store is not None:
            return self._review_history_store.clear_review_session()
        self._review_session = create_review_session()
        return self._review_session

    def get_catalogue_reviewed_coffee(self, coffee_id: str) -> ReviewedCoffeeDetails:
        selection = select_catalogue_reviewed_coffee(self.load_catalogue(), coffee_id)
        return ReviewedCoffeeDetails(
            features=coffee_features_to_payload(selection.features),
            metadata=selection.metadata,
            sensory=selection.sensory,
            source_type="catalogue",
        )

    def get_reviewed_coffee_from_url(self, url: str) -> ReviewedCoffeeDetails:
        reviewed = prepare_url_selection(url)
        selection = selection_from_url_reviewed_coffee(reviewed)
        return ReviewedCoffeeDetails(
            features=coffee_features_to_payload(selection.features),
            metadata=selection.metadata,
            sensory=selection.sensory,
            source_type="external_url",
            normalized_url=reviewed.url,
        )

    def submit_review(self, request: SubmitReviewRequest) -> SubmitReviewResponse:
        catalogue = self.load_catalogue()
        current_review_session = self.get_review_session()
        result = submit_review(
            review_session=current_review_session,
            review_text=request.review_text,
            reviewed_coffee=payload_to_coffee_features(request.reviewed_coffee.features),
            catalogue_features=catalogue.features,
            top_k=request.top_k,
            is_external_url=request.reviewed_coffee.source_type == "external_url",
        )
        metadata_lookup = catalogue.coffees.set_index("coffee_id").to_dict(orient="index")

        enriched_recommendations: list = []
        for recommendation in result.recommendations:
            metadata = metadata_lookup.get(recommendation.coffee_id, {})
            tasting_notes_raw = metadata.get("tasting_notes")
            tasting_notes = []
            if isinstance(tasting_notes_raw, list):
                tasting_notes = [str(note) for note in tasting_notes_raw if str(note).strip()]
            elif isinstance(tasting_notes_raw, str) and tasting_notes_raw.strip():
                try:
                    parsed_notes = literal_eval(tasting_notes_raw)
                    if isinstance(parsed_notes, list):
                        tasting_notes = [str(note) for note in parsed_notes if str(note).strip()]
                except (ValueError, SyntaxError):
                    tasting_notes = []

            enriched_recommendations.append(
                recommendation.model_copy(
                    update={
                        "roaster": _optional_string(metadata.get("roaster")),
                        "origin_country": _optional_string(metadata.get("origin_country")),
                        "producer": _optional_string(metadata.get("producer")),
                        "process": _optional_string(metadata.get("process")),
                        "tasting_notes": tasting_notes,
                        "source_url": _optional_string(metadata.get("source_url")),
                    }
                )
            )

        persisted_session = result.review_session
        if self._review_history_store is not None:
            persisted_session = self._review_history_store.persist_review_submission(
                review_text=request.review_text,
                reviewed_coffee=request.reviewed_coffee,
                event=result.event,
                recommendations=enriched_recommendations,
                algorithm_version=RECOMMENDATION_ALGORITHM_VERSION,
            )
        else:
            self._review_session = ReviewSessionPayload(
                review_events=result.review_session.review_events,
                reviewed_feature_overrides=result.review_session.reviewed_feature_overrides,
                last_event=result.review_session.last_event,
                last_recommendations=enriched_recommendations,
            )

        return SubmitReviewResponse(
            event=result.event,
            review_session=persisted_session,
            recommendations=enriched_recommendations,
        )

    def build_landscape(self, show_surface: bool = True) -> LandscapeResponse:
        catalogue = self.load_catalogue()
        review_session = self.get_review_session()
        if not review_session.review_events:
            return LandscapeResponse(message="Add at least one review to plot the score landscape.")

        scoring_features = build_scoring_features(
            catalogue.features,
            {
                coffee_id: payload_to_coffee_features(coffee)
                for coffee_id, coffee in review_session.reviewed_feature_overrides.items()
            },
        )
        figure = build_projected_score_landscape_figure(
            catalogue_features=catalogue.features,
            scoring_features=scoring_features,
            reviews=[payload_to_review_event(event) for event in review_session.review_events],
            top_recommendations=[
                recommendation.model_dump(mode="python")
                for recommendation in review_session.last_recommendations
            ],
            show_surface=show_surface,
        )
        if figure is None:
            return LandscapeResponse(message="Need at least three coffees to project the score landscape.")
        return LandscapeResponse(figure=json.loads(pio.to_json(figure, pretty=False)))


def create_application_service(data_paths: DataPaths | None = None) -> ApplicationService:
    review_history_store: ReviewHistoryStore | None = None
    catalogue_store: CatalogueStore | None = None
    if get_optional_database_url():
        session_factory = create_session_factory()
        review_history_store = SqlAlchemyReviewHistoryStore(session_factory)
        catalogue_store = SqlAlchemyCatalogueStore(session_factory)
    return ApplicationService(
        data_paths=data_paths or get_data_paths(),
        _review_history_store=review_history_store,
        _catalogue_store=catalogue_store,
    )


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text or None
