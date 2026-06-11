from __future__ import annotations

import json
import math
from ast import literal_eval
from dataclasses import dataclass, field

import plotly.io as pio

from .api_models import (
    CatalogueCoffeeSummary,
    LandscapeResponse,
    ReviewedCoffeeDetails,
    ReviewSessionPayload,
    SubmitReviewRequest,
    SubmitReviewResponse,
    coffee_features_to_payload,
    payload_to_coffee_features,
    payload_to_review_event,
)
from .coffee_service import (
    CatalogueData,
    build_coffee_options,
    build_scoring_features,
    load_catalogue,
    prepare_url_selection,
    select_catalogue_reviewed_coffee,
    selection_from_url_reviewed_coffee,
    submit_review,
)
from .config import DataPaths, get_data_paths
from .review_session import create_review_session
from .visualize_landscape import build_projected_score_landscape_figure


@dataclass
class ApplicationService:
    data_paths: DataPaths
    _catalogue: CatalogueData | None = None
    _review_session: ReviewSessionPayload = field(default_factory=create_review_session)

    def load_catalogue(self) -> CatalogueData:
        if self._catalogue is None:
            self._catalogue = load_catalogue(
                self.data_paths.coffees_path,
                self.data_paths.sensory_path,
                self.data_paths.embeddings_path,
            )
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

    def get_review_session(self) -> ReviewSessionPayload:
        return self._review_session

    def clear_review_session(self) -> ReviewSessionPayload:
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
        result = submit_review(
            review_session=self._review_session,
            review_text=request.review_text,
            reviewed_coffee=payload_to_coffee_features(request.reviewed_coffee.features),
            catalogue_features=catalogue.features,
            top_k=request.top_k,
            is_external_url=request.reviewed_coffee.source_type == "external_url",
        )
        self._review_session = result.review_session
        metadata_lookup = catalogue.coffees.set_index("coffee_id").to_dict(orient="index")

        enriched_recommendations = []
        for recommendation in result.recommendations:
            metadata = metadata_lookup.get(recommendation.coffee_id, {})
            tasting_notes_raw = metadata.get("tasting_notes")
            tasting_notes = []
            if isinstance(tasting_notes_raw, str) and tasting_notes_raw.strip():
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

        return SubmitReviewResponse(
            event=result.event,
            review_session=ReviewSessionPayload(
                review_events=result.review_session.review_events,
                reviewed_feature_overrides=result.review_session.reviewed_feature_overrides,
                last_event=result.review_session.last_event,
                last_recommendations=enriched_recommendations,
            ),
            recommendations=enriched_recommendations,
        )

    def build_landscape(self, show_surface: bool = True) -> LandscapeResponse:
        catalogue = self.load_catalogue()
        if not self._review_session.review_events:
            return LandscapeResponse(message="Add at least one review to plot the score landscape.")

        scoring_features = build_scoring_features(
            catalogue.features,
            {
                coffee_id: payload_to_coffee_features(coffee)
                for coffee_id, coffee in self._review_session.reviewed_feature_overrides.items()
            },
        )
        figure = build_projected_score_landscape_figure(
            catalogue_features=catalogue.features,
            scoring_features=scoring_features,
            reviews=[payload_to_review_event(event) for event in self._review_session.review_events],
            top_recommendations=[
                recommendation.model_dump(mode="python")
                for recommendation in self._review_session.last_recommendations
            ],
            show_surface=show_surface,
        )
        if figure is None:
            return LandscapeResponse(message="Need at least three coffees to project the score landscape.")
        return LandscapeResponse(figure=json.loads(pio.to_json(figure, pretty=False)))


def create_application_service(data_paths: DataPaths | None = None) -> ApplicationService:
    return ApplicationService(data_paths=data_paths or get_data_paths())


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text or None
