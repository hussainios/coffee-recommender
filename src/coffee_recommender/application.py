from __future__ import annotations

import json
from dataclasses import dataclass

import plotly.io as pio

from .api_models import (
    CatalogueCoffeeSummary,
    LandscapeRequest,
    LandscapeResponse,
    ProcessUrlResponse,
    ReviewedCoffeePayload,
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
from .visualize_landscape import build_projected_score_landscape_figure


@dataclass
class ApplicationService:
    data_paths: DataPaths
    _catalogue: CatalogueData | None = None

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
        metadata_by_id = {
            str(row["coffee_id"]): row.to_dict()
            for _, row in catalogue.coffees.iterrows()
        }
        return [
            CatalogueCoffeeSummary(
                coffee_id=coffee_id,
                name=label.rsplit(" (", 1)[0],
                metadata=metadata_by_id.get(coffee_id),
            )
            for label, coffee_id in options.items()
        ]

    def get_catalogue_coffee(self, coffee_id: str) -> ReviewedCoffeePayload:
        selection = select_catalogue_reviewed_coffee(self.load_catalogue(), coffee_id)
        return ReviewedCoffeePayload(
            features=coffee_features_to_payload(selection.features),
            metadata=selection.metadata,
            sensory=selection.sensory,
            is_temporary=selection.is_temporary,
        )

    def process_url(self, url: str) -> ProcessUrlResponse:
        reviewed = prepare_url_selection(url)
        selection = selection_from_url_reviewed_coffee(reviewed)
        return ProcessUrlResponse(
            normalized_url=reviewed.url,
            reviewed_coffee=ReviewedCoffeePayload(
                features=coffee_features_to_payload(selection.features),
                metadata=selection.metadata,
                sensory=selection.sensory,
                is_temporary=selection.is_temporary,
            ),
        )

    def submit_review(self, request: SubmitReviewRequest) -> SubmitReviewResponse:
        result = submit_review(
            review_session=request.review_session,
            review_text=request.review_text,
            reviewed_coffee=payload_to_coffee_features(request.reviewed_coffee.features),
            catalogue_features=self.load_catalogue().features,
            top_k=request.top_k,
            is_temporary=request.reviewed_coffee.is_temporary,
        )
        return SubmitReviewResponse(
            event=result.event,
            review_session=result.review_session,
            recommendations=result.recommendations,
        )

    def build_landscape(self, request: LandscapeRequest) -> LandscapeResponse:
        catalogue = self.load_catalogue()
        if not request.review_session.review_events:
            return LandscapeResponse(message="Add at least one review to plot the score landscape.")

        scoring_features = build_scoring_features(
            catalogue.features,
            {
                coffee_id: payload_to_coffee_features(coffee)
                for coffee_id, coffee in request.review_session.reviewed_feature_overrides.items()
            },
        )
        figure = build_projected_score_landscape_figure(
            catalogue_features=catalogue.features,
            scoring_features=scoring_features,
            reviews=[payload_to_review_event(event) for event in request.review_session.review_events],
            top_recommendations=[
                recommendation.model_dump(mode="python")
                for recommendation in request.review_session.last_recommendations
            ],
            show_surface=request.show_surface,
        )
        if figure is None:
            return LandscapeResponse(message="Need at least three coffees to project the score landscape.")
        return LandscapeResponse(figure=json.loads(pio.to_json(figure, pretty=False)))


def create_application_service(data_paths: DataPaths | None = None) -> ApplicationService:
    return ApplicationService(data_paths=data_paths or get_data_paths())
