from __future__ import annotations

import json
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
        result = submit_review(
            review_session=self._review_session,
            review_text=request.review_text,
            reviewed_coffee=payload_to_coffee_features(request.reviewed_coffee.features),
            catalogue_features=self.load_catalogue().features,
            top_k=request.top_k,
            is_external_url=request.reviewed_coffee.source_type == "external_url",
        )
        self._review_session = result.review_session
        return SubmitReviewResponse(
            event=result.event,
            review_session=result.review_session,
            recommendations=result.recommendations,
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
