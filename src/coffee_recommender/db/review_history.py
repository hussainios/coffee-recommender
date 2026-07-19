from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from typing import Protocol

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from ..api_models import (
    CoffeeFeaturesPayload,
    RecommendationPayload,
    RecommendationRunPayload,
    ReviewedCoffeeDetails,
    ReviewEventPayload,
    ReviewHistoryItemPayload,
    ReviewSessionPayload,
)
from ..config import get_default_user_email
from .models import (
    RecommendationItemModel,
    RecommendationRunModel,
    ReviewEventModel,
    UserModel,
)


class ReviewHistoryStore(Protocol):
    def get_review_session(self) -> ReviewSessionPayload: ...

    def clear_review_session(self) -> ReviewSessionPayload: ...

    def persist_review_submission(
        self,
        *,
        review_text: str,
        reviewed_coffee: ReviewedCoffeeDetails,
        event: ReviewEventPayload,
        recommendations: list[RecommendationPayload],
        algorithm_version: str,
    ) -> ReviewSessionPayload: ...

    def list_reviews(self) -> list[ReviewHistoryItemPayload]: ...

    def list_recommendation_runs(self) -> list[RecommendationRunPayload]: ...

    def get_recommendation_run(self, run_id: int) -> RecommendationRunPayload: ...


class SqlAlchemyReviewHistoryStore:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        user_email_provider: Callable[[], str] = get_default_user_email,
    ) -> None:
        self._session_factory = session_factory
        self._user_email_provider = user_email_provider

    def get_review_session(self) -> ReviewSessionPayload:
        with self._session_factory() as session:
            user = self._get_or_create_user(session)
            review_events = self._load_review_events(session, user.id)
            last_recommendations = self._load_last_recommendations(session, user.id)
            reviewed_feature_overrides: dict[str, CoffeeFeaturesPayload] = {}

            for persisted in review_events:
                payload = self._get_review_payload(session, persisted.id)
                if payload is None:
                    continue
                if payload.get("source_type") != "external_url":
                    continue
                reviewed_coffee_payload = payload.get("reviewed_coffee")
                if not isinstance(reviewed_coffee_payload, dict):
                    continue
                features = reviewed_coffee_payload.get("features")
                if isinstance(features, dict):
                    coffee_features = CoffeeFeaturesPayload.model_validate(features)
                    reviewed_feature_overrides[coffee_features.coffee_id] = coffee_features

            return ReviewSessionPayload(
                review_events=[self._to_review_event_payload(event) for event in review_events],
                reviewed_feature_overrides=reviewed_feature_overrides,
                last_event=self._to_review_event_payload(review_events[-1]) if review_events else None,
                last_recommendations=last_recommendations,
            )

    def clear_review_session(self) -> ReviewSessionPayload:
        with self._session_factory() as session:
            user = self._get_or_create_user(session)
            run_ids = session.scalars(
                select(RecommendationRunModel.id).where(RecommendationRunModel.user_id == user.id)
            ).all()
            if run_ids:
                session.execute(
                    delete(RecommendationItemModel).where(
                        RecommendationItemModel.recommendation_run_id.in_(run_ids)
                    )
                )
            session.execute(
                delete(RecommendationRunModel).where(RecommendationRunModel.user_id == user.id)
            )
            session.execute(delete(ReviewEventModel).where(ReviewEventModel.user_id == user.id))
            session.commit()

        return ReviewSessionPayload()

    def persist_review_submission(
        self,
        *,
        review_text: str,
        reviewed_coffee: ReviewedCoffeeDetails,
        event: ReviewEventPayload,
        recommendations: list[RecommendationPayload],
        algorithm_version: str,
    ) -> ReviewSessionPayload:
        with self._session_factory() as session:
            user = self._get_or_create_user(session)
            review_event = ReviewEventModel(
                user_id=user.id,
                coffee_id=event.coffee_id,
                review_text=review_text,
                overall_score=Decimal(str(event.overall)),
                parsed_review_json={
                    "event": event.model_dump(mode="python"),
                    "reviewed_coffee": reviewed_coffee.model_dump(mode="python"),
                    "source_type": reviewed_coffee.source_type,
                    "normalized_url": reviewed_coffee.normalized_url,
                },
            )
            session.add(review_event)
            session.flush()

            recommendation_run = RecommendationRunModel(
                user_id=user.id,
                seed_review_event_id=review_event.id,
                algorithm_version=algorithm_version,
            )
            session.add(recommendation_run)
            session.flush()

            for rank, recommendation in enumerate(recommendations, start=1):
                session.add(
                    RecommendationItemModel(
                        recommendation_run_id=recommendation_run.id,
                        coffee_id=recommendation.coffee_id,
                        rank=rank,
                        score=Decimal(str(recommendation.score)),
                        debug_json=recommendation.model_dump(mode="python"),
                    )
                )

            session.commit()

        return self.get_review_session()

    def list_reviews(self) -> list[ReviewHistoryItemPayload]:
        with self._session_factory() as session:
            user = self._get_or_create_user(session)
            review_events = self._load_review_events(session, user.id)
            return [
                ReviewHistoryItemPayload(
                    review_id=event.id,
                    coffee_id=event.coffee_id,
                    review_text=event.review_text,
                    overall=float(event.overall_score or 0),
                    created_at=event.created_at,
                )
                for event in review_events
            ]

    def list_recommendation_runs(self) -> list[RecommendationRunPayload]:
        with self._session_factory() as session:
            user = self._get_or_create_user(session)
            runs = list(
                session.scalars(
                    select(RecommendationRunModel)
                    .where(RecommendationRunModel.user_id == user.id)
                    .order_by(RecommendationRunModel.id.desc())
                ).all()
            )
            return [self._to_recommendation_run_payload(session, run) for run in runs]

    def get_recommendation_run(self, run_id: int) -> RecommendationRunPayload:
        with self._session_factory() as session:
            user = self._get_or_create_user(session)
            run = session.get(RecommendationRunModel, run_id)
            if run is None or run.user_id != user.id:
                raise KeyError(f"Recommendation run not found: {run_id}")
            return self._to_recommendation_run_payload(session, run)

    def _get_or_create_user(self, session: Session) -> UserModel:
        email = self._user_email_provider()
        user = session.scalar(select(UserModel).where(UserModel.email == email))
        if user is None:
            user = UserModel(
                email=email,
                display_name="Local User",
            )
            session.add(user)
            session.flush()
        return user

    def _load_review_events(self, session: Session, user_id: int) -> list[ReviewEventModel]:
        return list(
            session.scalars(
                select(ReviewEventModel)
                .where(ReviewEventModel.user_id == user_id)
                .order_by(ReviewEventModel.id.asc())
            ).all()
        )

    def _load_last_recommendations(self, session: Session, user_id: int) -> list[RecommendationPayload]:
        recommendation_run = session.scalar(
            select(RecommendationRunModel)
            .where(RecommendationRunModel.user_id == user_id)
            .order_by(RecommendationRunModel.id.desc())
        )
        if recommendation_run is None:
            return []

        items = list(
            session.scalars(
                select(RecommendationItemModel)
                .where(RecommendationItemModel.recommendation_run_id == recommendation_run.id)
                .order_by(RecommendationItemModel.rank.asc())
            ).all()
        )
        recommendations: list[RecommendationPayload] = []
        for item in items:
            if isinstance(item.debug_json, dict):
                recommendations.append(RecommendationPayload.model_validate(item.debug_json))
        return recommendations

    def _load_recommendations_for_run(
        self,
        session: Session,
        run_id: int,
    ) -> list[RecommendationPayload]:
        items = list(
            session.scalars(
                select(RecommendationItemModel)
                .where(RecommendationItemModel.recommendation_run_id == run_id)
                .order_by(RecommendationItemModel.rank.asc())
            ).all()
        )
        recommendations: list[RecommendationPayload] = []
        for item in items:
            if isinstance(item.debug_json, dict):
                recommendations.append(RecommendationPayload.model_validate(item.debug_json))
        return recommendations

    def _to_recommendation_run_payload(
        self,
        session: Session,
        model: RecommendationRunModel,
    ) -> RecommendationRunPayload:
        return RecommendationRunPayload(
            run_id=model.id,
            seed_review_event_id=model.seed_review_event_id,
            algorithm_version=model.algorithm_version,
            created_at=model.created_at,
            recommendations=self._load_recommendations_for_run(session, model.id),
        )

    def _to_review_event_payload(self, model: ReviewEventModel) -> ReviewEventPayload:
        payload = self._extract_event_payload(model)
        if payload is not None:
            return payload
        return ReviewEventPayload(
            coffee_id=model.coffee_id,
            overall=float(model.overall_score or 0),
        )

    def _extract_event_payload(self, model: ReviewEventModel) -> ReviewEventPayload | None:
        if not isinstance(model.parsed_review_json, dict):
            return None
        payload = model.parsed_review_json.get("event")
        if not isinstance(payload, dict):
            return None
        return ReviewEventPayload.model_validate(payload)

    def _get_review_payload(self, session: Session, review_event_id: int) -> dict[str, object] | None:
        model = session.get(ReviewEventModel, review_event_id)
        if model is None:
            return None
        if not isinstance(model.parsed_review_json, dict):
            return None
        return model.parsed_review_json
