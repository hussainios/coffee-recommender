from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


JsonDict = dict[str, Any]
StringList = list[str]
FloatList = list[float]


class UserModel(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    display_name: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    review_events: Mapped[list[ReviewEventModel]] = relationship(back_populates="user")
    recommendation_runs: Mapped[list[RecommendationRunModel]] = relationship(back_populates="user")


class CatalogueCoffeeModel(Base):
    __tablename__ = "catalogue_coffees"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    roaster: Mapped[str | None] = mapped_column(String(255))
    origin_country: Mapped[str | None] = mapped_column(String(255))
    region: Mapped[str | None] = mapped_column(String(255))
    producer: Mapped[str | None] = mapped_column(String(255))
    farm: Mapped[str | None] = mapped_column(String(255))
    process_primary: Mapped[str | None] = mapped_column(String(64))
    process_detail: Mapped[str | None] = mapped_column(Text)
    variety_text: Mapped[str | None] = mapped_column(Text)
    roast_level: Mapped[str | None] = mapped_column(String(64))
    tasting_notes_json: Mapped[StringList] = mapped_column(JSONB, default=list, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    price_minor: Mapped[int | None] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(8), default="GBP", nullable=False)
    weight_g: Mapped[int | None] = mapped_column(Integer)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    coffee_sources: Mapped[list[CoffeeSourceModel]] = relationship(back_populates="coffee")
    sensory_profiles: Mapped[list[SensoryProfileModel]] = relationship(back_populates="coffee")
    embeddings: Mapped[list[EmbeddingModel]] = relationship(back_populates="coffee")
    review_events: Mapped[list[ReviewEventModel]] = relationship(back_populates="coffee")
    recommendation_items: Mapped[list[RecommendationItemModel]] = relationship(back_populates="coffee")


class CoffeeSourceModel(Base):
    __tablename__ = "coffee_sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    coffee_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("catalogue_coffees.id"),
        nullable=False,
        index=True,
    )
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    raw_html_path: Mapped[str | None] = mapped_column(Text)
    scraped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    parser_version: Mapped[str | None] = mapped_column(String(64))
    extraction_model: Mapped[str | None] = mapped_column(String(128))
    extraction_status: Mapped[str | None] = mapped_column(String(64))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    coffee: Mapped[CatalogueCoffeeModel] = relationship(back_populates="coffee_sources")


class SensoryProfileModel(Base):
    __tablename__ = "sensory_profiles"

    id: Mapped[int] = mapped_column(primary_key=True)
    coffee_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("catalogue_coffees.id"),
        nullable=False,
        index=True,
    )
    profile_version: Mapped[str] = mapped_column(String(64), nullable=False)
    acidity: Mapped[Decimal] = mapped_column(Numeric(3, 2), nullable=False)
    sweetness: Mapped[Decimal] = mapped_column(Numeric(3, 2), nullable=False)
    body: Mapped[Decimal] = mapped_column(Numeric(3, 2), nullable=False)
    bitterness: Mapped[Decimal] = mapped_column(Numeric(3, 2), nullable=False)
    fruitiness: Mapped[Decimal] = mapped_column(Numeric(3, 2), nullable=False)
    chocolate_nutty: Mapped[Decimal] = mapped_column(Numeric(3, 2), nullable=False)
    floral: Mapped[Decimal] = mapped_column(Numeric(3, 2), nullable=False)
    funky_fermented: Mapped[Decimal] = mapped_column(Numeric(3, 2), nullable=False)
    roasty: Mapped[Decimal] = mapped_column(Numeric(3, 2), nullable=False)
    clean_cup: Mapped[Decimal] = mapped_column(Numeric(3, 2), nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(3, 2), nullable=False)
    evidence_json: Mapped[JsonDict] = mapped_column(JSONB, default=dict, nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    coffee: Mapped[CatalogueCoffeeModel] = relationship(back_populates="sensory_profiles")


class EmbeddingModel(Base):
    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(primary_key=True)
    coffee_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("catalogue_coffees.id"),
        nullable=False,
        index=True,
    )
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False)
    vector_json: Mapped[FloatList] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    coffee: Mapped[CatalogueCoffeeModel] = relationship(back_populates="embeddings")


class ReviewEventModel(Base):
    __tablename__ = "review_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    coffee_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("catalogue_coffees.id"),
        nullable=False,
        index=True,
    )
    review_text: Mapped[str] = mapped_column(Text, nullable=False)
    overall_score: Mapped[Decimal | None] = mapped_column(Numeric(4, 2))
    parsed_review_json: Mapped[JsonDict] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    user: Mapped[UserModel] = relationship(back_populates="review_events")
    coffee: Mapped[CatalogueCoffeeModel] = relationship(back_populates="review_events")
    recommendation_runs: Mapped[list[RecommendationRunModel]] = relationship(
        back_populates="seed_review_event"
    )


class RecommendationRunModel(Base):
    __tablename__ = "recommendation_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    seed_review_event_id: Mapped[int | None] = mapped_column(
        ForeignKey("review_events.id"),
        index=True,
    )
    algorithm_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    user: Mapped[UserModel] = relationship(back_populates="recommendation_runs")
    seed_review_event: Mapped[ReviewEventModel | None] = relationship(
        back_populates="recommendation_runs"
    )
    items: Mapped[list[RecommendationItemModel]] = relationship(back_populates="recommendation_run")


class RecommendationItemModel(Base):
    __tablename__ = "recommendation_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    recommendation_run_id: Mapped[int] = mapped_column(
        ForeignKey("recommendation_runs.id"),
        nullable=False,
        index=True,
    )
    coffee_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("catalogue_coffees.id"),
        nullable=False,
        index=True,
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[Decimal] = mapped_column(Numeric(6, 4), nullable=False)
    debug_json: Mapped[JsonDict] = mapped_column(JSONB, default=dict, nullable=False)

    recommendation_run: Mapped[RecommendationRunModel] = relationship(back_populates="items")
    coffee: Mapped[CatalogueCoffeeModel] = relationship(back_populates="recommendation_items")
