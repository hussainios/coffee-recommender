"""initial schema

Revision ID: 20260719_000001
Revises:
Create Date: 2026-07-19 00:00:01
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "20260719_000001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "catalogue_coffees",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("roaster", sa.String(length=255), nullable=True),
        sa.Column("origin_country", sa.String(length=255), nullable=True),
        sa.Column("region", sa.String(length=255), nullable=True),
        sa.Column("producer", sa.String(length=255), nullable=True),
        sa.Column("farm", sa.String(length=255), nullable=True),
        sa.Column("process_primary", sa.String(length=64), nullable=True),
        sa.Column("process_detail", sa.Text(), nullable=True),
        sa.Column("variety_text", sa.Text(), nullable=True),
        sa.Column("roast_level", sa.String(length=64), nullable=True),
        sa.Column("tasting_notes_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("price_minor", sa.Integer(), nullable=True),
        sa.Column("currency", sa.String(length=8), nullable=False),
        sa.Column("weight_g", sa.Integer(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "coffee_sources",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("coffee_id", sa.String(length=64), nullable=False),
        sa.Column("source_type", sa.String(length=64), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("raw_html_path", sa.Text(), nullable=True),
        sa.Column("scraped_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("parser_version", sa.String(length=64), nullable=True),
        sa.Column("extraction_model", sa.String(length=128), nullable=True),
        sa.Column("extraction_status", sa.String(length=64), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["coffee_id"], ["catalogue_coffees.id"]),
    )
    op.create_index("ix_coffee_sources_coffee_id", "coffee_sources", ["coffee_id"], unique=False)

    op.create_table(
        "sensory_profiles",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("coffee_id", sa.String(length=64), nullable=False),
        sa.Column("profile_version", sa.String(length=64), nullable=False),
        sa.Column("acidity", sa.Numeric(precision=3, scale=2), nullable=False),
        sa.Column("sweetness", sa.Numeric(precision=3, scale=2), nullable=False),
        sa.Column("body", sa.Numeric(precision=3, scale=2), nullable=False),
        sa.Column("bitterness", sa.Numeric(precision=3, scale=2), nullable=False),
        sa.Column("fruitiness", sa.Numeric(precision=3, scale=2), nullable=False),
        sa.Column("chocolate_nutty", sa.Numeric(precision=3, scale=2), nullable=False),
        sa.Column("floral", sa.Numeric(precision=3, scale=2), nullable=False),
        sa.Column("funky_fermented", sa.Numeric(precision=3, scale=2), nullable=False),
        sa.Column("roasty", sa.Numeric(precision=3, scale=2), nullable=False),
        sa.Column("clean_cup", sa.Numeric(precision=3, scale=2), nullable=False),
        sa.Column("confidence", sa.Numeric(precision=3, scale=2), nullable=False),
        sa.Column("evidence_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["coffee_id"], ["catalogue_coffees.id"]),
    )
    op.create_index("ix_sensory_profiles_coffee_id", "sensory_profiles", ["coffee_id"], unique=False)

    op.create_table(
        "embeddings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("coffee_id", sa.String(length=64), nullable=False),
        sa.Column("embedding_model", sa.String(length=128), nullable=False),
        sa.Column("vector_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["coffee_id"], ["catalogue_coffees.id"]),
    )
    op.create_index("ix_embeddings_coffee_id", "embeddings", ["coffee_id"], unique=False)

    op.create_table(
        "review_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("coffee_id", sa.String(length=64), nullable=False),
        sa.Column("review_text", sa.Text(), nullable=False),
        sa.Column("overall_score", sa.Numeric(precision=4, scale=2), nullable=True),
        sa.Column("parsed_review_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["coffee_id"], ["catalogue_coffees.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
    )
    op.create_index("ix_review_events_coffee_id", "review_events", ["coffee_id"], unique=False)
    op.create_index("ix_review_events_user_id", "review_events", ["user_id"], unique=False)

    op.create_table(
        "recommendation_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("seed_review_event_id", sa.Integer(), nullable=True),
        sa.Column("algorithm_version", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["seed_review_event_id"], ["review_events.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
    )
    op.create_index("ix_recommendation_runs_seed_review_event_id", "recommendation_runs", ["seed_review_event_id"], unique=False)
    op.create_index("ix_recommendation_runs_user_id", "recommendation_runs", ["user_id"], unique=False)

    op.create_table(
        "recommendation_items",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("recommendation_run_id", sa.Integer(), nullable=False),
        sa.Column("coffee_id", sa.String(length=64), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("score", sa.Numeric(precision=6, scale=4), nullable=False),
        sa.Column("debug_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(["coffee_id"], ["catalogue_coffees.id"]),
        sa.ForeignKeyConstraint(["recommendation_run_id"], ["recommendation_runs.id"]),
    )
    op.create_index(
        "ix_recommendation_items_coffee_id",
        "recommendation_items",
        ["coffee_id"],
        unique=False,
    )
    op.create_index(
        "ix_recommendation_items_recommendation_run_id",
        "recommendation_items",
        ["recommendation_run_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_recommendation_items_recommendation_run_id", table_name="recommendation_items")
    op.drop_index("ix_recommendation_items_coffee_id", table_name="recommendation_items")
    op.drop_table("recommendation_items")

    op.drop_index("ix_recommendation_runs_user_id", table_name="recommendation_runs")
    op.drop_index("ix_recommendation_runs_seed_review_event_id", table_name="recommendation_runs")
    op.drop_table("recommendation_runs")

    op.drop_index("ix_review_events_user_id", table_name="review_events")
    op.drop_index("ix_review_events_coffee_id", table_name="review_events")
    op.drop_table("review_events")

    op.drop_index("ix_embeddings_coffee_id", table_name="embeddings")
    op.drop_table("embeddings")

    op.drop_index("ix_sensory_profiles_coffee_id", table_name="sensory_profiles")
    op.drop_table("sensory_profiles")

    op.drop_index("ix_coffee_sources_coffee_id", table_name="coffee_sources")
    op.drop_table("coffee_sources")

    op.drop_table("catalogue_coffees")

    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
