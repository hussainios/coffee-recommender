"""expand coffee id length

Revision ID: 20260719_000002
Revises: 20260719_000001
Create Date: 2026-07-19 00:00:02
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260719_000002"
down_revision = "20260719_000001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("catalogue_coffees", "id", type_=sa.String(length=128))
    op.alter_column("coffee_sources", "coffee_id", type_=sa.String(length=128))
    op.alter_column("sensory_profiles", "coffee_id", type_=sa.String(length=128))
    op.alter_column("embeddings", "coffee_id", type_=sa.String(length=128))
    op.alter_column("review_events", "coffee_id", type_=sa.String(length=128))
    op.alter_column("recommendation_items", "coffee_id", type_=sa.String(length=128))


def downgrade() -> None:
    op.alter_column("recommendation_items", "coffee_id", type_=sa.String(length=64))
    op.alter_column("review_events", "coffee_id", type_=sa.String(length=64))
    op.alter_column("embeddings", "coffee_id", type_=sa.String(length=64))
    op.alter_column("sensory_profiles", "coffee_id", type_=sa.String(length=64))
    op.alter_column("coffee_sources", "coffee_id", type_=sa.String(length=64))
    op.alter_column("catalogue_coffees", "id", type_=sa.String(length=64))
