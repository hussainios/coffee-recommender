"""add source capture and parse runs

Revision ID: 20260719_000003
Revises: 20260719_000002
Create Date: 2026-07-19 00:00:03
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260719_000003"
down_revision = "20260719_000002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "source_stores",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("base_url", sa.Text(), nullable=False),
        sa.Column("allowed_domains_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "source_pages",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("store_id", sa.Integer(), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("normalized_url", sa.Text(), nullable=False),
        sa.Column("page_type", sa.String(length=64), nullable=False),
        sa.Column("fetch_status", sa.String(length=64), nullable=False),
        sa.Column("html_content", sa.Text(), nullable=True),
        sa.Column("visible_text", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=True),
        sa.Column("discovered_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["store_id"], ["source_stores.id"]),
        sa.UniqueConstraint("source_url"),
        sa.UniqueConstraint("normalized_url"),
    )
    op.create_index("ix_source_pages_store_id", "source_pages", ["store_id"], unique=False)
    op.create_index("ix_source_pages_content_hash", "source_pages", ["content_hash"], unique=False)

    op.create_table(
        "source_page_parse_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source_page_id", sa.Integer(), nullable=False),
        sa.Column("coffee_id", sa.String(length=128), nullable=True),
        sa.Column("schema_version", sa.String(length=64), nullable=False),
        sa.Column("parser_version", sa.String(length=64), nullable=False),
        sa.Column("extraction_model", sa.String(length=128), nullable=False),
        sa.Column("parse_status", sa.String(length=64), nullable=False),
        sa.Column("warnings_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["coffee_id"], ["catalogue_coffees.id"]),
        sa.ForeignKeyConstraint(["source_page_id"], ["source_pages.id"]),
    )
    op.create_index(
        "ix_source_page_parse_runs_source_page_id",
        "source_page_parse_runs",
        ["source_page_id"],
        unique=False,
    )
    op.create_index(
        "ix_source_page_parse_runs_coffee_id",
        "source_page_parse_runs",
        ["coffee_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_source_page_parse_runs_coffee_id", table_name="source_page_parse_runs")
    op.drop_index("ix_source_page_parse_runs_source_page_id", table_name="source_page_parse_runs")
    op.drop_table("source_page_parse_runs")

    op.drop_index("ix_source_pages_content_hash", table_name="source_pages")
    op.drop_index("ix_source_pages_store_id", table_name="source_pages")
    op.drop_table("source_pages")

    op.drop_table("source_stores")
