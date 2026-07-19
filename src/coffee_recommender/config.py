from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class DataPaths:
    coffees_path: Path
    sensory_path: Path
    embeddings_path: Path


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str | None = Field(
        default=None,
        alias="COFFEE_RECOMMENDER_DATABASE_URL",
    )
    api_base_url: str = Field(
        default="http://127.0.0.1:8000",
        alias="COFFEE_RECOMMENDER_API_BASE_URL",
    )
    cors_origins: str | None = Field(
        default=None,
        alias="COFFEE_RECOMMENDER_CORS_ORIGINS",
    )


def get_app_settings() -> AppSettings:
    return AppSettings()


def get_data_paths() -> DataPaths:
    return DataPaths(
        coffees_path=Path(os.environ.get("COFFEE_RECOMMENDER_COFFEES_PATH", "data/processed/coffees.csv")),
        sensory_path=Path(
            os.environ.get(
                "COFFEE_RECOMMENDER_SENSORY_PATH",
                "data/processed/coffee_sensory_vectors.csv",
            )
        ),
        embeddings_path=Path(
            os.environ.get(
                "COFFEE_RECOMMENDER_EMBEDDINGS_PATH",
                "data/processed/coffee_embeddings.csv",
            )
        ),
    )


def get_api_base_url() -> str:
    return get_app_settings().api_base_url.rstrip("/")


def get_cors_origins() -> list[str]:
    raw_origins = get_app_settings().cors_origins
    if raw_origins:
        return [origin.strip().rstrip("/") for origin in raw_origins.split(",") if origin.strip()]
    return [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ]


def get_database_url() -> str:
    settings = get_app_settings()
    if settings.database_url is None:
        raise RuntimeError(
            "COFFEE_RECOMMENDER_DATABASE_URL is required for database-backed workflows."
        )
    return settings.database_url
