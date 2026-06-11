from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    coffees_path: Path
    sensory_path: Path
    embeddings_path: Path


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
    return os.environ.get("COFFEE_RECOMMENDER_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
