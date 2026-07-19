from __future__ import annotations

import ast
import json
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .coffee_service import CatalogueData, load_catalogue
from .config import DataPaths
from .db.models import CatalogueCoffeeModel, EmbeddingModel, SensoryProfileModel
from .landscape import build_feature_index

DEFAULT_SENSORY_PROFILE_VERSION = "csv_import_v1"


class CatalogueStore(Protocol):
    def load_catalogue(self) -> CatalogueData: ...


@dataclass(frozen=True)
class CsvCatalogueStore:
    data_paths: DataPaths

    def load_catalogue(self) -> CatalogueData:
        return load_catalogue(
            self.data_paths.coffees_path,
            self.data_paths.sensory_path,
            self.data_paths.embeddings_path,
        )


@dataclass(frozen=True)
class SqlAlchemyCatalogueStore:
    session_factory: sessionmaker[Session]

    def load_catalogue(self) -> CatalogueData:
        with self.session_factory() as session:
            coffees = list(
                session.scalars(
                    select(CatalogueCoffeeModel)
                    .where(CatalogueCoffeeModel.is_active.is_(True))
                    .order_by(CatalogueCoffeeModel.name.asc())
                ).all()
            )
            sensory_profiles = list(session.scalars(select(SensoryProfileModel)).all())
            embeddings = list(session.scalars(select(EmbeddingModel)).all())

        coffees_df = pd.DataFrame([_coffee_to_row(coffee) for coffee in coffees])
        sensory_df = pd.DataFrame(_latest_sensory_rows(sensory_profiles))
        embeddings_df = pd.DataFrame(_latest_embedding_rows(embeddings))
        features = build_feature_index(coffees_df, sensory_df, embeddings_df) if not coffees_df.empty else {}

        return CatalogueData(
            coffees=coffees_df,
            features=features,
            data_paths_key=("database", "database", "database"),
        )


def import_catalogue_from_csvs(
    *,
    data_paths: DataPaths,
    session_factory: sessionmaker[Session],
    profile_version: str = DEFAULT_SENSORY_PROFILE_VERSION,
) -> int:
    coffees_df = pd.read_csv(data_paths.coffees_path)
    sensory_df = pd.read_csv(data_paths.sensory_path)
    embeddings_df = pd.read_csv(data_paths.embeddings_path)

    sensory_lookup = {
        str(row["coffee_id"]): row.to_dict()
        for _, row in sensory_df.iterrows()
    }
    embeddings_lookup = {
        str(row["coffee_id"]): row.to_dict()
        for _, row in embeddings_df.iterrows()
    }

    imported_count = 0
    with session_factory() as session:
        for _, row in coffees_df.iterrows():
            coffee_id = str(row["coffee_id"])
            model = session.get(CatalogueCoffeeModel, coffee_id)
            if model is None:
                model = CatalogueCoffeeModel(id=coffee_id, name=str(row.get("name", coffee_id)))
                session.add(model)

            model.name = str(row.get("name") or coffee_id)
            model.roaster = _optional_text(row.get("roaster"))
            model.origin_country = _optional_text(row.get("origin_country"))
            model.region = _optional_text(row.get("region"))
            model.producer = _optional_text(row.get("producer"))
            model.farm = _optional_text(row.get("farm"))
            model.process_primary = _optional_text(row.get("process"))
            model.process_detail = _optional_text(row.get("process"))
            model.variety_text = _optional_text(row.get("variety"))
            model.roast_level = _optional_text(row.get("roast_level"))
            model.tasting_notes_json = _parse_string_list(row.get("tasting_notes"))
            model.description = _optional_text(row.get("description"))
            model.price_minor = _price_minor(row.get("price"))
            model.currency = _optional_text(row.get("currency")) or "GBP"
            model.weight_g = _optional_int(row.get("weight_g"))
            model.is_active = True

            for existing in session.scalars(
                select(SensoryProfileModel).where(SensoryProfileModel.coffee_id == coffee_id)
            ).all():
                session.delete(existing)

            sensory_row = sensory_lookup.get(coffee_id)
            if sensory_row is not None:
                session.add(
                    SensoryProfileModel(
                        coffee_id=coffee_id,
                        profile_version=profile_version,
                        acidity=_decimal_value(sensory_row.get("acidity")),
                        sweetness=_decimal_value(sensory_row.get("sweetness")),
                        body=_decimal_value(sensory_row.get("body")),
                        bitterness=_decimal_value(sensory_row.get("bitterness")),
                        fruitiness=_decimal_value(sensory_row.get("fruitiness")),
                        chocolate_nutty=_decimal_value(sensory_row.get("chocolate_nutty")),
                        floral=_decimal_value(sensory_row.get("floral")),
                        funky_fermented=_decimal_value(sensory_row.get("funky_fermented")),
                        roasty=_decimal_value(sensory_row.get("roasty")),
                        clean_cup=_decimal_value(sensory_row.get("clean_cup")),
                        confidence=_decimal_value(sensory_row.get("confidence", 0.0)),
                        evidence_json=_parse_dict(sensory_row.get("evidence")),
                        model_name="csv_import",
                    )
                )

            for existing in session.scalars(
                select(EmbeddingModel).where(EmbeddingModel.coffee_id == coffee_id)
            ).all():
                session.delete(existing)

            embedding_row = embeddings_lookup.get(coffee_id)
            if embedding_row is not None:
                session.add(
                    EmbeddingModel(
                        coffee_id=coffee_id,
                        embedding_model=str(embedding_row.get("embedding_model") or "unknown"),
                        vector_json=_parse_embedding(embedding_row.get("embedding")),
                    )
                )

            imported_count += 1

        session.commit()

    return imported_count


def _coffee_to_row(coffee: CatalogueCoffeeModel) -> dict[str, object]:
    return {
        "coffee_id": coffee.id,
        "name": coffee.name,
        "roaster": coffee.roaster,
        "origin_country": coffee.origin_country,
        "region": coffee.region,
        "producer": coffee.producer,
        "farm": coffee.farm,
        "process": coffee.process_primary,
        "variety": coffee.variety_text,
        "roast_level": coffee.roast_level,
        "tasting_notes": coffee.tasting_notes_json,
        "description": coffee.description,
        "price": _price_major(coffee.price_minor),
        "currency": coffee.currency,
        "weight_g": coffee.weight_g,
        "source_url": None,
    }


def _latest_sensory_rows(models: Iterable[SensoryProfileModel]) -> list[dict[str, object]]:
    latest: dict[str, SensoryProfileModel] = {}
    for model in models:
        existing = latest.get(model.coffee_id)
        if existing is None or model.id > existing.id:
            latest[model.coffee_id] = model

    rows: list[dict[str, object]] = []
    for model in latest.values():
        rows.append(
            {
                "coffee_id": model.coffee_id,
                "acidity": float(model.acidity),
                "sweetness": float(model.sweetness),
                "body": float(model.body),
                "bitterness": float(model.bitterness),
                "fruitiness": float(model.fruitiness),
                "chocolate_nutty": float(model.chocolate_nutty),
                "floral": float(model.floral),
                "funky_fermented": float(model.funky_fermented),
                "roasty": float(model.roasty),
                "clean_cup": float(model.clean_cup),
                "confidence": float(model.confidence),
                "evidence": model.evidence_json,
            }
        )
    return rows


def _latest_embedding_rows(models: Iterable[EmbeddingModel]) -> list[dict[str, object]]:
    latest: dict[str, EmbeddingModel] = {}
    for model in models:
        existing = latest.get(model.coffee_id)
        if existing is None or model.id > existing.id:
            latest[model.coffee_id] = model

    rows: list[dict[str, object]] = []
    for model in latest.values():
        rows.append(
            {
                "coffee_id": model.coffee_id,
                "embedding": json.dumps(model.vector_json),
            }
        )
    return rows


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(float(value))


def _price_minor(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(round(float(value) * 100))


def _price_major(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / 100, 2)


def _parse_string_list(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = [item.strip() for item in text.split(",")]
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return []


def _parse_dict(value: object) -> dict[str, object]:
    if value is None or pd.isna(value):
        return {}
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_embedding(value: object) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    text = str(value or "").strip()
    if not text:
        raise ValueError("Embedding value is required for catalogue import.")
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("Embedding must be a JSON list.")
    return [float(item) for item in parsed]


def _decimal_value(value: object) -> Decimal:
    if value is None or pd.isna(value):
        return Decimal("0")
    return Decimal(str(value))
