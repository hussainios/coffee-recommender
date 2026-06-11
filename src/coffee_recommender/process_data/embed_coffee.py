from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .. import openai_client
from ..schemas import CoffeeRecord


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_embedding_text(row: pd.Series) -> str:
    parts = [
        f"Name: {_clean(row.get('name'))}",
        f"Roaster: {_clean(row.get('roaster'))}",
        f"Origin: {_clean(row.get('origin_country'))} {_clean(row.get('region'))}",
        f"Producer: {_clean(row.get('producer'))}",
        f"Process: {_clean(row.get('process'))}",
        f"Roast level: {_clean(row.get('roast_level'))}",
        f"Variety: {_clean(row.get('variety'))}",
        f"Tasting notes: {_clean(row.get('tasting_notes'))}",
        f"Description: {_clean(row.get('description'))}",
    ]
    return "\n".join(part for part in parts if part.split(": ", 1)[-1])


def build_embedding_text_from_record(coffee: CoffeeRecord) -> str:
    return build_embedding_text(pd.Series(coffee.model_dump(mode="json")))


def embed_texts(texts: list[str], model: str = openai_client.DEFAULT_EMBEDDING_MODEL) -> list[list[float]]:
    response = openai_client.get_openai_client("coffee embeddings").embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def embed_coffee_record(coffee: CoffeeRecord, model: str = openai_client.DEFAULT_EMBEDDING_MODEL) -> list[float]:
    return embed_texts([build_embedding_text_from_record(coffee)], model=model)[0]


def build_embedding_records(
    coffees: pd.DataFrame,
    model: str = openai_client.DEFAULT_EMBEDDING_MODEL,
) -> list[dict[str, str]]:
    texts = [build_embedding_text(row) for _, row in coffees.iterrows()]
    embeddings = embed_texts(texts, model=model)
    return [
        {
            "coffee_id": str(row.get("coffee_id")),
            "embedding_model": model,
            "embedding_text": text,
            "embedding": json.dumps(embedding),
        }
        for ((_, row), text, embedding) in zip(coffees.iterrows(), texts, embeddings)
    ]
