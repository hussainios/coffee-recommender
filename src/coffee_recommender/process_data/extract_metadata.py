from __future__ import annotations

import json
from typing import Any

from pydantic import HttpUrl

from .. import openai_client
from ..schemas import BrewMethod, CoffeeRecord, Process, RoastLevel

NULLABLE_STRING_SCHEMA: dict[str, Any] = {
    "anyOf": [
        {"type": "string"},
        {"type": "null"},
    ]
}

METADATA_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "roaster": NULLABLE_STRING_SCHEMA,
        "origin_country": NULLABLE_STRING_SCHEMA,
        "region": NULLABLE_STRING_SCHEMA,
        "producer": NULLABLE_STRING_SCHEMA,
        "farm": NULLABLE_STRING_SCHEMA,
        "process": {
            "type": "string",
            "enum": [process.value for process in Process],
        },
        "variety": {
            "type": "array",
            "items": {"type": "string"},
        },
        "roast_level": {
            "type": "string",
            "enum": [roast_level.value for roast_level in RoastLevel],
        },
        "tasting_notes": {
            "type": "array",
            "items": {"type": "string"},
        },
        "description": NULLABLE_STRING_SCHEMA,
        "price": {
            "anyOf": [
                {"type": "number", "minimum": 0},
                {"type": "null"},
            ]
        },
        "currency": {"type": "string"},
        "weight_g": {
            "anyOf": [
                {"type": "integer", "minimum": 1},
                {"type": "null"},
            ]
        },
        "brew_methods": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [brew_method.value for brew_method in BrewMethod],
            },
        },
    },
    "required": [
        "name",
        "roaster",
        "origin_country",
        "region",
        "producer",
        "farm",
        "process",
        "variety",
        "roast_level",
        "tasting_notes",
        "description",
        "price",
        "currency",
        "weight_g",
        "brew_methods",
    ],
}


def build_metadata_extraction_input(text: str, source_url: str | None = None) -> str:
    source_line = f"Source URL: {source_url}\n" if source_url else ""
    return (
        "Coffee product page text:\n\n"
        f"{source_line}"
        f"{text.strip()}"
    )


def extract_metadata_llm(
    *,
    text: str,
    source: str,
    source_url: str | None = None,
    model: str = openai_client.DEFAULT_CHAT_MODEL,
    temperature: float = 0.0,
) -> CoffeeRecord:
    prompt = f"""
You extract factual specialty coffee product metadata from messy storefront page text.

Return only structured JSON matching the schema.

Rules:
- Extract product facts only. Do not infer unsupported facts.
- Prefer the actual coffee/product title, not the site title or navigation text.
- Remove retailer suffixes from the product name when possible, for example text like "| Roaster Name".
- Map farmer, producer, cooperative, co-op, grower, or producer group fields into "producer" when they identify who produced the coffee.
- Only fill "farm" when the text clearly names a farm, estate, finca, hacienda, or mill-like producing place.
- Use "process" = "unknown" when the process is not explicit.
- Use "roast_level" = "unknown" when the roast level is not explicit.
- Use empty arrays for missing lists.
- Use null for missing nullable scalar fields.
- Keep tasting notes concise and literal.
- For description, keep only the most product-relevant paragraph(s), not the full page boilerplate.
- If a price uses the pound symbol, set currency to "GBP".
- Ignore shipping, newsletter, login, footer, and navigation content.
- Ignore subscription, gift card, accessory, and workshop framing unless it is clearly the current product.

{build_metadata_extraction_input(text, source_url=source_url)}
""".strip()

    response = openai_client.get_openai_client("metadata extraction").responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
        text={
            "format": {
                "type": "json_schema",
                "name": "coffee_metadata",
                "schema": METADATA_JSON_SCHEMA,
                "strict": True,
            }
        },
    )

    parsed = json.loads(response.output_text)
    from .parse_metadata import build_coffee_id

    coffee_id = build_coffee_id(parsed["name"], source, source_url=source_url)
    validated_source_url = HttpUrl(source_url) if source_url else None

    return CoffeeRecord(
        coffee_id=coffee_id,
        source_url=validated_source_url,
        source_file=source,
        **parsed,
    )
