from __future__ import annotations

import json
from typing import Any

from .. import openai_client
from ..coffee_dimensions import SENSORY_DIMENSIONS
from ..schemas import CoffeeRecord, SensoryVector

SENSORY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "acidity": {"type": "number", "minimum": 0, "maximum": 1},
        "sweetness": {"type": "number", "minimum": 0, "maximum": 1},
        "body": {"type": "number", "minimum": 0, "maximum": 1},
        "bitterness": {"type": "number", "minimum": 0, "maximum": 1},
        "fruitiness": {"type": "number", "minimum": 0, "maximum": 1},
        "chocolate_nutty": {"type": "number", "minimum": 0, "maximum": 1},
        "floral": {"type": "number", "minimum": 0, "maximum": 1},
        "funky_fermented": {"type": "number", "minimum": 0, "maximum": 1},
        "roasty": {"type": "number", "minimum": 0, "maximum": 1},
        "clean_cup": {"type": "number", "minimum": 0, "maximum": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                dimension: {
                    "type": "array",
                    "items": {"type": "string"},
                }
                for dimension in SENSORY_DIMENSIONS
            },
            "required": list(SENSORY_DIMENSIONS),
        },
    },
    "required": [
        "acidity",
        "sweetness",
        "body",
        "bitterness",
        "fruitiness",
        "chocolate_nutty",
        "floral",
        "funky_fermented",
        "roasty",
        "clean_cup",
        "confidence",
        "evidence",
    ],
}


def build_extraction_input(coffee: CoffeeRecord) -> str:
    return f"""
Coffee product data:

Name: {coffee.name}
Origin country: {coffee.origin_country}
Region: {coffee.region}
Producer: {coffee.producer}
Process: {coffee.process}
Variety: {coffee.variety}
Roast level: {coffee.roast_level}
Tasting notes: {", ".join(coffee.tasting_notes)}
Description:
{coffee.description}
""".strip()


def extract_sensory_vector_llm(
    coffee: CoffeeRecord,
    model: str = openai_client.DEFAULT_CHAT_MODEL,
    temperature: float = 0.0,
) -> SensoryVector:
    prompt = f"""
You are extracting sensory attributes from specialty coffee product text.

Return only structured JSON matching the schema.

Rules:
- Score each sensory attribute from 0.0 to 1.0.
- Use only evidence from the provided text.
- Do not invent factual metadata.
- If evidence is weak, use a neutral score around 0.5 and lower confidence.
- "funky_fermented" should be high for anaerobic, boozy, winey, overripe, lactic, or heavily fermented descriptions.
- "clean_cup" should be high for washed, clean, balanced, transparent, delicate coffees.
- "roasty" should be high for dark roast, smoky, burnt, bitter, or roast-forward notes.
- Include short evidence phrases for attributes where possible.

{build_extraction_input(coffee)}
""".strip()

    response = openai_client.get_openai_client("sensory extraction").responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
        text={
            "format": {
                "type": "json_schema",
                "name": "coffee_sensory_vector",
                "schema": SENSORY_JSON_SCHEMA,
                "strict": True,
            }
        },
    )

    raw_json = response.output_text
    parsed = json.loads(raw_json)

    return SensoryVector(
        coffee_id=coffee.coffee_id,
        **parsed,
    )
