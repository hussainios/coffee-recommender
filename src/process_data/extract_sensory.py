from __future__ import annotations

import json
import os
import sys
from typing import Any

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
PROCESS_DATA_DIR = Path(__file__).resolve().parent

for path in (SRC_DIR, PROCESS_DATA_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from dotenv import load_dotenv
from openai import OpenAI

from schemas import CoffeeRecord, SensoryVector

load_dotenv()

client = None

SENSORY_DIMENSIONS: tuple[str, ...] = (
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
)


def get_client() -> OpenAI:
    global client
    if client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for sensory extraction. "
                "Add it to .env or export it before running build_dataset.py."
            )
        client = OpenAI(api_key=api_key)
    return client


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
    model: str = "gpt-5.4-nano",
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

    response = get_client().responses.create(
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


def extract_sensory_vector(
    coffee: CoffeeRecord,
    model: str = "gpt-5.4-nano",
    temperature: float = 0.0,
) -> SensoryVector:
    """
    Backwards-compatible wrapper for the LLM-backed sensory extractor.

    Keeping this alias lets the rest of the pipeline stay unchanged while the
    LLM boundary remains explicit in this module.
    """

    return extract_sensory_vector_llm(coffee, model=model, temperature=temperature)
