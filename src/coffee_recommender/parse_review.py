from __future__ import annotations

import json
from typing import Any

from . import openai_client
from .coffee_dimensions import SENSORY_DIMENSIONS
from .landscape import CoffeeFeatures, ReviewEvent


REVIEW_EVENT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "overall": {"type": "number", "minimum": -1, "maximum": 1},
        "change_requests": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "attribute": {"type": "string", "enum": list(SENSORY_DIMENSIONS)},
                    "direction": {"type": "string", "enum": ["higher", "lower"]},
                    "strength": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["attribute", "direction", "strength"],
            },
        },
        "attribute_opinions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "attribute": {"type": "string", "enum": list(SENSORY_DIMENSIONS)},
                    "sentiment": {"type": "string", "enum": ["liked", "disliked"]},
                    "strength": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["attribute", "sentiment", "strength"],
            },
        },
    },
    "required": ["overall", "change_requests", "attribute_opinions"],
}


def _clip(value: float, minimum: float = -1.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _coffee_context(reviewed_coffee: CoffeeFeatures) -> str:
    sensory = json.dumps(reviewed_coffee.sensory, sort_keys=True)
    process = json.dumps(reviewed_coffee.process, sort_keys=True)
    return f"""
Reviewed coffee:
- coffee_id: {reviewed_coffee.coffee_id}
- name: {reviewed_coffee.name}
- sensory: {sensory}
- process: {process}
""".strip()


def _to_review_event(parsed: dict[str, Any], reviewed_coffee: CoffeeFeatures) -> ReviewEvent:
    change_requests: dict[str, dict[str, float | str]] = {}
    attribute_opinions: dict[str, dict[str, float | str]] = {}

    for change_request in parsed.get("change_requests", []):
        attribute = str(change_request.get("attribute", ""))
        direction = change_request.get("direction")
        strength = _clip(float(change_request.get("strength", 0.0)), 0.0, 1.0)

        if attribute not in SENSORY_DIMENSIONS or direction not in ("higher", "lower") or strength == 0.0:
            continue

        change_requests[attribute] = {
            "direction": direction,
            "strength": strength,
        }

    for opinion in parsed.get("attribute_opinions", []):
        attribute = str(opinion.get("attribute", ""))
        sentiment = opinion.get("sentiment")
        strength = _clip(float(opinion.get("strength", 0.0)), 0.0, 1.0)

        if attribute not in SENSORY_DIMENSIONS or sentiment not in ("liked", "disliked") or strength == 0.0:
            continue

        attribute_opinions[attribute] = {
            "sentiment": sentiment,
            "strength": strength,
        }

    return {
        "coffee_id": reviewed_coffee.coffee_id,
        "overall": _clip(float(parsed.get("overall", 0.0))),
        "change_requests": change_requests,
        "attribute_opinions": attribute_opinions,
    }


def parse_review_event(
    review_text: str,
    reviewed_coffee: CoffeeFeatures,
    model: str = openai_client.DEFAULT_CHAT_MODEL,
    temperature: float = 0.0,
) -> ReviewEvent:
    prompt = f"""
You are parsing a user's coffee review into a structured event for a local recommendation landscape.

Return only JSON matching the schema.

Rules:
- "overall" is how much the user liked this reviewed coffee as a whole.
- Use -1.0 for strong dislike, 0.0 for mixed/neutral, and 1.0 for strong like.
- "change_requests" are explicit requests for more or less of an attribute.
- "attribute_opinions" are things the user liked or disliked at this coffee's current level.
- Do not decide whether an attribute opinion means higher or lower; only record liked/disliked.
- "I liked the berry notes" should create fruitiness liked in attribute_opinions.
- "I wanted more sweetness" should create sweetness higher in change_requests.
- "Too acidic" should create acidity lower in change_requests.
- "Too funky and bitter" should create funky_fermented lower and bitterness lower in change_requests.
- "I liked the roast level" should create roasty liked in attribute_opinions.
- "I disliked the body" should create body disliked in attribute_opinions.
- If a sentence contains both, extract both. "The sweetness was nice, but I wanted it sweeter" should create sweetness liked in attribute_opinions and sweetness higher in change_requests.
- Use lower strengths for mild language like "a little" and higher strengths for strong language like "way too".

Supported attributes:
{", ".join(SENSORY_DIMENSIONS)}

{_coffee_context(reviewed_coffee)}

User review:
{review_text}
""".strip()

    response = openai_client.get_openai_client("review parsing").responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
        text={
            "format": {
                "type": "json_schema",
                "name": "coffee_review_event",
                "schema": REVIEW_EVENT_JSON_SCHEMA,
                "strict": True,
            }
        },
    )

    parsed = json.loads(response.output_text)
    return _to_review_event(parsed, reviewed_coffee)
