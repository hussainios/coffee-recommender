from __future__ import annotations

from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl, computed_field, field_validator


class Process(StrEnum):
    WASHED = "washed"
    NATURAL = "natural"
    HONEY = "honey"
    ANAEROBIC = "anaerobic"
    ANAEROBIC_NATURAL = "anaerobic_natural"
    CARBONIC_MACERATION = "carbonic_maceration"
    COFERMENTED = "cofermented"
    UNKNOWN = "unknown"


class RoastLevel(StrEnum):
    LIGHT = "light"
    LIGHT_MEDIUM = "light_medium"
    MEDIUM = "medium"
    MEDIUM_DARK = "medium_dark"
    DARK = "dark"
    UNKNOWN = "unknown"


class BrewMethod(StrEnum):
    ESPRESSO = "espresso"
    FILTER = "filter"
    V60 = "v60"
    AEROPRESS = "aeropress"
    CHEMEX = "chemex"
    FRENCH_PRESS = "french_press"
    MOKA_POT = "moka_pot"
    BATCH_BREW = "batch_brew"


class CoffeeRecord(BaseModel):
    """
    Canonical factual product record.

    This should contain product facts from the source page.
    Avoid putting inferred taste scores here.
    """

    coffee_id: str

    name: str
    roaster: Optional[str] = None

    origin_country: Optional[str] = None
    region: Optional[str] = None
    producer: Optional[str] = None
    farm: Optional[str] = None

    process: Process = Process.UNKNOWN
    variety: list[str] = Field(default_factory=list)
    roast_level: RoastLevel = RoastLevel.UNKNOWN

    tasting_notes: list[str] = Field(default_factory=list)
    description: Optional[str] = None

    price: Optional[float] = Field(default=None, ge=0)
    currency: str = "GBP"
    weight_g: Optional[int] = Field(default=None, gt=0)

    brew_methods: list[BrewMethod] = Field(default_factory=list)

    source_url: Optional[HttpUrl] = None
    source_file: str

    @computed_field
    @property
    def price_per_kg(self) -> Optional[float]:
        if self.price is None or self.weight_g is None:
            return None
        return round(self.price / self.weight_g * 1000, 2)

    @field_validator("tasting_notes", "variety", mode="before")
    @classmethod
    def normalise_string_list(cls, value: object) -> list[str]:
        if value is None:
            return []

        if isinstance(value, str):
            value = value.split(",")

        if not isinstance(value, list):
            raise TypeError("Expected a string, list of strings, or None")

        return [
            str(item).strip().lower()
            for item in value
            if str(item).strip()
        ]

    @field_validator("origin_country", "region", "producer", "farm", "roaster")
    @classmethod
    def empty_strings_to_none(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None

        value = value.strip()
        return value or None


class SensoryVector(BaseModel):
    """
    Model-friendly inferred sensory representation.

    This is derived from tasting notes, product description, process and roast level.
    It is not raw source metadata.
    """

    coffee_id: str

    acidity: float = Field(ge=0, le=1)
    sweetness: float = Field(ge=0, le=1)
    body: float = Field(ge=0, le=1)
    bitterness: float = Field(ge=0, le=1)

    fruitiness: float = Field(ge=0, le=1)
    chocolate_nutty: float = Field(ge=0, le=1)
    floral: float = Field(ge=0, le=1)
    funky_fermented: float = Field(ge=0, le=1)
    roasty: float = Field(ge=0, le=1)
    clean_cup: float = Field(ge=0, le=1)

    confidence: float = Field(ge=0, le=1)
    evidence: dict[str, list[str]] = Field(default_factory=dict)


class SensoryExtractionResult(BaseModel):
    """
    Full LLM extraction output.

    Keep this separate from the final sensory vector so extraction quality
    can be debugged and audited later.
    """

    coffee_id: str
    sensory: SensoryVector
    extraction_model: str
    extraction_prompt_version: str
    warnings: list[str] = Field(default_factory=list)