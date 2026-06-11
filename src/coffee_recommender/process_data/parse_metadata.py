from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit, urlunsplit

from ..schemas import BrewMethod, CoffeeRecord, Process, RoastLevel


PROCESS_PATTERNS: list[tuple[str, Process]] = [
    (r"\banaerobic\s+natural\b", Process.ANAEROBIC_NATURAL),
    (r"\bcarbonic\s+maceration\b", Process.CARBONIC_MACERATION),
    (r"\bcofermented\b", Process.COFERMENTED),
    (r"\banaerobic\b", Process.ANAEROBIC),
    (r"\bwashed\b", Process.WASHED),
    (r"\bnatural\b", Process.NATURAL),
    (r"\bhoney\b", Process.HONEY),
]

ROAST_PATTERNS: list[tuple[str, RoastLevel]] = [
    (r"\blight[-\s]?medium\b", RoastLevel.LIGHT_MEDIUM),
    (r"\bmedium[-\s]?dark\b", RoastLevel.MEDIUM_DARK),
    (r"\blight\b", RoastLevel.LIGHT),
    (r"\bmedium\b", RoastLevel.MEDIUM),
    (r"\bdark\b", RoastLevel.DARK),
]

BREW_METHOD_PATTERNS: list[tuple[str, BrewMethod]] = [
    (r"\bespresso\b", BrewMethod.ESPRESSO),
    (r"\bfilter\b", BrewMethod.FILTER),
    (r"\bv60\b", BrewMethod.V60),
    (r"\baeropress\b", BrewMethod.AEROPRESS),
    (r"\bchemex\b", BrewMethod.CHEMEX),
    (r"\bfrench\s+press\b", BrewMethod.FRENCH_PRESS),
    (r"\bcafeti[eè]re\b", BrewMethod.FRENCH_PRESS),
    (r"\bmoka\s+pot\b", BrewMethod.MOKA_POT),
    (r"\bbatch\s+brew\b", BrewMethod.BATCH_BREW),
]


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def normalise_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None

    value = normalise_whitespace(value)
    return value or None


def first_match(patterns: list[str], text: str, flags: int = re.I) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags)
        if match:
            return empty_to_none(match.group(1))
    return None


def parse_price(text: str) -> float | None:
    patterns = [
        r"£\s?(\d+(?:\.\d{1,2})?)",
        r"GBP\s?(\d+(?:\.\d{1,2})?)",
    ]

    value = first_match(patterns, text)
    if value is None:
        return None

    return float(value)


def parse_weight_g(text: str) -> int | None:
    gram_match = re.search(r"\b(\d{2,5})\s?g\b", text, re.I)
    if gram_match:
        return int(gram_match.group(1))

    kg_match = re.search(r"\b(\d+(?:\.\d+)?)\s?kg\b", text, re.I)
    if kg_match:
        return int(float(kg_match.group(1)) * 1000)

    return None


def parse_labelled_field(label: str, text: str) -> str | None:
    """
    Extract values from lines like:
    Origin: Colombia
    Process - Washed
    Producer | Juan Perez

    Stops at the end of the line.
    """
    pattern = rf"(?im)^\s*{re.escape(label)}\s*[:\-\|]\s*(.+?)\s*$"
    match = re.search(pattern, text)
    if not match:
        return None

    return empty_to_none(match.group(1))


def parse_name(text: str, file_path: Path) -> str:
    lines = [normalise_whitespace(line) for line in text.splitlines() if line.strip()]

    ignored_prefixes = (
        "£",
        "from £",
        "add to cart",
        "buy now",
        "by ",
        "subscribe",
        "description",
        "tasting notes",
        "notes:",
        "origin:",
        "process:",
        "producer:",
        "variety:",
        "roast:",
    )

    for line in lines[:10]:
        lower = line.lower()

        if lower.startswith(ignored_prefixes):
            continue

        if len(line) <= 120:
            return line

    return file_path.stem.replace("_", " ").replace("-", " ").title()


def parse_tasting_notes(text: str) -> list[str]:
    labelled = first_match(
        [
            r"(?im)^\s*tasting notes\s*[:\-\|]\s*(.+?)\s*$",
            r"(?im)^\s*notes\s*[:\-\|]\s*(.+?)\s*$",
            r"(?im)^\s*flavo[u]?r notes\s*[:\-\|]\s*(.+?)\s*$",
        ],
        text,
    )

    if labelled is None:
        labelled = first_match(
            [
                r"(?i)notes of\s+(.+?)(?:\.|\n|$)",
                r"(?i)flavo[u]?rs of\s+(.+?)(?:\.|\n|$)",
                r"(?i)tastes? of\s+(.+?)(?:\.|\n|$)",
            ],
            text,
        )

    if labelled is None:
        return []

    # Avoid dragging in long prose after the notes.
    labelled = re.split(r"\.|\n", labelled)[0]

    parts = re.split(r",|/|·|•|\||;|\band\b", labelled, flags=re.I)

    return [
        normalise_whitespace(part).lower()
        for part in parts
        if normalise_whitespace(part)
    ]


def parse_process(text: str) -> Process:
    labelled = parse_labelled_field("process", text)
    search_text = labelled if labelled else text

    for pattern, process in PROCESS_PATTERNS:
        if re.search(pattern, search_text, re.I):
            return process

    return Process.UNKNOWN


def parse_roast_level(text: str) -> RoastLevel:
    labelled = parse_labelled_field("roast", text) or parse_labelled_field("roast level", text)
    search_text = labelled if labelled else text

    for pattern, roast_level in ROAST_PATTERNS:
        if re.search(pattern, search_text, re.I):
            return roast_level

    return RoastLevel.UNKNOWN


def parse_brew_methods(text: str) -> list[BrewMethod]:
    methods: list[BrewMethod] = []

    for pattern, method in BREW_METHOD_PATTERNS:
        if re.search(pattern, text, re.I) and method not in methods:
            methods.append(method)

    return methods


def parse_variety(text: str) -> list[str]:
    raw = (
        parse_labelled_field("variety", text)
        or parse_labelled_field("varieties", text)
        or parse_labelled_field("cultivar", text)
        or parse_labelled_field("cultivars", text)
    )

    if raw is None:
        return []

    parts = re.split(r",|/|·|\||;|\band\b", raw, flags=re.I)

    return [
        normalise_whitespace(part).lower()
        for part in parts
        if normalise_whitespace(part)
    ]


def parse_origin_country(text: str) -> str | None:
    labelled = parse_labelled_field("origin", text) or parse_labelled_field("country", text)

    if labelled:
        return empty_to_none(labelled.split(",")[0])

    match = first_match(
        [
            r"(?i)\bfrom\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
        ],
        text,
    )
    return empty_to_none(match)


def parse_description(text: str, max_chars: int = 3000) -> str | None:
    cleaned = normalise_whitespace(text)
    if not cleaned:
        return None

    return cleaned[:max_chars]


def normalise_source_url(url: str) -> str:
    parts = urlsplit(url.strip())
    if not parts.scheme or not parts.netloc:
        raise ValueError("Coffee URL must include a valid http or https host.")

    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError("Coffee URL must use http or https.")

    return urlunsplit((scheme, parts.netloc.lower(), parts.path or "/", parts.query, ""))


def build_coffee_id(name: str, source: str, source_url: str | None = None) -> str:
    identifier = normalise_source_url(source_url) if source_url else source
    return f"{slugify(name)}-{stable_id(identifier)}"


def parse_metadata_text(
    text: str,
    source: str,
    source_url: str | None = None,
) -> CoffeeRecord:
    source_path = Path(source)
    name = parse_name(text, source_path)
    coffee_id = build_coffee_id(name, source, source_url=source_url)

    return CoffeeRecord(
        coffee_id=coffee_id,
        name=name,
        roaster=parse_labelled_field("roaster", text),
        origin_country=parse_origin_country(text),
        region=parse_labelled_field("region", text),
        producer=parse_labelled_field("producer", text),
        farm=parse_labelled_field("farm", text),
        process=parse_process(text),
        variety=parse_variety(text),
        roast_level=parse_roast_level(text),
        tasting_notes=parse_tasting_notes(text),
        description=parse_description(text),
        price=parse_price(text),
        currency="GBP",
        weight_g=parse_weight_g(text),
        brew_methods=parse_brew_methods(text),
        source_url=cast(object, source_url),
        source_file=source,
    )


def parse_metadata(file_path: Path) -> CoffeeRecord:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    return parse_metadata_text(text, source=str(file_path))
