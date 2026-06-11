from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .landscape import CoffeeFeatures, build_single_coffee_features
from .openai_client import DEFAULT_CHAT_MODEL, DEFAULT_EMBEDDING_MODEL
from .process_data.embed_coffee import embed_coffee_record
from .process_data.extract_sensory import extract_sensory_vector_llm
from .process_data.parse_metadata import normalise_source_url, parse_metadata_text
from .schemas import CoffeeRecord, SensoryVector


USER_AGENT = "CoffeeRecommender/1.0 (+https://local.streamlit.app)"
MIN_EXTRACTED_WORDS = 20


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        if tag in {"p", "div", "section", "article", "h1", "h2", "h3", "h4", "li", "br"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        if tag in {"p", "div", "section", "article", "h1", "h2", "h3", "h4", "li"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        cleaned = " ".join(unescape(data).split())
        if cleaned:
            self._parts.append(cleaned)

    def text(self) -> str:
        lines = [" ".join(part.split()) for part in "".join(self._parts).splitlines()]
        deduped: list[str] = []
        previous = ""
        for line in lines:
            line = line.strip()
            if not line or line == previous:
                continue
            deduped.append(line)
            previous = line
        return "\n".join(deduped)


@dataclass(frozen=True)
class ReviewedCoffeeFromUrl:
    url: str
    coffee: CoffeeRecord
    sensory: SensoryVector
    features: CoffeeFeatures
    extracted_text: str


def fetch_url_html(url: str, timeout: int = 15) -> tuple[str, str]:
    normalized_url = normalise_source_url(url)
    request = Request(
        normalized_url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
        },
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get_content_type()
            if content_type != "text/html":
                raise ValueError(f"Coffee URL must return HTML content, got {content_type}.")
            charset = response.headers.get_content_charset() or "utf-8"
            html = response.read().decode(charset, errors="ignore")
    except HTTPError as exc:
        raise ValueError(f"Could not fetch coffee URL ({exc.code}).") from exc
    except URLError as exc:
        raise ValueError(f"Could not fetch coffee URL: {exc.reason}.") from exc

    if not html.strip():
        raise ValueError("Coffee URL returned an empty HTML page.")

    return normalized_url, html


def extract_product_text_from_html(html: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(html)
    text = parser.text()
    if len(text.split()) < MIN_EXTRACTED_WORDS:
        raise ValueError("Coffee URL page did not contain enough visible text to parse.")
    return text


def _validate_parsed_coffee(coffee: CoffeeRecord) -> None:
    if not coffee.name.strip():
        raise ValueError("Could not parse a coffee name from the pasted URL.")

    has_supporting_metadata = any(
        (
            coffee.origin_country,
            coffee.region,
            coffee.producer,
            coffee.farm,
            coffee.price is not None,
            coffee.weight_g is not None,
            bool(coffee.tasting_notes),
            bool(coffee.description),
            coffee.process.value != "unknown",
        )
    )
    if not has_supporting_metadata:
        raise ValueError("Coffee URL page did not contain enough product metadata to review.")


def prepare_reviewed_coffee_from_url(
    url: str,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    sensory_model: str = DEFAULT_CHAT_MODEL,
) -> ReviewedCoffeeFromUrl:
    normalized_url, html = fetch_url_html(url)
    extracted_text = extract_product_text_from_html(html)
    coffee = parse_metadata_text(
        extracted_text,
        source=f"url:{normalized_url}",
        source_url=normalized_url,
    )
    _validate_parsed_coffee(coffee)
    sensory = extract_sensory_vector_llm(coffee, model=sensory_model, temperature=0.0)
    embedding = embed_coffee_record(coffee, model=embedding_model)
    features = build_single_coffee_features(coffee, sensory, embedding)
    return ReviewedCoffeeFromUrl(
        url=normalized_url,
        coffee=coffee,
        sensory=sensory,
        features=features,
        extracted_text=extracted_text,
    )
