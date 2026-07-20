from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Callable
from urllib.parse import urljoin, urlsplit

from sqlalchemy.orm import Session, sessionmaker

from .process_data.parse_metadata import normalise_source_url
from .reviewed_coffee_url import fetch_url_html
from .source_pipeline import FetchPageResult, fetch_and_store_page

PRODUCT_PATH_HINTS = (
    "/product-page/",
    "/product/",
    "/products/",
    "/coffee/",
    "/coffees/",
    "/bean/",
    "/beans/",
)
LISTING_PATH_HINTS = (
    "/shop",
    "/coffee",
    "/coffees",
    "/products",
    "/collections",
    "/category",
)
SKIP_PATH_HINTS = (
    "/account",
    "/login",
    "/cart",
    "/checkout",
    "/search",
    "/blog",
    "/blogs",
    "/about",
    "/contact",
    "/policy",
    "/privacy",
    "/terms",
    "/faq",
    "/subscription",
)
SKIP_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg", ".pdf", ".zip")
SKIP_TEXT_HINTS = (
    "subscription",
    "subscribe",
    "gift card",
    "workshop",
)
PRODUCT_TEXT_HINTS = (
    "coffee",
    "filter",
    "espresso",
    "gesha",
    "geisha",
    "washed",
    "natural",
    "anaerobic",
    "honey",
)


class _AnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._current_href: str | None = None
        self._current_text_parts: list[str] = []
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attributes = dict(attrs)
        href = attributes.get("href")
        if href:
            self._current_href = href
            self._current_text_parts = []

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            cleaned = " ".join(unescape(data).split())
            if cleaned:
                self._current_text_parts.append(cleaned)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._current_href is None:
            return
        self.links.append((self._current_href, " ".join(self._current_text_parts).strip()))
        self._current_href = None
        self._current_text_parts = []


@dataclass(frozen=True)
class DiscoveryCandidate:
    url: str
    score: int
    anchor_text: str
    discovered_from: str
    reasons: list[str]


@dataclass(frozen=True)
class DiscoveryResult:
    listing_pages_scanned: int
    discovered_candidates: list[DiscoveryCandidate]
    stored_pages: list[FetchPageResult]


def discover_product_page_urls(
    *,
    start_url: str,
    allowed_domains: list[str] | None = None,
    max_listing_pages: int = 10,
    max_depth: int = 2,
    max_product_pages: int = 25,
    fetcher: Callable[[str], tuple[str, str]] = fetch_url_html,
) -> list[DiscoveryCandidate]:
    candidates, _ = _discover_candidates(
        start_url=start_url,
        allowed_domains=allowed_domains,
        max_listing_pages=max_listing_pages,
        max_depth=max_depth,
        max_product_pages=max_product_pages,
        fetcher=fetcher,
    )
    return candidates


def discover_and_store_product_pages(
    *,
    session_factory: sessionmaker[Session],
    start_url: str,
    allowed_domains: list[str] | None = None,
    store_id: int | None = None,
    max_listing_pages: int = 10,
    max_depth: int = 2,
    max_product_pages: int = 25,
    fetcher: Callable[[str], tuple[str, str]] = fetch_url_html,
) -> DiscoveryResult:
    candidates, listing_pages_scanned = _discover_candidates(
        start_url=start_url,
        allowed_domains=allowed_domains,
        max_listing_pages=max_listing_pages,
        max_depth=max_depth,
        max_product_pages=max_product_pages,
        fetcher=fetcher,
    )

    stored_pages: list[FetchPageResult] = []
    for candidate in candidates:
        try:
            stored_pages.append(
                fetch_and_store_page(
                    session_factory=session_factory,
                    url=candidate.url,
                    store_id=store_id,
                    page_type="product",
                )
            )
        except ValueError:
            continue

    return DiscoveryResult(
        listing_pages_scanned=listing_pages_scanned,
        discovered_candidates=candidates,
        stored_pages=stored_pages,
    )


def _discover_candidates(
    *,
    start_url: str,
    allowed_domains: list[str] | None,
    max_listing_pages: int,
    max_depth: int,
    max_product_pages: int,
    fetcher: Callable[[str], tuple[str, str]],
) -> tuple[list[DiscoveryCandidate], int]:
    normalized_start_url = normalise_source_url(start_url)
    allowed = _build_allowed_domains(normalized_start_url, allowed_domains or [])

    queue: list[tuple[str, int]] = [(normalized_start_url, 0)]
    scanned_pages: set[str] = set()
    queued_pages = {normalized_start_url}
    candidates: dict[str, DiscoveryCandidate] = {}

    while queue and len(scanned_pages) < max_listing_pages and len(candidates) < max_product_pages:
        page_url, depth = queue.pop(0)
        scanned_pages.add(page_url)

        try:
            normalized_page_url, html = fetcher(page_url)
        except ValueError:
            if page_url == normalized_start_url:
                raise
            continue
        for candidate_url, anchor_text in _extract_links(html, normalized_page_url):
            if not _is_allowed_url(candidate_url, allowed):
                continue
            if _should_skip_candidate(candidate_url, anchor_text):
                continue

            score, reasons = _score_product_candidate(candidate_url, anchor_text)
            if score >= 2:
                current = candidates.get(candidate_url)
                discovered = DiscoveryCandidate(
                    url=candidate_url,
                    score=score,
                    anchor_text=anchor_text,
                    discovered_from=normalized_page_url,
                    reasons=reasons,
                )
                if current is None or discovered.score > current.score:
                    candidates[candidate_url] = discovered
                continue

            if depth >= max_depth:
                continue
            if not _looks_like_listing_page(candidate_url, anchor_text):
                continue
            if candidate_url in scanned_pages or candidate_url in queued_pages:
                continue

            queue.append((candidate_url, depth + 1))
            queued_pages.add(candidate_url)

    return (
        sorted(
            candidates.values(),
            key=lambda candidate: (-candidate.score, candidate.url),
        )[:max_product_pages],
        len(scanned_pages),
    )


def _extract_links(html: str, base_url: str) -> list[tuple[str, str]]:
    parser = _AnchorParser()
    parser.feed(html)

    links: list[tuple[str, str]] = []
    seen: set[str] = set()
    for href, anchor_text in parser.links:
        absolute_url = urljoin(base_url, href)
        try:
            normalized_url = normalise_source_url(absolute_url)
        except ValueError:
            continue
        if normalized_url in seen:
            continue
        seen.add(normalized_url)
        links.append((normalized_url, anchor_text))
    return links


def _build_allowed_domains(start_url: str, allowed_domains: list[str]) -> set[str]:
    domains = {_normalize_domain(urlsplit(start_url).netloc)}
    domains.update(_normalize_domain(domain) for domain in allowed_domains if domain.strip())
    return domains


def _normalize_domain(domain: str) -> str:
    host = domain.lower().strip()
    if "://" in host:
        host = urlsplit(host).netloc.lower().strip()
    if ":" in host:
        host = host.split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    return host


def _is_allowed_url(url: str, allowed_domains: set[str]) -> bool:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    domain = _normalize_domain(parsed.netloc)
    return any(domain == allowed or domain.endswith(f".{allowed}") for allowed in allowed_domains)


def _should_skip_candidate(url: str, anchor_text: str) -> bool:
    lowered = url.lower()
    if any(lowered.endswith(suffix) for suffix in SKIP_SUFFIXES):
        return True
    lowered_anchor_text = anchor_text.lower()
    return (
        any(hint in lowered for hint in SKIP_PATH_HINTS)
        or any(hint in lowered for hint in SKIP_TEXT_HINTS)
        or any(hint in lowered_anchor_text for hint in SKIP_TEXT_HINTS)
    )


def _looks_like_listing_page(url: str, anchor_text: str) -> bool:
    lowered_url = url.lower()
    lowered_text = anchor_text.lower()
    return any(hint in lowered_url for hint in LISTING_PATH_HINTS) or any(
        token in lowered_text for token in ("shop", "coffee", "browse", "collection", "products")
    )


def _score_product_candidate(url: str, anchor_text: str) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    lowered_url = url.lower()
    lowered_text = anchor_text.lower()

    if any(hint in lowered_url for hint in PRODUCT_PATH_HINTS):
        score += 2
        reasons.append("product-like url")

    if "/products/" in lowered_url and lowered_url.rstrip("/").count("/") >= 4:
        score += 1
        reasons.append("deep product path")

    if any(token in lowered_text for token in PRODUCT_TEXT_HINTS):
        score += 1
        reasons.append("coffee-like anchor text")

    if lowered_text and any(char.isdigit() for char in lowered_text):
        score += 1
        reasons.append("anchor includes variant detail")

    return score, reasons
