from __future__ import annotations

import unittest
from unittest.mock import patch

from coffee_recommender.source_discovery import (
    discover_and_store_product_pages,
    discover_product_page_urls,
)
from coffee_recommender.source_pipeline import FetchPageResult


def _html(*links: tuple[str, str]) -> str:
    anchor_tags = "\n".join(f'<a href="{href}">{text}</a>' for href, text in links)
    return f"<html><body>{anchor_tags}</body></html>"


class SourceDiscoveryTests(unittest.TestCase):
    def test_discover_product_page_urls_follows_listing_pages(self) -> None:
        pages = {
            "https://example.com/shop": (
                "https://example.com/shop",
                _html(
                    ("/collections/filter", "Filter Coffee"),
                    ("/blog/how-we-roast", "Roasting Notes"),
                ),
            ),
            "https://example.com/collections/filter": (
                "https://example.com/collections/filter",
                _html(
                    ("/products/kenya-gichathaini", "Kenya Gichathaini Washed"),
                    ("/products/colombia-gesha", "Colombia Gesha Filter"),
                    ("https://other.com/products/skip-me", "External Coffee"),
                ),
            ),
        }

        def fake_fetcher(url: str) -> tuple[str, str]:
            return pages[url]

        candidates = discover_product_page_urls(
            start_url="https://example.com/shop",
            fetcher=fake_fetcher,
        )

        self.assertEqual(
            [candidate.url for candidate in candidates],
            [
                "https://example.com/products/colombia-gesha",
                "https://example.com/products/kenya-gichathaini",
            ],
        )
        self.assertTrue(all(candidate.discovered_from == "https://example.com/collections/filter" for candidate in candidates))

    def test_discovery_respects_allowed_domains_and_skips_non_product_pages(self) -> None:
        pages = {
            "https://shop.example.com/coffee": (
                "https://shop.example.com/coffee",
                _html(
                    ("https://cdn.example.com/products/image.jpg", "Image"),
                    ("https://shop.example.com/account", "Account"),
                    ("https://shop.example.com/products/panama-gesha", "Panama Gesha"),
                    ("https://news.example.com/blog/post", "Blog"),
                ),
            ),
        }

        def fake_fetcher(url: str) -> tuple[str, str]:
            return pages[url]

        candidates = discover_product_page_urls(
            start_url="https://shop.example.com/coffee",
            allowed_domains=["shop.example.com"],
            fetcher=fake_fetcher,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].url, "https://shop.example.com/products/panama-gesha")

    def test_discovery_skips_subscription_products(self) -> None:
        pages = {
            "https://example.com/coffees": (
                "https://example.com/coffees",
                _html(
                    ("/product-page/filter-coffee-subscription", "Filter Coffee Subscription"),
                    ("/product-page/kenya-karinga-farmers", "Kenya Karinga Washed"),
                ),
            ),
        }

        def fake_fetcher(url: str) -> tuple[str, str]:
            return pages[url]

        candidates = discover_product_page_urls(
            start_url="https://example.com/coffees",
            fetcher=fake_fetcher,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].url, "https://example.com/product-page/kenya-karinga-farmers")

    def test_discover_and_store_product_pages_passes_candidates_to_source_pipeline(self) -> None:
        pages = {
            "https://example.com/shop": (
                "https://example.com/shop",
                _html(
                    ("/products/kenya-gichathaini", "Kenya Gichathaini Washed"),
                    ("/products/colombia-gesha", "Colombia Gesha Filter"),
                ),
            ),
        }

        def fake_fetcher(url: str) -> tuple[str, str]:
            return pages[url]

        stored = {
            "https://example.com/products/colombia-gesha": FetchPageResult(
                source_page_id=11,
                normalized_url="https://example.com/products/colombia-gesha",
                content_hash="hash-1",
            ),
            "https://example.com/products/kenya-gichathaini": FetchPageResult(
                source_page_id=12,
                normalized_url="https://example.com/products/kenya-gichathaini",
                content_hash="hash-2",
            ),
        }

        with patch(
            "coffee_recommender.source_discovery.fetch_and_store_page",
            side_effect=lambda **kwargs: stored[kwargs["url"]],
        ) as fetch_and_store:
            result = discover_and_store_product_pages(
                session_factory=object(),
                start_url="https://example.com/shop",
                store_id=7,
                fetcher=fake_fetcher,
            )

        self.assertEqual(result.listing_pages_scanned, 1)
        self.assertEqual(
            [page.normalized_url for page in result.stored_pages],
            [
                "https://example.com/products/colombia-gesha",
                "https://example.com/products/kenya-gichathaini",
            ],
        )
        self.assertEqual(fetch_and_store.call_count, 2)
        self.assertTrue(all(call.kwargs["store_id"] == 7 for call in fetch_and_store.call_args_list))


if __name__ == "__main__":
    unittest.main()
