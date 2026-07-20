from __future__ import annotations

import argparse

from ..db import create_session_factory
from ..source_discovery import discover_and_store_product_pages
from ..source_pipeline import ensure_source_store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("start_url", help="Store or collection URL to crawl for coffee product pages.")
    parser.add_argument("--store-name", help="Optional store name for grouping source pages.")
    parser.add_argument("--base-url", help="Optional canonical store base URL.")
    parser.add_argument(
        "--allow-domain",
        action="append",
        default=[],
        help="Allowed domain for discovery. Repeat to add more than one.",
    )
    parser.add_argument(
        "--max-listing-pages",
        type=int,
        default=10,
        help="Maximum number of non-product pages to scan during discovery.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum number of link hops from the start URL.",
    )
    parser.add_argument(
        "--max-product-pages",
        type=int,
        default=25,
        help="Maximum number of product pages to discover and store.",
    )
    args = parser.parse_args()

    session_factory = create_session_factory()
    store_id: int | None = None
    if args.store_name and args.base_url:
        store_id = ensure_source_store(
            session_factory=session_factory,
            name=args.store_name,
            base_url=args.base_url,
            allowed_domains=args.allow_domain,
        )

    result = discover_and_store_product_pages(
        session_factory=session_factory,
        start_url=args.start_url,
        allowed_domains=args.allow_domain,
        store_id=store_id,
        max_listing_pages=args.max_listing_pages,
        max_depth=args.max_depth,
        max_product_pages=args.max_product_pages,
    )

    print(
        f"Discovered {len(result.discovered_candidates)} product pages and stored "
        f"{len(result.stored_pages)} source pages."
    )
    for stored_page in result.stored_pages:
        print(f"- source_page_id={stored_page.source_page_id} url={stored_page.normalized_url}")


if __name__ == "__main__":
    main()
