from __future__ import annotations

import argparse

from ..db import create_session_factory
from ..source_pipeline import ensure_source_store, fetch_and_store_page


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="Product page URL to fetch and store.")
    parser.add_argument("--store-name", help="Optional store name for grouping source pages.")
    parser.add_argument("--base-url", help="Optional store base URL.")
    parser.add_argument(
        "--allow-domain",
        action="append",
        default=[],
        help="Allowed domain for this store. Repeat to add more than one.",
    )
    parser.add_argument(
        "--page-type",
        default="product",
        help="Stored page type label, for example product or category.",
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

    result = fetch_and_store_page(
        session_factory=session_factory,
        url=args.url,
        store_id=store_id,
        page_type=args.page_type,
    )
    print(
        f"Stored source page {result.source_page_id} for {result.normalized_url} "
        f"(hash {result.content_hash[:12]}...)."
    )


if __name__ == "__main__":
    main()
