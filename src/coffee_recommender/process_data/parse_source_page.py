from __future__ import annotations

import argparse

from ..db import create_session_factory
from ..source_pipeline import parse_stored_page


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_page_id", type=int, help="Stored source page ID to parse.")
    parser.add_argument(
        "--metadata-model",
        default="gpt-5.4-mini",
        help="Model used for factual metadata extraction.",
    )
    parser.add_argument(
        "--sensory-model",
        default="gpt-5.4-mini",
        help="Model used for sensory extraction.",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Model used for embeddings.",
    )
    parser.add_argument(
        "--schema-version",
        default="coffee_record_v1",
        help="Version label recorded on the parse run.",
    )
    parser.add_argument(
        "--parser-version",
        default="page_parser_v2",
        help="Version label recorded on the parse run.",
    )
    args = parser.parse_args()

    result = parse_stored_page(
        session_factory=create_session_factory(),
        source_page_id=args.source_page_id,
        metadata_model=args.metadata_model,
        sensory_model=args.sensory_model,
        embedding_model=args.embedding_model,
        schema_version=args.schema_version,
        parser_version=args.parser_version,
    )
    print(
        f"Parsed source page {args.source_page_id} into coffee {result.coffee_id} "
        f"(run {result.parse_run_id}, status {result.parse_status})."
    )


if __name__ == "__main__":
    main()
