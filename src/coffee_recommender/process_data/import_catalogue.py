from __future__ import annotations

import argparse

from ..catalogue_store import import_catalogue_from_csvs
from ..config import get_data_paths
from ..db import create_session_factory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile-version",
        default="csv_import_v1",
        help="Version label stored on imported sensory profiles.",
    )
    args = parser.parse_args()

    imported = import_catalogue_from_csvs(
        data_paths=get_data_paths(),
        session_factory=create_session_factory(),
        profile_version=args.profile_version,
    )
    print(f"Imported {imported} catalogue coffees into the database.")


if __name__ == "__main__":
    main()
