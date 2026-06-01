from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
PROCESS_DATA_DIR = Path(__file__).resolve().parent

for path in (SRC_DIR, PROCESS_DATA_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from process_data.embed_coffee import build_embedding_records
from openai_client import DEFAULT_EMBEDDING_MODEL


def main(coffees_path: Path, output_path: Path, model: str) -> None:
    if not coffees_path.exists():
        raise FileNotFoundError(f"Coffee CSV not found: {coffees_path}")

    coffees = pd.read_csv(coffees_path)
    embeddings = pd.DataFrame(build_embedding_records(coffees, model=model))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate coffee text embeddings.")
    parser.add_argument("--coffees-path", default="data/processed/coffees.csv")
    parser.add_argument("--output-path", default="data/processed/coffee_embeddings.csv")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)

    args = parser.parse_args()
    main(
        coffees_path=Path(args.coffees_path),
        output_path=Path(args.output_path),
        model=args.model,
    )
