from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[1]
PROCESS_DATA_DIR = Path(__file__).resolve().parent

for path in (SRC_DIR, PROCESS_DATA_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from process_data.parse_metadata import parse_metadata
from process_data.extract_sensory import extract_sensory_vector_llm
from process_data.embed_coffee import build_embedding_records
from openai_client import DEFAULT_CHAT_MODEL, DEFAULT_EMBEDDING_MODEL


def main(raw_dir: Path, output_dir: Path, sensory_model: str, embedding_model: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    text_files = sorted(raw_dir.glob("*.txt"))

    if not text_files:
        raise FileNotFoundError(f"No .txt files found in {raw_dir}")

    coffee_records = []
    sensory_records = []

    for file_path in tqdm(text_files, desc="Processing coffee pages"):
        coffee = parse_metadata(file_path)
        coffee_records.append(coffee.model_dump())

        sensory = extract_sensory_vector_llm(coffee, model=sensory_model)
        sensory_records.append(sensory.model_dump())

    coffees_df = pd.DataFrame(coffee_records)
    coffees_path = output_dir / "coffees.csv"
    coffees_df.to_csv(coffees_path, index=False)

    print(f"Wrote {coffees_path}")

    sensory_df = pd.DataFrame(sensory_records)
    sensory_path = output_dir / "coffee_sensory_vectors.csv"
    sensory_df.to_csv(sensory_path, index=False)
    print(f"Wrote {sensory_path}")

    embeddings_df = pd.DataFrame(build_embedding_records(coffees_df, model=embedding_model))
    embeddings_path = output_dir / "coffee_embeddings.csv"
    embeddings_df.to_csv(embeddings_path, index=False)
    print(f"Wrote {embeddings_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--sensory-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)

    args = parser.parse_args()

    main(
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        sensory_model=args.sensory_model,
        embedding_model=args.embedding_model,
    )
