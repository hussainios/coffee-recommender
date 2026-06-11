from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from coffee_recommender.openai_client import DEFAULT_CHAT_MODEL, DEFAULT_EMBEDDING_MODEL
from coffee_recommender.process_data import build_dataset, build_embeddings
from coffee_recommender.schemas import CoffeeRecord, Process, RoastLevel, SensoryVector


def _coffee_record(coffee_id: str, name: str, source_file: str) -> CoffeeRecord:
    return CoffeeRecord(
        coffee_id=coffee_id,
        name=name,
        roaster="April",
        origin_country="Kenya",
        region="Nyeri",
        producer="Producer A",
        process=Process.WASHED,
        variety=["sl28"],
        roast_level=RoastLevel.LIGHT,
        tasting_notes=["blackberry", "citrus"],
        description="Bright and sweet filter coffee.",
        source_file=source_file,
    )


def _sensory_vector(coffee_id: str) -> SensoryVector:
    return SensoryVector(
        coffee_id=coffee_id,
        acidity=0.7,
        sweetness=0.6,
        body=0.5,
        bitterness=0.1,
        fruitiness=0.8,
        chocolate_nutty=0.2,
        floral=0.3,
        funky_fermented=0.0,
        roasty=0.1,
        clean_cup=0.9,
        confidence=0.8,
        evidence={
            "acidity": ["bright"],
            "sweetness": ["sweet"],
            "body": [],
            "bitterness": [],
            "fruitiness": ["berry"],
            "chocolate_nutty": [],
            "floral": [],
            "funky_fermented": [],
            "roasty": [],
            "clean_cup": ["clean"],
        },
    )


class PipelineTests(unittest.TestCase):
    def test_build_dataset_writes_all_expected_csv_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            output_dir = Path(tmpdir) / "processed"
            raw_dir.mkdir()
            (raw_dir / "coffee-a.txt").write_text("Coffee A", encoding="utf-8")
            (raw_dir / "coffee-b.txt").write_text("Coffee B", encoding="utf-8")

            records = [
                _coffee_record("coffee-a", "Coffee A", str(raw_dir / "coffee-a.txt")),
                _coffee_record("coffee-b", "Coffee B", str(raw_dir / "coffee-b.txt")),
            ]
            sensory_vectors = {
                record.coffee_id: _sensory_vector(record.coffee_id)
                for record in records
            }

            with (
                patch.object(build_dataset, "parse_metadata", side_effect=records),
                patch.object(
                    build_dataset,
                    "extract_sensory_vector_llm",
                    side_effect=lambda coffee, model: sensory_vectors[coffee.coffee_id],
                ),
                patch.object(
                    build_dataset,
                    "build_embedding_records",
                    return_value=[
                        {
                            "coffee_id": "coffee-a",
                            "embedding_model": DEFAULT_EMBEDDING_MODEL,
                            "embedding_text": "A",
                            "embedding": "[0.1, 0.2, 0.3]",
                        },
                        {
                            "coffee_id": "coffee-b",
                            "embedding_model": DEFAULT_EMBEDDING_MODEL,
                            "embedding_text": "B",
                            "embedding": "[0.3, 0.2, 0.1]",
                        },
                    ],
                ),
            ):
                build_dataset.main(
                    raw_dir=raw_dir,
                    output_dir=output_dir,
                    sensory_model=DEFAULT_CHAT_MODEL,
                    embedding_model=DEFAULT_EMBEDDING_MODEL,
                )

            coffees = pd.read_csv(output_dir / "coffees.csv")
            sensory = pd.read_csv(output_dir / "coffee_sensory_vectors.csv")
            embeddings = pd.read_csv(output_dir / "coffee_embeddings.csv")

            self.assertEqual(set(coffees["coffee_id"]), {"coffee-a", "coffee-b"})
            self.assertEqual(set(sensory["coffee_id"]), {"coffee-a", "coffee-b"})
            self.assertEqual(set(embeddings["coffee_id"]), {"coffee-a", "coffee-b"})

    def test_build_embeddings_writes_expected_output_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            coffees_path = Path(tmpdir) / "coffees.csv"
            output_path = Path(tmpdir) / "coffee_embeddings.csv"
            pd.DataFrame(
                [
                    {
                        "coffee_id": "coffee-a",
                        "name": "Coffee A",
                        "process": "washed",
                        "tasting_notes": "berry, citrus",
                    },
                    {
                        "coffee_id": "coffee-b",
                        "name": "Coffee B",
                        "process": "natural",
                        "tasting_notes": "chocolate, plum",
                    },
                ]
            ).to_csv(coffees_path, index=False)

            with patch.object(
                build_embeddings,
                "build_embedding_records",
                return_value=[
                    {
                        "coffee_id": "coffee-a",
                        "embedding_model": DEFAULT_EMBEDDING_MODEL,
                        "embedding_text": "A",
                        "embedding": "[0.1, 0.2, 0.3]",
                    },
                    {
                        "coffee_id": "coffee-b",
                        "embedding_model": DEFAULT_EMBEDDING_MODEL,
                        "embedding_text": "B",
                        "embedding": "[0.3, 0.2, 0.1]",
                    },
                ],
            ):
                build_embeddings.main(
                    coffees_path=coffees_path,
                    output_path=output_path,
                    model=DEFAULT_EMBEDDING_MODEL,
                )

            embeddings = pd.read_csv(output_path)
            self.assertEqual(
                list(embeddings.columns),
                ["coffee_id", "embedding_model", "embedding_text", "embedding"],
            )
            self.assertEqual(len(embeddings), 2)


if __name__ == "__main__":
    unittest.main()
