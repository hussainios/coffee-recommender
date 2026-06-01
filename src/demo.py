from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from landscape import load_feature_index, recommend_from_landscape
from parse_review import parse_review_event


def main(
    review: str,
    reviewed_coffee_id: str,
    coffees_path: Path,
    sensory_path: Path,
    embeddings_path: Path,
    top_k: int,
) -> None:
    features = load_feature_index(coffees_path, sensory_path, embeddings_path)
    reviewed_coffee = features[reviewed_coffee_id]
    event = parse_review_event(review, reviewed_coffee)
    recommendations = recommend_from_landscape(features, [event], top_k=top_k)

    print("Parsed review event:")
    print(json.dumps(event, indent=2, ensure_ascii=False))
    print()
    print("Top recommendations:")
    for item in recommendations:
        print(f"- {item['name']} ({item['coffee_id']}): {item['score']}")
        print(f"  temperature: {item['temperature']}")
        print(f"  debug: {json.dumps(item['debug'], ensure_ascii=False)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the landscape coffee recommender.")
    parser.add_argument("--review", required=True, help="Natural-language coffee review text.")
    parser.add_argument("--reviewed-coffee-id", required=True, help="Coffee ID the review is about.")
    parser.add_argument("--coffees-path", default="data/processed/coffees.csv")
    parser.add_argument("--sensory-path", default="data/processed/coffee_sensory_vectors.csv")
    parser.add_argument("--embeddings-path", default="data/processed/coffee_embeddings.csv")
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    main(
        review=args.review,
        reviewed_coffee_id=args.reviewed_coffee_id,
        coffees_path=Path(args.coffees_path),
        sensory_path=Path(args.sensory_path),
        embeddings_path=Path(args.embeddings_path),
        top_k=args.top_k,
    )
