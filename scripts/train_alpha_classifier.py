from __future__ import annotations

import json
from pathlib import Path

import joblib

from adaptive_retrieval.classifier import AlphaLabel, LabeledQuery
from adaptive_retrieval.train import train_classifier
from config_loader import create_embedding_model


INPUT_PATH = Path("data/labeled/alpha_train_balanced.jsonl")
OUTPUT_PATH = Path("models/alpha_classifier/model.joblib")


def main() -> None:
    labeled_queries = []

    with INPUT_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            row = json.loads(line)

            query = str(row.get("query", "")).strip()
            label_text = str(row.get("label", "")).strip()

            if not query or not label_text:
                continue

            label = AlphaLabel(label_text)

            labeled_queries.append(
                LabeledQuery(
                    query=query,
                    label=label,
                )
            )

    print("=" * 70)
    print("TRAINING LEARNED ALPHA CLASSIFIER")
    print("=" * 70)
    print("TRAINING EXAMPLES:", len(labeled_queries))

    embedding_model = create_embedding_model()

    classifier = train_classifier(
        labeled_queries=labeled_queries,
        embedding_model=embedding_model,
        max_iter=1000,
        seed=42,
    )

    if not classifier.is_fitted:
        raise RuntimeError("Classifier training completed but model is not fitted.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        classifier._model,
        OUTPUT_PATH,
    )

    print()
    print("TRAINING COMPLETE")
    print("MODEL FITTED:", classifier.is_fitted)
    print("MODEL SAVED:", OUTPUT_PATH)
    print("=" * 70)


if __name__ == "__main__":
    main()
