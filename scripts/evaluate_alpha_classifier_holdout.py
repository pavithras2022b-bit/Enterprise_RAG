from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from config_loader import create_embedding_model


DATA_PATH = Path("data/labeled/alpha_train_balanced.jsonl")
TEST_SIZE = 0.20
SEED = 42


def relation_key(row):
    relation = row.get("relation")

    if not relation:
        return None

    return (
        str(relation.get("subject", "")).strip().casefold(),
        str(relation.get("predicate", "")).strip().casefold(),
        str(relation.get("object", "")).strip().casefold(),
    )


def main():

    rows = []

    with DATA_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    graph_groups = defaultdict(list)
    vector_indices = []

    for index, row in enumerate(rows):

        if row["label"] == "GRAPH_FAVORING":

            key = relation_key(row)

            if key is None:
                raise ValueError(
                    f"GRAPH_FAVORING row {index} has no relation metadata."
                )

            graph_groups[key].append(index)

        else:
            vector_indices.append(index)

    print("=" * 90)
    print("RELATION-GROUPED HELD-OUT ALPHA EVALUATION")
    print("=" * 90)

    print("TOTAL EXAMPLES:", len(rows))
    print("GRAPH RELATION GROUPS:", len(graph_groups))
    print("GRAPH EXAMPLES:", sum(len(v) for v in graph_groups.values()))
    print("VECTOR EXAMPLES:", len(vector_indices))

    rng = random.Random(SEED)

    # ------------------------------------------------------------
    # GRAPH SPLIT
    # Keep all paraphrases belonging to one relationship together.
    # ------------------------------------------------------------

    graph_keys = list(graph_groups.keys())
    rng.shuffle(graph_keys)

    target_graph_test = round(
        sum(len(v) for v in graph_groups.values()) * TEST_SIZE
    )

    graph_test_indices = []
    graph_train_indices = []

    for key in graph_keys:

        group = graph_groups[key]

        if len(graph_test_indices) < target_graph_test:
            graph_test_indices.extend(group)
        else:
            graph_train_indices.extend(group)

    # ------------------------------------------------------------
    # VECTOR SPLIT
    # Vector distant-supervision examples are independent rows.
    # ------------------------------------------------------------

    rng.shuffle(vector_indices)

    vector_test_count = round(
        len(vector_indices) * TEST_SIZE
    )

    vector_test_indices = vector_indices[:vector_test_count]
    vector_train_indices = vector_indices[vector_test_count:]

    train_idx = graph_train_indices + vector_train_indices
    test_idx = graph_test_indices + vector_test_indices

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    train_graph_keys = {
        relation_key(rows[i])
        for i in graph_train_indices
    }

    test_graph_keys = {
        relation_key(rows[i])
        for i in graph_test_indices
    }

    leakage = train_graph_keys.intersection(test_graph_keys)

    print()
    print("TRAIN EXAMPLES:", len(train_idx))
    print("TEST EXAMPLES:", len(test_idx))

    print(
        "TRAIN GRAPH:",
        len(graph_train_indices),
        "| TRAIN VECTOR:",
        len(vector_train_indices),
    )

    print(
        "TEST GRAPH:",
        len(graph_test_indices),
        "| TEST VECTOR:",
        len(vector_test_indices),
    )

    print()
    print("TRAIN GRAPH RELATION GROUPS:", len(train_graph_keys))
    print("TEST GRAPH RELATION GROUPS:", len(test_graph_keys))
    print("RELATION GROUP LEAKAGE:", len(leakage))

    if leakage:
        raise RuntimeError(
            "Relation leakage detected between training and test sets."
        )

    labels = np.array(
        [
            0 if row["label"] == "GRAPH_FAVORING" else 1
            for row in rows
        ],
        dtype=int,
    )

    queries = [row["query"] for row in rows]

    print()
    print("Loading embedding model...")

    embedding_model = create_embedding_model()

    print("Embedding queries...")

    X = np.stack(
        [
            embedding_model.embed(query)
            for query in queries
        ]
    )

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = labels[train_idx]
    y_test = labels[test_idx]

    print()
    print(
        "TRAIN LABELS:",
        Counter(
            "GRAPH_FAVORING" if x == 0 else "VECTOR_FAVORING"
            for x in y_train
        ),
    )

    print(
        "TEST LABELS:",
        Counter(
            "GRAPH_FAVORING" if x == 0 else "VECTOR_FAVORING"
            for x in y_test
        ),
    )

    # Temporary evaluation model only.
    # Production model.joblib is NEVER overwritten.

    model = LogisticRegression(
        max_iter=1000,
        random_state=SEED,
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)

    vector_class_index = list(model.classes_).index(1)

    alphas = probabilities[:, vector_class_index]

    predictions = (alphas >= 0.5).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    graph_alphas = alphas[y_test == 0]
    vector_alphas = alphas[y_test == 1]

    print()
    print("=" * 90)
    print("LEAKAGE-SAFE HELD-OUT RESULTS")
    print("=" * 90)

    print("ACCURACY:", round(float(accuracy), 4))

    print()
    print("AVERAGE ALPHA")
    print(
        "GRAPH_FAVORING:",
        round(float(graph_alphas.mean()), 4),
    )
    print(
        "VECTOR_FAVORING:",
        round(float(vector_alphas.mean()), 4),
    )

    print()
    print("ALPHA RANGE")

    print(
        "GRAPH:",
        round(float(graph_alphas.min()), 4),
        "->",
        round(float(graph_alphas.max()), 4),
    )

    print(
        "VECTOR:",
        round(float(vector_alphas.min()), 4),
        "->",
        round(float(vector_alphas.max()), 4),
    )

    print()
    print("CONFUSION MATRIX")
    print("                  Pred Graph   Pred Vector")

    print(
        f"Actual Graph      {matrix[0,0]:10d}   {matrix[0,1]:11d}"
    )

    print(
        f"Actual Vector     {matrix[1,0]:10d}   {matrix[1,1]:11d}"
    )

    print()
    print("CLASSIFICATION REPORT")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "GRAPH_FAVORING",
                "VECTOR_FAVORING",
            ],
            digits=4,
        )
    )

    print("=" * 90)
    print("MISCLASSIFIED HELD-OUT QUERIES")
    print("=" * 90)

    mistakes = []

    for local_index, original_index in enumerate(test_idx):

        if predictions[local_index] == y_test[local_index]:
            continue

        mistakes.append(
            (
                float(alphas[local_index]),
                rows[original_index],
                int(y_test[local_index]),
                int(predictions[local_index]),
            )
        )

    print("TOTAL MISCLASSIFIED:", len(mistakes))

    for alpha, row, actual, predicted in sorted(
        mistakes,
        key=lambda item: item[0],
    ):

        print()
        print("-" * 90)

        print("QUERY:", row["query"])

        print(
            "ACTUAL:",
            "GRAPH_FAVORING"
            if actual == 0
            else "VECTOR_FAVORING",
        )

        print(
            "PREDICTED:",
            "GRAPH_FAVORING"
            if predicted == 0
            else "VECTOR_FAVORING",
        )

        print("ALPHA:", round(alpha, 4))
        print("SOURCE:", row.get("source"))

        if row.get("relation"):
            print("RELATION:", row["relation"])

    print()
    print("=" * 90)
    print("EVALUATION SAFETY CHECK")
    print("=" * 90)

    print("RELATION GROUP LEAKAGE:", len(leakage))

    print(
        "Production model models/alpha_classifier/model.joblib"
        " was NOT modified."
    )

    print("=" * 90)


if __name__ == "__main__":
    main()
