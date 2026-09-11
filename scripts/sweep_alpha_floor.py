from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_loader import (
    create_alpha_classifier,
    create_embedding_model,
    create_neo4j_connection,
    create_vector_store,
)
from evaluation.retrieval_metrics import calculate_mrr, calculate_ndcg
from retrieval.entity_linker import EntityLinker
from retrieval.hybrid_controller import HybridRetrievalController
from scripts.run_production_pipeline import parse_val_examples


def main():
    print("Loading retrieval components...")

    embedding_model = create_embedding_model()
    vector_store = create_vector_store()
    neo4j_connection = create_neo4j_connection()

    entity_linker = EntityLinker(
        neo4j_connection,
        refresh_on_init=True,
    )

    controller = HybridRetrievalController(
        vector_store=vector_store,
        graph_connection=neo4j_connection,
        embedding_model=embedding_model,
        entity_linker=entity_linker,
    )

    classifier = create_alpha_classifier()

    examples = parse_val_examples(
        PROJECT_ROOT / "data" / "labeled" / "lora_val.jsonl"
    )

    floors = [None, 0.40, 0.45, 0.50, 0.55, 0.60]

    totals = {
        floor: {"mrr": 0.0, "ndcg": 0.0}
        for floor in floors
    }

    print(f"Validation examples: {len(examples)}")
    print("Running alpha-floor sweep...")

    for i, item in enumerate(examples, start=1):

        query_embedding = embedding_model.embed(item.query)
        raw_alpha = float(
            classifier.predict_alpha(item.query, query_embedding)
        )

        entities = entity_linker.link(item.query)

        vector_hits = controller.run_vector_retrieval(
            item.query,
            k=5,
        )

        graph_hits = controller.run_graph_retrieval(
            entities,
            k=5,
        )

        for floor in floors:

            alpha = (
                raw_alpha
                if floor is None
                else max(raw_alpha, floor)
            )

            candidates = controller.static_fuse(
                vector_hits,
                graph_hits,
                alpha=alpha,
            )

            ids = [c.candidate_id for c in candidates]

            totals[floor]["mrr"] += calculate_mrr(
                ids,
                item.ground_truth_chunk_ids,
                k=5,
            )

            totals[floor]["ndcg"] += calculate_ndcg(
                ids,
                item.ground_truth_chunk_ids,
                k=5,
            )

        print(
            f"[{i:02d}/{len(examples):02d}] "
            f"raw alpha={raw_alpha:.4f}"
        )

    n = len(examples)

    print()
    print("=" * 54)
    print("ALPHA FLOOR SWEEP")
    print("=" * 54)
    print(f"{'Floor':<12}{'MRR':>12}{'NDCG':>12}")
    print("-" * 54)

    for floor in floors:
        label = "V1/raw" if floor is None else f"{floor:.2f}"

        print(
            f"{label:<12}"
            f"{totals[floor]['mrr'] / n:>12.6f}"
            f"{totals[floor]['ndcg'] / n:>12.6f}"
        )

    print("=" * 54)


if __name__ == "__main__":
    main()
