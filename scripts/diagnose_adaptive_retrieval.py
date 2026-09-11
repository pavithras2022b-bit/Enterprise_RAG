from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_retrieval.fusion import AdaptiveFusionEngine
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


def first_gold_rank(retrieved_ids, gold_ids, k=5):
    gold = set(gold_ids)

    for rank, candidate_id in enumerate(retrieved_ids[:k], start=1):
        if candidate_id in gold:
            return rank

    return None


def main():
    print("=" * 80)
    print("LIGHTWEIGHT LEARNED ADAPTIVE RETRIEVAL DIAGNOSTIC")
    print("=" * 80)

    print("[1/6] Loading MiniLM embedding model...")
    embedding_model = create_embedding_model()

    print("[2/6] Opening ChromaDB vector store...")
    vector_store = create_vector_store()

    print("[3/6] Connecting to Neo4j...")
    neo4j_connection = create_neo4j_connection()

    print("[4/6] Building EntityLinker...")
    entity_linker = EntityLinker(
        neo4j_connection,
        refresh_on_init=True,
    )

    print("[5/6] Building hybrid retrieval controller...")
    hybrid_controller = HybridRetrievalController(
        vector_store=vector_store,
        graph_connection=neo4j_connection,
        embedding_model=embedding_model,
        entity_linker=entity_linker,
    )

    print("[6/6] Loading existing learned alpha classifier...")
    alpha_classifier = create_alpha_classifier()

    adaptive_engine = AdaptiveFusionEngine(
        controller=hybrid_controller,
        embedding_model=embedding_model,
        entity_linker=entity_linker,
        classifier=alpha_classifier,
    )

    val_file = PROJECT_ROOT / "data" / "labeled" / "lora_val.jsonl"
    examples = parse_val_examples(val_file)

    print()
    print(f"Validation examples: {len(examples)}")
    print("Beginning per-query comparison...")
    print()

    better_cases = []
    equal_cases = 0
    worse_cases = []

    vector_mrr_total = 0.0
    adaptive_mrr_total = 0.0
    vector_ndcg_total = 0.0
    adaptive_ndcg_total = 0.0

    for i, item in enumerate(examples, start=1):

        vector_candidates = hybrid_controller.retrieve(
            query=item.query,
            alpha=1.0,
            k_vector=5,
            k_graph=5,
        )

        alpha_used, adaptive_candidates = adaptive_engine.retrieve(
            query=item.query,
            k_vector=5,
            k_graph=5,
        )

        vector_ids = [c.candidate_id for c in vector_candidates]
        adaptive_ids = [c.candidate_id for c in adaptive_candidates]

        vector_mrr = calculate_mrr(
            vector_ids,
            item.ground_truth_chunk_ids,
            k=5,
        )

        adaptive_mrr = calculate_mrr(
            adaptive_ids,
            item.ground_truth_chunk_ids,
            k=5,
        )

        vector_ndcg = calculate_ndcg(
            vector_ids,
            item.ground_truth_chunk_ids,
            k=5,
        )

        adaptive_ndcg = calculate_ndcg(
            adaptive_ids,
            item.ground_truth_chunk_ids,
            k=5,
        )

        vector_mrr_total += vector_mrr
        adaptive_mrr_total += adaptive_mrr
        vector_ndcg_total += vector_ndcg
        adaptive_ndcg_total += adaptive_ndcg

        result = {
            "index": i,
            "query": item.query,
            "gold": list(item.ground_truth_chunk_ids),
            "alpha": float(alpha_used),
            "vector_rank": first_gold_rank(
                vector_ids,
                item.ground_truth_chunk_ids,
            ),
            "adaptive_rank": first_gold_rank(
                adaptive_ids,
                item.ground_truth_chunk_ids,
            ),
            "vector_mrr": vector_mrr,
            "adaptive_mrr": adaptive_mrr,
            "vector_ndcg": vector_ndcg,
            "adaptive_ndcg": adaptive_ndcg,
            "vector_ids": vector_ids[:5],
            "adaptive_ids": adaptive_ids[:5],
        }

        if adaptive_mrr < vector_mrr or adaptive_ndcg < vector_ndcg:
            worse_cases.append(result)
        elif adaptive_mrr > vector_mrr or adaptive_ndcg > vector_ndcg:
            better_cases.append(result)
        else:
            equal_cases += 1

        print(
            f"[{i:02d}/{len(examples):02d}] "
            f"alpha={float(alpha_used):.4f} | "
            f"vector rank={result['vector_rank']} | "
            f"adaptive rank={result['adaptive_rank']}"
        )

    n = len(examples)

    print()
    print("=" * 80)
    print("AGGREGATE SANITY CHECK")
    print("=" * 80)

    if n:
        print(f"Vector MRR     : {vector_mrr_total / n:.6f}")
        print(f"Adaptive MRR   : {adaptive_mrr_total / n:.6f}")
        print(f"Vector NDCG    : {vector_ndcg_total / n:.6f}")
        print(f"Adaptive NDCG  : {adaptive_ndcg_total / n:.6f}")

    print()
    print(f"Learned BETTER than Vector : {len(better_cases)}")
    print(f"Learned EQUAL to Vector    : {equal_cases}")
    print(f"Learned WORSE than Vector  : {len(worse_cases)}")

    print()
    print("=" * 80)
    print("QUERIES WHERE LEARNED ADAPTIVE LOSES")
    print("=" * 80)

    for r in worse_cases:
        print()
        print("-" * 80)
        print(f"Query #{r['index']}")
        print(f"Question      : {r['query']}")
        print(f"Gold chunk(s) : {r['gold']}")
        print(f"Alpha used    : {r['alpha']:.4f}")

        print()
        print(f"Vector rank   : {r['vector_rank']}")
        print(f"Adaptive rank : {r['adaptive_rank']}")

        print()
        print(f"Vector MRR    : {r['vector_mrr']:.4f}")
        print(f"Adaptive MRR  : {r['adaptive_mrr']:.4f}")
        print(
            f"Delta MRR     : "
            f"{r['adaptive_mrr'] - r['vector_mrr']:+.4f}"
        )

        print()
        print(f"Vector NDCG   : {r['vector_ndcg']:.4f}")
        print(f"Adaptive NDCG : {r['adaptive_ndcg']:.4f}")
        print(
            f"Delta NDCG    : "
            f"{r['adaptive_ndcg'] - r['vector_ndcg']:+.4f}"
        )

        print()
        print("Vector Top-5:")

        for rank, cid in enumerate(r["vector_ids"], start=1):
            marker = " <-- GOLD" if cid in r["gold"] else ""
            print(f"  {rank}. {cid}{marker}")

        print("Adaptive Top-5:")

        for rank, cid in enumerate(r["adaptive_ids"], start=1):
            marker = " <-- GOLD" if cid in r["gold"] else ""
            print(f"  {rank}. {cid}{marker}")

    print()
    print("=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
