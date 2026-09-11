from __future__ import annotations

import json
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
from retrieval.entity_linker import EntityLinker
from retrieval.hybrid_controller import HybridRetrievalController


V1_GENERATION_FILE = (
    PROJECT_ROOT / "outputs" / "llm_generation_evaluation.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "retrieval_v1_vs_guarded_v2_generation_queries.json"
)

MODEL_NAME = "qwen2.5-3b-lora-ragqa"
NUM_QUERIES = 15
ALPHA_FLOOR = 0.50

# Production generation uses top_k=5 and retrieves 2x from each branch.
K_VECTOR = 10
K_GRAPH = 10


def load_frozen_generation_queries():
    with open(V1_GENERATION_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = [
        item
        for item in data["per_query_results"]
        if item["model"] == MODEL_NAME
    ]

    results.sort(key=lambda x: x["query_index"])

    if len(results) < NUM_QUERIES:
        raise RuntimeError(
            f"Expected at least {NUM_QUERIES} Qwen V1 results, "
            f"found {len(results)}."
        )

    return results[:NUM_QUERIES]


def candidate_ids(candidates):
    return [candidate.candidate_id for candidate in candidates]


def main():
    print("=" * 80)
    print("V1 vs GUARDED ADAPTIVE V2 — 15 GENERATION QUERIES")
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

    print("[5/6] Building HybridRetrievalController...")
    hybrid_controller = HybridRetrievalController(
        vector_store=vector_store,
        graph_connection=neo4j_connection,
        embedding_model=embedding_model,
        entity_linker=entity_linker,
    )

    print("[6/6] Loading learned alpha classifier...")
    alpha_classifier = create_alpha_classifier()

    adaptive_engine = AdaptiveFusionEngine(
        controller=hybrid_controller,
        embedding_model=embedding_model,
        entity_linker=entity_linker,
        classifier=alpha_classifier,
    )

    frozen_results = load_frozen_generation_queries()

    alpha_changed_count = 0
    top5_changed_count = 0
    top10_changed_count = 0

    per_query_results = []

    print()
    print("Beginning controlled comparison...")
    print()

    for index, frozen in enumerate(frozen_results, start=1):
        query = frozen["query"]

        # --------------------------------------------------------------
        # 1. Compute original learned alpha.
        # --------------------------------------------------------------
        alpha_v1 = float(adaptive_engine.compute_alpha(query))

        # --------------------------------------------------------------
        # 2. Apply ONLY the experimental V2 guard.
        # --------------------------------------------------------------
        alpha_v2 = max(alpha_v1, ALPHA_FLOOR)

        if alpha_v2 != alpha_v1:
            alpha_changed_count += 1

        # --------------------------------------------------------------
        # 3. Run retrieval branches ONCE.
        #    Both V1 and V2 therefore see identical raw vector/graph hits.
        # --------------------------------------------------------------
        entities = entity_linker.link(query)

        vector_hits = hybrid_controller.run_vector_retrieval(
            query,
            k=K_VECTOR,
        )

        graph_hits = hybrid_controller.run_graph_retrieval(
            entities,
            k=K_GRAPH,
        )

        # --------------------------------------------------------------
        # 4. Same fusion implementation; ONLY alpha differs.
        # --------------------------------------------------------------
        fused_v1 = HybridRetrievalController.static_fuse(
            vector_hits,
            graph_hits,
            alpha=alpha_v1,
        )

        fused_v2 = HybridRetrievalController.static_fuse(
            vector_hits,
            graph_hits,
            alpha=alpha_v2,
        )

        ids_v1 = candidate_ids(fused_v1)
        ids_v2 = candidate_ids(fused_v2)

        top5_v1 = ids_v1[:5]
        top5_v2 = ids_v2[:5]

        top10_v1 = ids_v1[:10]
        top10_v2 = ids_v2[:10]

        top5_changed = top5_v1 != top5_v2
        top10_changed = top10_v1 != top10_v2

        if top5_changed:
            top5_changed_count += 1

        if top10_changed:
            top10_changed_count += 1

        print("=" * 80)
        print(f"QUERY {index:02d}/{NUM_QUERIES}")
        print(query)
        print("-" * 80)
        print(f"V1 alpha       : {alpha_v1:.6f}")
        print(f"V2 alpha       : {alpha_v2:.6f}")
        print(
            f"Alpha changed  : "
            f"{'YES' if alpha_v1 != alpha_v2 else 'NO'}"
        )
        print(
            f"Top-5 changed  : "
            f"{'YES' if top5_changed else 'NO'}"
        )
        print(
            f"Top-10 changed : "
            f"{'YES' if top10_changed else 'NO'}"
        )

        if top5_changed:
            print()
            print("V1 Top-5:")
            for rank, cid in enumerate(top5_v1, start=1):
                print(f"  {rank}. {cid}")

            print()
            print("V2 Top-5:")
            for rank, cid in enumerate(top5_v2, start=1):
                print(f"  {rank}. {cid}")

        per_query_results.append(
            {
                "query_index": index,
                "query": query,
                "alpha_v1": alpha_v1,
                "alpha_v2_guarded": alpha_v2,
                "alpha_floor": ALPHA_FLOOR,
                "alpha_changed": alpha_v1 != alpha_v2,
                "v1_top5_ids": top5_v1,
                "v2_top5_ids": top5_v2,
                "top5_changed": top5_changed,
                "v1_top10_ids": top10_v1,
                "v2_top10_ids": top10_v2,
                "top10_changed": top10_changed,
            }
        )

    summary = {
        "queries_evaluated": NUM_QUERIES,
        "alpha_floor": ALPHA_FLOOR,
        "k_vector": K_VECTOR,
        "k_graph": K_GRAPH,
        "alpha_changed_queries": alpha_changed_count,
        "top5_changed_queries": top5_changed_count,
        "top10_changed_queries": top10_changed_count,
    }

    output = {
        "evaluation_type": (
            "controlled_v1_vs_guarded_v2_retrieval_on_generation_queries"
        ),
        "v1_generation_source": str(V1_GENERATION_FILE),
        "experimental_change": (
            "Only the learned adaptive vector weight is changed: "
            "alpha_v2 = max(alpha_v1, 0.50). "
            "Vector hits, graph hits, fusion implementation, query set, "
            "and retrieval depths are identical."
        ),
        "summary": summary,
        "per_query_results": per_query_results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Queries tested         : {NUM_QUERIES}")
    print(f"Alpha changed          : {alpha_changed_count}/{NUM_QUERIES}")
    print(f"Top-5 evidence changed : {top5_changed_count}/{NUM_QUERIES}")
    print(f"Top-10 list changed    : {top10_changed_count}/{NUM_QUERIES}")
    print()
    print(f"Saved: {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
