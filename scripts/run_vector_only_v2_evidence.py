from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_loader import (
    create_embedding_model,
    create_neo4j_connection,
    create_vector_store,
)

from retrieval.entity_linker import EntityLinker
from retrieval.hybrid_controller import HybridRetrievalController
from reranking.model import CrossEncoderReranker


V1_GENERATION_FILE = (
    PROJECT_ROOT / "outputs" / "llm_generation_evaluation.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "outputs" / "vector_only_v2_evidence.json"
)

MODEL_NAME = "qwen2.5-3b-lora-ragqa"

NUM_QUERIES = 15
TOP_K = 5
K_VECTOR = TOP_K * 2

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


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
            f"Expected at least {NUM_QUERIES} frozen Qwen V1 results, "
            f"found {len(results)}."
        )

    return results[:NUM_QUERIES]


def main():
    print("=" * 80)
    print("VECTOR-ONLY ABLATION V2 — EVIDENCE GENERATION")
    print("=" * 80)

    print("[1/6] Loading MiniLM embedding model...")
    embedding_model = create_embedding_model()

    print("[2/6] Opening ChromaDB vector store...")
    vector_store = create_vector_store()

    # HybridRetrievalController requires these dependencies even though
    # V2 will intentionally execute only its vector branch.
    print("[3/6] Connecting to Neo4j...")
    neo4j_connection = create_neo4j_connection()

    print("[4/6] Building EntityLinker...")
    entity_linker = EntityLinker(
        neo4j_connection,
        refresh_on_init=True,
    )

    print("[5/6] Building HybridRetrievalController...")
    controller = HybridRetrievalController(
        vector_store=vector_store,
        graph_connection=neo4j_connection,
        embedding_model=embedding_model,
        entity_linker=entity_linker,
    )

    print("[6/6] Loading cross-encoder reranker...")
    reranker = CrossEncoderReranker(
        RERANKER_MODEL,
        device="cpu",
    )

    frozen_v1 = load_frozen_generation_queries()

    changed_count = 0
    identical_count = 0
    overlap_total = 0.0

    per_query_results = []

    print()
    print("Beginning V1 Adaptive-Hybrid vs V2 Vector-Only evidence comparison...")
    print()

    for index, v1 in enumerate(frozen_v1, start=1):
        query = v1["query"]

        v1_ids = list(v1["valid_source_ids"])

        # --------------------------------------------------------------
        # V2 STEP 1:
        # Vector retrieval ONLY.
        # Same depth as production pipeline: top_k * 2 = 10.
        # --------------------------------------------------------------
        vector_hits = controller.run_vector_retrieval(
            query,
            k=K_VECTOR,
        )

        # --------------------------------------------------------------
        # V2 STEP 2:
        # Convert vector hits into the same FusedCandidate structure
        # used by the production reranker.
        #
        # Empty graph list + alpha=1.0 means pure vector evidence.
        # --------------------------------------------------------------
        vector_candidates = HybridRetrievalController.static_fuse(
            vector_hits,
            [],
            alpha=1.0,
        )

        # --------------------------------------------------------------
        # V2 STEP 3:
        # Same cross-encoder reranker as V1.
        # --------------------------------------------------------------
        ranked_v2 = reranker.rerank(
            query,
            vector_candidates,
            top_n=TOP_K,
        )

        v2_ids = [
            candidate.candidate_id
            for candidate in ranked_v2
        ]

        exact_same = v1_ids == v2_ids

        if exact_same:
            identical_count += 1
        else:
            changed_count += 1

        shared_ids = set(v1_ids) & set(v2_ids)

        overlap_ratio = (
            len(shared_ids) / TOP_K
            if TOP_K
            else 0.0
        )

        overlap_total += overlap_ratio

        print("=" * 80)
        print(f"QUERY {index:02d}/{NUM_QUERIES}")
        print(query)
        print("-" * 80)

        print(
            f"Evidence changed : "
            f"{'NO' if exact_same else 'YES'}"
        )

        print(
            f"Top-5 overlap    : "
            f"{len(shared_ids)}/{TOP_K} "
            f"({overlap_ratio * 100:.1f}%)"
        )

        if not exact_same:
            print()
            print("V1 Adaptive-Hybrid Top-5:")
            for rank, cid in enumerate(v1_ids, start=1):
                print(f"  {rank}. {cid}")

            print()
            print("V2 Vector-Only Top-5:")
            for rank, cid in enumerate(v2_ids, start=1):
                print(f"  {rank}. {cid}")

        per_query_results.append(
            {
                "query_index": index,
                "query": query,

                "v1_architecture": "adaptive_hybrid",
                "v2_architecture": "vector_only",

                "v1_top5_ids": v1_ids,
                "v2_top5_ids": v2_ids,

                "exact_order_match": exact_same,

                "shared_top5_count": len(shared_ids),
                "top5_overlap_ratio": overlap_ratio,

                "v2_evidence": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "text": candidate.text,
                        "rerank_score": float(candidate.rerank_score),
                        "rank": candidate.rank,
                    }
                    for candidate in ranked_v2
                ],
            }
        )

    average_overlap = overlap_total / NUM_QUERIES

    summary = {
        "queries_evaluated": NUM_QUERIES,

        "v1_architecture": (
            "learned adaptive hybrid vector + knowledge graph retrieval"
        ),

        "v2_architecture": (
            "vector-only retrieval followed by the same cross-encoder reranker"
        ),

        "top_k": TOP_K,
        "initial_vector_depth": K_VECTOR,

        "identical_top5_order_queries": identical_count,
        "changed_top5_queries": changed_count,

        "average_top5_set_overlap": average_overlap,
    }

    output = {
        "evaluation_type": "vector_only_v2_evidence_ablation",

        "experimental_change": (
            "V1 adaptive hybrid retrieval was replaced by vector-only "
            "retrieval. The embedding model, vector database, retrieval "
            "depth, cross-encoder reranker, top-k evidence size, query set, "
            "generation models, prompting, citation validation, and "
            "generation evaluation methodology remain unchanged."
        ),

        "summary": summary,
        "per_query_results": per_query_results,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 80)
    print("VECTOR-ONLY V2 EVIDENCE SUMMARY")
    print("=" * 80)

    print(
        f"Queries evaluated       : "
        f"{NUM_QUERIES}"
    )

    print(
        f"Evidence order changed  : "
        f"{changed_count}/{NUM_QUERIES}"
    )

    print(
        f"Identical Top-5 order   : "
        f"{identical_count}/{NUM_QUERIES}"
    )

    print(
        f"Average Top-5 overlap   : "
        f"{average_overlap:.4f} "
        f"({average_overlap * 100:.2f}%)"
    )

    print()
    print(f"Saved: {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
