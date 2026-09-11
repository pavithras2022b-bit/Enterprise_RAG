from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_loader import (
    create_embedding_model,
    create_vector_store,
    create_neo4j_connection,
)
from retrieval.entity_linker import EntityLinker
from retrieval.hybrid_controller import HybridRetrievalController


QUERY = "Who is the project lead for Project Helios?"
GOLD = "html_528f666cef2741f0_chunk_0000"


def main():
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

    print("=" * 80)
    print("PROJECT HELIOS ALPHA SWEEP")
    print("=" * 80)
    print(f"Query: {QUERY}")
    print(f"Gold : {GOLD}")
    print()

    for alpha in [
        0.0,
        0.1,
        0.2,
        0.3,
        0.3837,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]:

        candidates = controller.retrieve(
            query=QUERY,
            alpha=alpha,
            k_vector=5,
            k_graph=5,
        )

        ids = [c.candidate_id for c in candidates[:5]]

        try:
            gold_rank = ids.index(GOLD) + 1
        except ValueError:
            gold_rank = None

        print("-" * 80)
        print(f"alpha={alpha:.4f} | gold rank={gold_rank}")

        for rank, candidate in enumerate(candidates[:5], start=1):

            marker = " <-- GOLD" if candidate.candidate_id == GOLD else ""

            print(
                f"{rank}. {candidate.candidate_id}"
                f" | source={candidate.source}"
                f" | vector={candidate.vector_score}"
                f" | graph={candidate.graph_score}"
                f" | fused={candidate.fused_score:.4f}"
                f"{marker}"
            )

    print()
    print("=" * 80)
    print("ALPHA SWEEP COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
