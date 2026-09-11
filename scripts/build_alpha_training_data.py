from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from config_loader import (
    create_embedding_model,
    create_neo4j_connection,
    create_vector_store,
)
from retrieval.entity_linker import EntityLinker
from retrieval.hybrid_controller import HybridRetrievalController


INPUT_PATH = Path("data/labeled/lora_candidates.jsonl")
OUTPUT_PATH = Path("data/labeled/alpha_queries.jsonl")

K_VECTOR = 10
K_GRAPH = 10


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def meaningful_tokens(text: str) -> set[str]:
    stop = {
        "the", "a", "an", "is", "are", "was", "were",
        "to", "of", "and", "or", "in", "on", "for",
        "who", "what", "which", "how", "does", "do",
        "with", "by", "from", "this", "that",
        "department", "policy",
    }

    return {
        token
        for token in normalize(text).split()
        if token not in stop and len(token) >= 3
    }


def graph_supports_answer(answer: str, graph_facts) -> bool:
    """
    Conservative lexical support test.

    A single graph fact must substantially support the gold answer.
    Evidence from unrelated graph facts is not pooled together.
    """
    answer_tokens = meaningful_tokens(answer)

    if not answer_tokens:
        return False

    for fact in graph_facts:
        fact_text = (
            f"{fact.subject} {fact.predicate} "
            f"{fact.object} {fact.text}"
        )

        fact_tokens = meaningful_tokens(fact_text)
        overlap = answer_tokens & fact_tokens

        if len(answer_tokens) >= 3:
            coverage = len(overlap) / len(answer_tokens)

            if len(overlap) >= 2 and coverage >= 0.40:
                return True
        elif len(overlap) >= 1:
            return True

    return False


def main() -> None:
    print("Loading retrieval components...")

    connection = create_neo4j_connection()
    embedding_model = create_embedding_model()
    vector_store = create_vector_store()

    linker = EntityLinker(
        connection,
        similarity_threshold=80.0,
        refresh_on_init=True,
    )

    controller = HybridRetrievalController(
        vector_store=vector_store,
        graph_connection=connection,
        embedding_model=embedding_model,
        entity_linker=linker,
    )

    rows = []

    with INPUT_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            row = json.loads(line)

            if row.get("status") != "approved":
                continue

            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            gold_chunk_id = str(row.get("candidate_id", "")).strip()

            if not question or not answer or not gold_chunk_id:
                continue

            rows.append(
                {
                    "question": question,
                    "answer": answer,
                    "gold_chunk_id": gold_chunk_id,
                }
            )

    # Deduplicate by question.
    unique_rows = {}
    for row in rows:
        unique_rows.setdefault(row["question"], row)

    rows = list(unique_rows.values())

    print("Approved unique questions:", len(rows))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()

    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for index, row in enumerate(rows, start=1):
            question = row["question"]
            answer = row["answer"]
            gold_chunk_id = row["gold_chunk_id"]

            print(f"[{index}/{len(rows)}] {question}")

            # ----------------------------------------------------------
            # Vector branch
            # ----------------------------------------------------------
            vector_hits = controller.run_vector_retrieval(
                question,
                k=K_VECTOR,
            )

            vector_ids = [hit.chunk_id for hit in vector_hits]
            vector_support = gold_chunk_id in vector_ids

            vector_rank = None
            if vector_support:
                vector_rank = vector_ids.index(gold_chunk_id) + 1

            # ----------------------------------------------------------
            # Graph branch
            # ----------------------------------------------------------
            linked_entities = linker.link(question)

            graph_hits = controller.run_graph_retrieval(
                linked_entities,
                k=K_GRAPH,
            )

            graph_support = graph_supports_answer(
                answer,
                graph_hits,
            )

            # ----------------------------------------------------------
            # Label
            # ----------------------------------------------------------
            if vector_support and graph_support:
                label = "MIXED"
            elif graph_support:
                label = "GRAPH_FAVORING"
            elif vector_support:
                label = "VECTOR_FAVORING"
            else:
                label = "UNRESOLVED"

            counts[label] += 1

            output_row = {
                "query": question,
                "answer": answer,
                "gold_chunk_id": gold_chunk_id,
                "label": label,
                "vector_support": vector_support,
                "vector_rank": vector_rank,
                "graph_support": graph_support,
                "linked_entities": [
                    ref.canonical_name
                    for ref in linked_entities
                ],
                "graph_facts": [
                    {
                        "subject": fact.subject,
                        "predicate": fact.predicate,
                        "object": fact.object,
                        "text": fact.text,
                        "hop_count": fact.hop_count,
                    }
                    for fact in graph_hits
                ],
            }

            out.write(
                json.dumps(
                    output_row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print("=" * 70)
    print("DISTANT SUPERVISION COMPLETE")
    print("=" * 70)

    for label, count in counts.items():
        print(f"{label}: {count}")

    print("TOTAL:", sum(counts.values()))
    print("OUTPUT:", OUTPUT_PATH)


if __name__ == "__main__":
    main()