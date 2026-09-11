from __future__ import annotations

import json
import random
import re
from pathlib import Path

from config_loader import create_neo4j_connection


SOURCE_ALPHA = Path("data/labeled/alpha_queries.jsonl")
OUTPUT = Path("data/labeled/alpha_train_balanced.jsonl")

SEED = 42


def clean_name(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def obviously_bad(name: str) -> bool:
    if not name:
        return True

    if len(name) < 5:
        return True

    bad_exact = {
        "Project",
        "Systems",
        "Proced",
        "System",
        "Tools",
    }

    if name in bad_exact:
        return True

    bad_fragments = (
        "Systems and Tools",
        "Project Charter",
        "Projects Owned",
        "Document",
    )

    if any(fragment in name for fragment in bad_fragments):
        return True

    return False


def good_employee(name: str) -> bool:
    if obviously_bad(name):
        return False

    parts = name.split()

    # Real employee names in this corpus are normally at least
    # first-name + surname.
    if len(parts) < 2:
        return False

    # Reject obvious tiny NER fragments such as "Ananya Ra".
    if any(len(part) < 3 for part in parts):
        return False

    return True


def good_project(name: str) -> bool:
    if obviously_bad(name):
        return False

    return name.startswith("Project ") and len(name.split()) >= 2


def good_software(name: str) -> bool:
    if obviously_bad(name):
        return False

    # Known clean software names already established in this corpus.
    known = {
        "AWS",
        "Bitwarden Enterprise",
        "Confluence",
        "Coupa",
        "GreytHR",
        "GreytHR Self-Service Portal",
        "Jira",
        "Meridian VPN Gateway",
        "Microsoft 365",
        "Okta SSO",
        "Salesforce",
        "SAP Concur",
        "ServiceNow",
        "Workday HRMS",
    }

    return name in known


def add_example(
    examples: list[dict],
    seen: set[str],
    query: str,
    relation: dict,
) -> None:
    query = query.strip()

    key = query.lower()

    if key in seen:
        return

    seen.add(key)

    examples.append(
        {
            "query": query,
            "label": "GRAPH_FAVORING",
            "source": "neo4j_relation",
            "relation": relation,
        }
    )


def main() -> None:
    connection = create_neo4j_connection()

    rows = connection.execute_read(
        """
        MATCH (s)-[r]->(o)
        RETURN
            labels(s)[0] AS subject_label,
            s.canonical_name AS subject,
            type(r) AS predicate,
            labels(o)[0] AS object_label,
            o.canonical_name AS object
        ORDER BY predicate, subject, object
        """
    )

    graph_examples = []
    seen = set()

    accepted_relations = 0

    # Build a clean employee-name vocabulary from the current graph.
    # Used only to repair obvious NER truncations in training examples.
    employee_rows = connection.execute_read(
        """
        MATCH (n:EMPLOYEE)
        WHERE n.canonical_name IS NOT NULL
        RETURN DISTINCT n.canonical_name AS name
        """
    )

    employee_names = sorted(
        {
            clean_name(row["name"])
            for row in employee_rows
            if clean_name(row["name"])
        },
        key=len,
        reverse=True,
    )

    def canonicalize_employee(name: str) -> str:
        clean = clean_name(name)

        if not clean:
            return clean

        clean_lower = clean.lower()

        # Exact full name already exists.
        if clean in employee_names:
            # Still prefer a clearly longer prefix completion if one exists.
            candidates = [
                candidate
                for candidate in employee_names
                if len(candidate) > len(clean)
                and candidate.lower().startswith(clean_lower)
                and len(candidate) - len(clean) <= 4
            ]

            if candidates:
                return min(candidates, key=len)

            return clean

        candidates = [
            candidate
            for candidate in employee_names
            if candidate.lower().startswith(clean_lower)
            and len(candidate) - len(clean) <= 4
        ]

        if candidates:
            return min(candidates, key=len)

        return clean

    for row in rows:
        subject = clean_name(row["subject"])
        obj = clean_name(row["object"])
        predicate = row["predicate"]

        if row["subject_label"] == "EMPLOYEE":
            subject = canonicalize_employee(subject)

        if row["object_label"] == "EMPLOYEE":
            obj = canonicalize_employee(obj)

        relation_info = {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
        }

        # --------------------------------------------------------------
        # MANAGES
        # manager -> employee
        # --------------------------------------------------------------
        if predicate == "MANAGES":
            if not good_employee(subject):
                continue

            if not good_employee(obj):
                continue

            accepted_relations += 1

            add_example(
                graph_examples,
                seen,
                f"Who manages {obj}?",
                relation_info,
            )

            add_example(
                graph_examples,
                seen,
                f"Who does {obj} report to?",
                relation_info,
            )

        # --------------------------------------------------------------
        # USES
        # project -> software
        # --------------------------------------------------------------
        elif predicate == "USES":
            if not good_project(subject):
                continue

            if not good_software(obj):
                continue

            accepted_relations += 1

            add_example(
                graph_examples,
                seen,
                f"What system does {subject} use?",
                relation_info,
            )

            add_example(
                graph_examples,
                seen,
                f"What software is used by {subject}?",
                relation_info,
            )

        # --------------------------------------------------------------
        # ASSIGNED_TO
        # employee -> project
        # --------------------------------------------------------------
        elif predicate == "ASSIGNED_TO":
            if not good_employee(subject):
                continue

            if not good_project(obj):
                continue

            accepted_relations += 1

            add_example(
                graph_examples,
                seen,
                f"Which project is {subject} assigned to?",
                relation_info,
            )

            add_example(
                graph_examples,
                seen,
                f"What project is {subject} working on?",
                relation_info,
            )

    # ------------------------------------------------------------------
    # Load the already-generated distant-supervision VECTOR examples.
    # Do not modify alpha_queries.jsonl.
    # ------------------------------------------------------------------
    vector_pool = []

    with SOURCE_ALPHA.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            row = json.loads(line)

            if row.get("label") != "VECTOR_FAVORING":
                continue

            query = str(row.get("query", "")).strip()

            if not query:
                continue

            vector_pool.append(
                {
                    "query": query,
                    "label": "VECTOR_FAVORING",
                    "source": "distant_supervision",
                }
            )

    random.seed(SEED)
    random.shuffle(vector_pool)

    # Balanced binary training set.
    vector_examples = vector_pool[:len(graph_examples)]

    combined = graph_examples + vector_examples
    random.shuffle(combined)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as handle:
        for row in combined:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=" * 80)
    print("BALANCED ALPHA TRAINING DATA")
    print("=" * 80)
    print("RAW GRAPH RELATIONSHIPS:", len(rows))
    print("ACCEPTED GRAPH RELATIONSHIPS:", accepted_relations)
    print("GRAPH_FAVORING EXAMPLES:", len(graph_examples))
    print("VECTOR_FAVORING EXAMPLES:", len(vector_examples))
    print("TOTAL TRAINING EXAMPLES:", len(combined))
    print("OUTPUT:", OUTPUT)

    print()
    print("SAMPLE GRAPH EXAMPLES:")
    for row in graph_examples[:25]:
        print(
            row["query"],
            "->",
            row["relation"]["subject"],
            row["relation"]["predicate"],
            row["relation"]["object"],
        )


if __name__ == "__main__":
    main()
