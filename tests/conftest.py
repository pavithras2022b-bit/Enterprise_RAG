"""
Root test configuration and fixture aggregation.

Imports pytest fixtures from individual module conftest files so that
running `pytest` from the project root automatically discovers all module
fixtures (e.g. tiny_bert_checkpoint_dir, tiny_relation_checkpoint_dir,
fake_neo4j_connection, etc.).
"""

pytest_plugins = [
    "ner.conftest",
    "relation_extraction.conftest",
    "reranking.conftest",
    "llm.conftest",
    "embeddings.conftest",
    "retrieval.conftest",
]
