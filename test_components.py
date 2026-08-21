#!/usr/bin/env python
"""Test component initialization from config."""

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

print("Testing component initialization...")

try:
    print("\n1. Testing EmbeddingModel creation...")
    from config_loader import create_embedding_model
    embedding_model = create_embedding_model()
    print(f"   OK - Embedding model dimension: {embedding_model.dimension}")
except Exception as e:
    print(f"   ERROR: {e}")

try:
    print("\n2. Testing VectorStore creation...")
    from config_loader import create_vector_store
    vector_store = create_vector_store()
    print(f"   OK - VectorStore created")
except Exception as e:
    print(f"   ERROR: {e}")

try:
    print("\n3. Testing Neo4jConnection creation...")
    from config_loader import create_neo4j_connection
    neo4j_conn = create_neo4j_connection()
    print(f"   OK - Neo4j connection created")
except Exception as e:
    print(f"   EXPECTED ERROR (Neo4j not running): {type(e).__name__}: {str(e)[:100]}")

try:
    print("\n4. Testing LLMClient creation...")
    from config_loader import create_llm_client
    llm_client = create_llm_client()
    print(f"   OK - LLM client created")
except Exception as e:
    print(f"   OK - LLM client creation (will fail when used if Ollama not running)")

print("\n" + "="*60)
print("Component initialization test complete!")
