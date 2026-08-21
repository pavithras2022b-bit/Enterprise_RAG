#!/usr/bin/env python
"""Quick test of config loader."""

from config_loader import get_embeddings_config, get_llm_config, get_vector_db_config

print("Config Load Test:")
emb = get_embeddings_config()
print(f"Embeddings checkpoint: {emb.get('model', {}).get('checkpoint')}")
llm = get_llm_config()
print(f"LLM model: {llm.get('client', {}).get('model_name')}")
vec = get_vector_db_config()
print(f"Vector DB collection: {vec.get('chroma', {}).get('collection_name')}")
print("\nAll configs loaded successfully!")

