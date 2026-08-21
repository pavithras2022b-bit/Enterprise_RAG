"""
Script to run the full test suite across all 14 modules.
"""

import sys
import pytest

def main():
    print("=" * 80)
    print("Running Enterprise RAG Full Test Suite (Modules 1-14)")
    print("=" * 80)
    
    test_files = [
        "tests/test_ingestion.py",
        "tests/test_preprocessing.py",
        "tests/test_chunking.py",
        "tests/test_ner.py",
        "tests/test_relation_extraction.py",
        "tests/test_knowledge_graph.py",
        "tests/test_embeddings.py",
        "tests/test_vector_db.py",
        "tests/test_hybrid_retrieval.py",
        "tests/test_adaptive_retrieval.py",
        "tests/test_reranking.py",
        "tests/test_llm.py",
        "tests/test_evaluation.py",
        "tests/test_api.py",
        "tests/test_end_to_end.py",
    ]
    
    ret_code = pytest.main(["-v"] + test_files)
    sys.exit(ret_code)

if __name__ == "__main__":
    main()
