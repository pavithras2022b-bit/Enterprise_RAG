# Enterprise Retrieval-Augmented Generation (RAG) System

A production-grade, enterprise-ready Hybrid Knowledge Graph & Vector Retrieval-Augmented Generation system.

---

## 🏗️ FOLDER STRUCTURE

```
Enterprise-Retrieval-Augmented-Generation/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
├── Makefile
│
├── configs/
│   ├── ingestion.yaml
│   ├── chunking.yaml
│   ├── ner.yaml
│   ├── relation_extraction.yaml
│   ├── knowledge_graph.yaml
│   ├── embeddings.yaml
│   ├── vector_db.yaml
│   ├── retrieval.yaml
│   ├── adaptive_retrieval.yaml
│   ├── reranking.yaml
│   ├── llm.yaml
│   └── evaluation.yaml
│
├── data/
│   ├── raw/                        # untouched source documents
│   ├── cleaned/                    # post-Module-2 text + metadata
│   ├── chunks/                     # post-Module-3 chunk JSONL
│   ├── labeled/
│   │   ├── ner_labels.jsonl
│   │   ├── relation_labels.jsonl
│   │   ├── query_alpha_labels.jsonl
│   │   └── eval_query_set.jsonl
│   └── schema/
│       └── entity_relationship_schema.yaml   # Section 4.4's fixed node/edge types
│
├── ingestion/
│   ├── __init__.py
│   ├── loaders.py                  # extract_pdf/docx/html
│   ├── dedupe.py                    # checksum-based dedup
│   └── schema.py                    # RawDocument dataclass
│
├── preprocessing/
│   ├── __init__.py
│   ├── cleaner.py                   # boilerplate strip, encoding fix
│   ├── metadata_tagger.py
│   └── schema.py                    # CleanDocument dataclass
│
├── chunking/
│   ├── __init__.py
│   ├── strategies.py                # fixed-size, sentence-aware, structure-aware
│   └── schema.py                    # Chunk dataclass
│
├── ner/
│   ├── __init__.py
│   ├── model.py                     # NERModel wrapper
│   ├── resolver.py                  # entity dedup/resolution
│   ├── train.py                     # fine-tuning entrypoint
│   └── schema.py                    # Entity dataclass
│
├── relation_extraction/
│   ├── __init__.py
│   ├── model.py                     # RelationClassifier
│   ├── rules.py                     # rule-based/pattern extractor
│   ├── train.py
│   └── schema.py                    # Relation dataclass
│
├── knowledge_graph/
│   ├── __init__.py
│   ├── builder.py                   # Neo4jGraphBuilder
│   ├── connection.py                # driver wrapper
│   ├── schema.py                    # GraphSchema (allowed labels/types)
│   └── cypher_queries.py            # reusable named Cypher queries
│
├── embeddings/
│   ├── __init__.py
│   ├── model.py                     # EmbeddingModel wrapper
│   ├── train.py                     # contrastive fine-tuning
│   └── schema.py                    # EmbeddedChunk dataclass
│
├── vector_db/
│   ├── __init__.py
│   ├── chroma_store.py              # primary VectorStore
│   ├── faiss_index.py               # benchmark-only comparison index
│   └── schema.py                    # VectorSearchResult dataclass
│
├── retrieval/
│   ├── __init__.py
│   ├── hybrid_controller.py         # parallel vector+graph retrieval, static fuse
│   ├── entity_linker.py             # query text -> graph node refs
│   └── schema.py                    # FusedCandidate dataclass
│
├── adaptive_retrieval/
│   ├── __init__.py
│   ├── classifier.py                # AlphaClassifier (base + learned)
│   ├── rules.py                     # RuleBasedAlphaClassifier
│   ├── fusion.py                    # AdaptiveFusionEngine (Phase 3's equation)
│   └── train.py                     # classifier training entrypoint
│
├── reranking/
│   ├── __init__.py
│   ├── cross_encoder.py             # CrossEncoderReranker
│   ├── train.py                     # fine-tuning entrypoint
│   └── schema.py                    # RankedCandidate dataclass
│
├── llm/
│   ├── __init__.py
│   ├── client.py                    # Ollama/vLLM wrapper & answer generation
│   ├── prompt_builder.py            # query+evidence -> prompt template
│   ├── lora_manager.py              # LoRA adapter load/swap
│   └── schema.py                    # GeneratedAnswer dataclass
│
├── evaluation/
│   ├── __init__.py
│   ├── retrieval_metrics.py         # Hit Rate, MRR, NDCG
│   ├── generation_metrics.py        # Faithfulness, Answer Relevance, Citation Precision
│   ├── ablation_runner.py           # runs comparative ablation studies
│   └── significance.py              # paired t-test / Wilcoxon
│
├── api/
│   ├── __init__.py
│   ├── main.py                      # FastAPI app entrypoint
│   ├── routes.py                    # /health, /api/v1/ingest, /api/v1/query endpoints
│   ├── pipeline.py                  # End-to-End Enterprise RAG Pipeline orchestrator
│   └── schema.py                    # Pydantic request/response schemas
│
└── tests/
    ├── conftest.py                  # Global test fixtures
    ├── test_ingestion.py
    ├── test_preprocessing.py
    ├── test_chunking.py
    ├── test_ner.py
    ├── test_relation_extraction.py
    ├── test_knowledge_graph.py
    ├── test_embeddings.py
    ├── test_vector_db.py
    ├── test_hybrid_retrieval.py
    ├── test_adaptive_retrieval.py
    ├── test_reranking.py
    ├── test_llm.py
    ├── test_evaluation.py
    ├── test_api.py
    └── test_end_to_end.py
```

---

## ⚡ Quick Start

### 1. Installation & Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running Unit & Integration Tests

```bash
# Run all tests across Modules 1 to 14
pytest tests/ -v
```

### 3. Launching the REST API

```bash
# Start FastAPI application with uvicorn
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. API Endpoints

- **`GET /health`**: Health check and system status.
- **`POST /api/v1/ingest`**: Ingest documents into vector store and knowledge graph.
- **`POST /api/v1/query`**: End-to-end query answering using hybrid adaptive retrieval & generation.