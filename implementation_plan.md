# Implementation Plan - Enterprise RAG Project End-to-End

## Overview
This implementation plan outlines the steps to take the partially implemented **Enterprise Retrieval-Augmented Generation (RAG)** capstone project to a complete, fully functional, tested end-to-end system without changing existing working code or module contracts.

---w

## Repository Audit & Module Status Map

| Module | Purpose | Status | Details / Issues |
|---|---|---|---|
| **Module 1: Ingestion** | Document loading (`loaders.py`, `dedupe.py`, `connectors.py`) | **COMPLETE** | 100% tests passing (`test_ingestion.py`) |
| **Module 2: Preprocessing** | Document cleaning & metadata tagging | **COMPLETE** | 100% tests passing (`test_preprocessing.py`) |
| **Module 3: Chunking** | Fixed, sentence-aware, structure-aware chunking | **COMPLETE** | 100% tests passing (`test_chunking.py`) |
| **Module 4: NER** | Named Entity Recognition & Entity Resolution | **MOSTLY COMPLETE** | 27 tests passing; fixture scope error when running from root |
| **Module 5: Relation Extraction** | Closed relation schema & model/rule extraction | **MOSTLY COMPLETE** | 25 tests passing; fixture scope error when running from root |
| **Module 6: Knowledge Graph** | Neo4j Graph Builder & Cypher queries | **BROKEN / PARTIALLY COMPLETE** | `knowledge_graph/schema.py` missing `GraphSchema`, `GraphBuildReport`, `GraphSchemaError` |
| **Module 7: Embeddings** | SentenceTransformer wrapper & contrastive fine-tuning | **BLOCKED BY DEPENDENCY** | Missing `sentence-transformers` in `.venv` |
| **Module 8: Vector DB** | ChromaDB store & FAISS index comparison | **BLOCKED BY DEPENDENCY** | Missing `chromadb` in `.venv` |
| **Module 9: Retrieval** | Hybrid controller (Vector + KG) & Entity Linker | **BLOCKED BY DEPENDENCY** | Code present, blocked by missing dependencies in `.venv` |
| **Module 10: Adaptive Retrieval** | Query classifier ($\alpha(q)$) & fusion engine | **BLOCKED BY DEPENDENCY** | Missing `scikit-learn` in `.venv` |
| **Module 11: Reranking** | CrossEncoder reranker & context assembler | **BLOCKED BY DEPENDENCY** | Missing `sentence-transformers` in `.venv` |
| **Module 12: LLM Integration** | Client wrapper, prompt builder, LoRA manager | **BLOCKED BY DEPENDENCY** | Code present, missing `peft` in `.venv` |
| **Module 13: Evaluation** | Retrieval & generation metrics, ablation runner | **PARTIALLY COMPLETE** | Stubs in place in `evaluation/`, requires metric implementations & tests |
| **Module 14: API & Pipeline** | FastAPI server & end-to-end orchestration flow | **PARTIALLY COMPLETE** | Stubs in place in `api/`, requires `pipeline.py` & `routes.py` |

---

## User Review Required

> [!IMPORTANT]
> - Dependencies (`sentence-transformers`, `chromadb`, `scikit-learn`, `peft`) need to be installed into `.venv` using standard pip.
> - `tests/conftest.py` will be created to import test fixtures from module `conftest.py` files so `pytest -v` from root works seamlessly.
> - `knowledge_graph/schema.py` will be augmented to add `GraphSchema`, `GraphBuildReport`, `GraphSchemaError` while preserving `Relation` and `RelationType`.

---

## Proposed Changes

### Phase 1: Environment & Dependencies Fix
- Install `scikit-learn`, `sentence-transformers`, `chromadb`, `peft` into `.venv`.
- Create `tests/conftest.py` to aggregate fixtures (`tiny_bert_checkpoint_dir`, `tiny_relation_checkpoint_dir`, `fake_neo4j_connection`, etc.).

### Phase 2: Fix Knowledge Graph Schema (Module 6)
- Modify `knowledge_graph/schema.py`:
  - Add `GraphSchemaError(Exception)`
  - Add `GraphSchema` dataclass with `allowed_node_labels`, `allowed_relationship_types`, `.default()`, `.validate_entity()`, `.validate_relation()`.
  - Add `GraphBuildReport` dataclass with `nodes_created`, `nodes_merged`, `relationships_created`, `relationships_merged`, `errors`, `total_nodes_processed`, `total_relationships_processed`, `has_errors`.

### Phase 3: Module-by-Module Verification (Modules 4 - 12)
- Run `pytest tests/test_ner.py -v`
- Run `pytest tests/test_relation_extraction.py -v`
- Run `pytest tests/test_knowledge_graph.py -v`
- Run `pytest tests/test_embeddings.py -v`
- Run `pytest tests/test_vector_db.py -v`
- Run `pytest tests/test_hybrid_retrieval.py -v`
- Run `pytest tests/test_adaptive_retrieval.py -v`
- Run `pytest tests/test_reranking.py -v`
- Run `pytest tests/test_llm.py -v`

### Phase 4: Module 13 Evaluation & Module 14 API/Pipeline
- Implement/verify `evaluation/retrieval_metrics.py`, `generation_metrics.py`, `ablation_runner.py`.
- Implement `api/pipeline.py` (wiring offline ingestion -> KG/Vector -> online hybrid retrieval -> LLM answer).
- Implement `api/routes.py` & `api/main.py` (FastAPI app).

### Phase 5: Integration & End-to-End Testing
- Add `tests/test_end_to_end.py` to test the full pipeline end-to-end offline.
- Run full `pytest -v` across the entire workspace.
- Update `README.md` with usage instructions, architecture notes, and running instructions.

---

## Verification Plan

### Automated Tests
- Run `pytest -v` to ensure all 180+ tests pass with zero errors.

### Manual Verification
- Execute an end-to-end dry-run script passing enterprise sample queries through the full pipeline:
  1. "What is the annual leave policy?"
  2. "Which department manages the cybersecurity policy?"
  3. "What software does the Finance project use?"
  4. "Which employees/roles are eligible for a particular leave?"
