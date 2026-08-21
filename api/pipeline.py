"""
Module 14 - API & Service Layer: End-to-End RAG Pipeline Orchestrator.

Orchestrates Modules 1 through 13 into a unified pipeline for document ingestion
and query answering.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np

from adaptive_retrieval.fusion import AdaptiveFusionEngine
from adaptive_retrieval.rules import RuleBasedAlphaClassifier
from config_loader import create_llm_client
from datetime import datetime, timezone
from pathlib import Path

from api.schema import CandidateDTO, IngestResponse, QueryResponse
from chunking.strategies import StructureAwareChunker
from embeddings.model import EmbeddingModel
from ingestion.dedupe import compute_checksum
from ingestion.loaders import DocumentLoader
from ingestion.schema import DocumentFormat, RawDocument, SourceSystem
from knowledge_graph.builder import Neo4jGraphBuilder
from knowledge_graph.connection import Neo4jConnection
from llm.client import LLMClient
from ner.extractor import EntityExtractor
from preprocessing.cleaner import DocumentCleaner
from preprocessing.schema import CleanDocument
from relation_extraction.model import RelationClassifier
from relation_extraction.rules import RulePatternExtractor
from reranking.model import CrossEncoderReranker
from reranking.schema import RankedCandidate
from retrieval.entity_linker import EntityLinker
from retrieval.hybrid_controller import HybridRetrievalController
from vector_db.chroma_store import VectorStore

logger = logging.getLogger(__name__)


class EnterpriseRAGPipeline:
    """
    End-to-end pipeline container linking all core RAG modules.
    """

    def __init__(
        self,
        neo4j_connection: Neo4jConnection,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        entity_extractor: Optional[EntityExtractor] = None,
        relation_classifier: Optional[RelationClassifier] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self.neo4j_connection = neo4j_connection
        self.vector_store = vector_store
        self.embedding_model = embedding_model

        self.document_loader = DocumentLoader()
        self.preprocessor = DocumentCleaner()
        self.chunker = StructureAwareChunker()

        self.entity_extractor = entity_extractor
        self.relation_classifier = relation_classifier
        # Source records often include a title/department parenthesis between
        # an employee and the explicit "reports to" phrase.
        self.rule_relation_extractor = RulePatternExtractor(max_connector_length=250)
        self.reranker = reranker
        self.llm_client = llm_client or create_llm_client()

        # Initialize EntityLinker with graceful Neo4j failure handling
        try:
            self.entity_linker = EntityLinker(neo4j_connection, refresh_on_init=True)
        except Exception as e:
            logger.warning(f"Failed to initialize EntityLinker (Neo4j may be unavailable): {e}")
            self.entity_linker = EntityLinker(neo4j_connection, refresh_on_init=False)
        
        self.hybrid_controller = HybridRetrievalController(
            vector_store=self.vector_store,
            graph_connection=self.neo4j_connection,
            embedding_model=self.embedding_model,
            entity_linker=self.entity_linker,
        )
        self.adaptive_engine = AdaptiveFusionEngine(
            controller=self.hybrid_controller,
            embedding_model=self.embedding_model,
            entity_linker=self.entity_linker,
            classifier=RuleBasedAlphaClassifier(),
        )

    def ingest_files(self, file_paths: list[str]) -> IngestResponse:
        """
        Runs document ingestion flow: load -> preprocess -> chunk -> embed & index -> extract entities/relations -> populate Knowledge Graph.
        """
        docs: list[RawDocument] = []
        for path_str in file_paths:
            path = Path(path_str)
            if self.document_loader.supports(path):
                docs.append(self.document_loader.load(path))
            elif path.suffix.lower() in (".txt", ".md"):
                text = path.read_text(encoding="utf-8")
                checksum = compute_checksum(text.encode("utf-8"))
                doc_id = f"txt_{checksum[:16]}"
                raw_doc = RawDocument(
                    document_id=doc_id,
                    text=text,
                    source_path=str(path),
                    source_system=SourceSystem.LOCAL_FILESYSTEM,
                    document_format=DocumentFormat.HTML,
                    collected_at=datetime.now(timezone.utc),
                    checksum=checksum,
                )
                docs.append(raw_doc)

        all_chunks = []
        for raw_doc in docs:
            cleaned_text = self.preprocessor.clean(raw_doc.text)
            clean_doc = CleanDocument.from_raw_document(raw_doc, cleaned_text)
            chunks = self.chunker.chunk(clean_doc)
            all_chunks.extend(chunks)

        if all_chunks:
            embedded_chunks = self.embedding_model.embed_chunks(all_chunks)
            self.vector_store.add(embedded_chunks)

        # KG population. The learned classifier is optional because the
        # project includes a schema-aware rule baseline for unlabeled data.
        entities_count = 0
        relations_count = 0
        if self.entity_extractor:
            from relation_extraction.model import predict_relations_for_chunk

            all_entities = []
            all_relations = []
            for chunk in all_chunks:
                chunk_entities = self.entity_extractor.extract_entities_from_chunk(chunk)
                all_entities.extend(chunk_entities)
                chunk_relations = []
                if self.relation_classifier:
                    chunk_relations.extend(predict_relations_for_chunk(
                        self.relation_classifier, chunk, chunk_entities
                    ))
                chunk_relations.extend(self.rule_relation_extractor.apply_rule_patterns(
                    chunk.text, chunk_entities, chunk.chunk_id
                ))
                all_relations.extend(chunk_relations)

            logger.info(
                "KG extraction produced %d entity span(s) and %d relation(s) "
                "before graph writes.", len(all_entities), len(all_relations)
            )

            graph_builder = Neo4jGraphBuilder(connection=self.neo4j_connection)
            report = graph_builder.build_graph(all_entities, all_relations)
            entities_count = report.nodes_created
            relations_count = report.relationships_created

        # Refresh entity linker index for graph search (gracefully handle Neo4j unavailability)
        try:
            self.entity_linker.refresh_index()
        except Exception as e:
            logger.warning(f"Failed to refresh entity linker index (Neo4j may be unavailable): {e}")

        return IngestResponse(
            documents_processed=len(docs),
            chunks_created=len(all_chunks),
            entities_extracted=entities_count,
            relations_extracted=relations_count,
            status="success",
        )

    def answer_query(
        self,
        query: str,
        alpha: Optional[float] = None,
        top_k: int = 5,
        use_adaptive: bool = True,
        use_reranking: bool = True,
    ) -> QueryResponse:
        """
        Executes end-to-end query answering flow:
        Retrieval -> (Optional) Reranking -> Evidence Prompt Assembly -> LLM Generation.
        """
        start_time = time.perf_counter()

        # 1. Hybrid / Adaptive Retrieval
        if alpha is not None:
            alpha_used = alpha
            fused_candidates = self.hybrid_controller.retrieve(
                query=query, alpha=alpha, k_vector=top_k * 2, k_graph=top_k * 2
            )
        elif use_adaptive:
            alpha_used, fused_candidates = self.adaptive_engine.retrieve(
                query=query, k_vector=top_k * 2, k_graph=top_k * 2
            )
        else:
            alpha_used = 0.5
            fused_candidates = self.hybrid_controller.retrieve(
                query=query, alpha=0.5, k_vector=top_k * 2, k_graph=top_k * 2
            )

        # 2. Reranking (optional)
        if use_reranking and self.reranker is not None and fused_candidates:
            ranked_candidates = self.reranker.rerank(query, fused_candidates, top_n=top_k)
        else:
            ranked_candidates = [
                RankedCandidate(
                    candidate_id=c.candidate_id,
                    text=c.text,
                    rerank_score=c.fused_score,
                    original_candidate=c,
                    rank=i + 1,
                )
                for i, c in enumerate(fused_candidates[:top_k])
            ]

        final_candidates = [
            CandidateDTO(
                candidate_id=r.candidate_id,
                source=r.original_candidate.source.value,
                fused_score=r.rerank_score,
                text=r.text,
                metadata=r.original_candidate.metadata,
            )
            for r in ranked_candidates
        ]

        # 3. LLM Generation
        try:
            gen_answer = self.llm_client.generate_answer(query=query, evidence=ranked_candidates)
            answer_text = gen_answer.text
            citations = list(gen_answer.citations)
        except Exception as e:
            logger.warning(f"LLM generation failed (Ollama may be unavailable): {e}")
            # Graceful fallback: summarize retrieved evidence
            if ranked_candidates:
                evidence_texts = [c.text for c in ranked_candidates[:3]]
                answer_text = f"Based on the retrieved policy documents: {' '.join(evidence_texts)}"
                citations = [c.candidate_id for c in ranked_candidates[:3]]
            else:
                answer_text = "No relevant policy documents found for this query."
                citations = []

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0

        return QueryResponse(
            query=query,
            answer=answer_text,
            citations=citations,
            alpha_used=alpha_used,
            latency_ms=latency_ms,
            source_candidates=final_candidates,
        )
