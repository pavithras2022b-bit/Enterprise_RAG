"""
Ingest all raw policy documents from data/raw/documents/ into the Enterprise RAG System.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from api.pipeline import EnterpriseRAGPipeline
from config_loader import create_embedding_model, create_vector_store, create_neo4j_connection, create_entity_extractor, create_relation_classifier


def main():
    raw_dir = Path("data/raw/documents")
    if not raw_dir.exists():
        logger.error("Raw documents directory %s not found!", raw_dir)
        sys.exit(1)

    file_paths = [str(p) for p in sorted(raw_dir.glob("*")) if p.is_file()]
    logger.info("Found %d raw policy documents in %s", len(file_paths), raw_dir)

    # Initialize components from configuration
    logger.info("Initializing RAG pipeline components from configuration...")
    try:
        embed_model = create_embedding_model()
        vector_store = create_vector_store()
        neo4j_conn = create_neo4j_connection()
        entity_extractor = create_entity_extractor()
        relation_classifier = create_relation_classifier()
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        sys.exit(1)

    pipeline = EnterpriseRAGPipeline(
        neo4j_connection=neo4j_conn,
        vector_store=vector_store,
        embedding_model=embed_model,
        entity_extractor=entity_extractor,
        relation_classifier=relation_classifier,
    )

    logger.info("Starting ingestion of %d policy documents...", len(file_paths))
    response = pipeline.ingest_files(file_paths)

    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE SUMMARY")
    logger.info("=" * 60)
    logger.info("Documents processed: %d", response.documents_processed)
    logger.info("Chunks created: %d", response.chunks_created)
    logger.info("Entities extracted: %d", response.entities_extracted)
    logger.info("Relations extracted: %d", response.relations_extracted)
    logger.info("Status: %s", response.status)


if __name__ == "__main__":
    main()




