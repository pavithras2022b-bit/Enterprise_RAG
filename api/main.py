"""
Module 14 - API & Service Layer: FastAPI Entrypoint.

Creates and configures the FastAPI application instance and initializes the default pipeline on startup.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.pipeline import EnterpriseRAGPipeline
from api.routes import get_pipeline, router, set_pipeline
from config_loader import create_embedding_model, create_vector_store, create_neo4j_connection

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to initialize the pipeline on server startup."""
    try:
        get_pipeline()
    except Exception:
        logger.info("Initializing default EnterpriseRAGPipeline from configuration...")
        try:
            embed_model = create_embedding_model()
            vector_store = create_vector_store()
            neo4j_conn = create_neo4j_connection()

            pipeline = EnterpriseRAGPipeline(
                neo4j_connection=neo4j_conn,
                vector_store=vector_store,
                embedding_model=embed_model,
            )
            set_pipeline(pipeline)
            logger.info("EnterpriseRAGPipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    yield


app = FastAPI(
    title="Enterprise Retrieval-Augmented Generation API",
    description="Production-grade Hybrid Graph-Vector Enterprise RAG system API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
