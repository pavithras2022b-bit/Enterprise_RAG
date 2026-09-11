"""
Configuration loader for the Enterprise RAG System.

Loads YAML config files from configs/ directory and provides
factory functions to initialize components with the configured
parameters.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Default config directory
CONFIG_DIR = Path(__file__).parent / "configs"


class ConfigError(Exception):
    """Raised when configuration loading or validation fails."""


def load_yaml(config_file: str) -> dict[str, Any]:
    """
    Load a YAML configuration file from the configs directory.
    Supports environment variable substitution via ${VAR_NAME} syntax.

    Args:
        config_file: Filename (e.g., "embeddings.yaml") or path

    Returns:
        Parsed YAML dictionary with environment variables resolved

    Raises:
        ConfigError: If file not found or YAML is invalid
    """
    if not config_file.endswith(".yaml"):
        config_file = f"{config_file}.yaml"

    config_path = CONFIG_DIR / config_file

    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Replace ${VAR_NAME} with environment variable values
        def replace_env(match):
            var_name = match.group(1)
            value = os.getenv(var_name)
            if value is None:
                logger.warning(f"Environment variable {var_name} not set, using default 'neo4j'")
                return "neo4j"  # Default password for development
            return value

        content = re.sub(r'\$\{([^}]+)\}', replace_env, content)

        return yaml.safe_load(content) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse {config_path}: {e}") from e


def get_embeddings_config() -> dict[str, Any]:
    """Get embeddings module configuration."""
    config = load_yaml("embeddings.yaml")
    return config.get("embeddings", {})


def get_llm_config() -> dict[str, Any]:
    """Get LLM module configuration."""
    config = load_yaml("llm.yaml")
    return config.get("llm", {})


def get_vector_db_config() -> dict[str, Any]:
    """Get vector database configuration."""
    config = load_yaml("vector_db.yaml")
    return config.get("vector_db", {})


def get_knowledge_graph_config() -> dict[str, Any]:
    """Get knowledge graph configuration."""
    config = load_yaml("knowledge_graph.yaml")
    return config.get("knowledge_graph", {})


def get_chunking_config() -> dict[str, Any]:
    """Get chunking configuration."""
    config = load_yaml("chunking.yaml")
    return config.get("chunking", {})


def get_ner_config() -> dict[str, Any]:
    """Get NER configuration."""
    config = load_yaml("ner.yaml")
    return config.get("ner", {})


def get_relation_extraction_config() -> dict[str, Any]:
    """Get relation extraction configuration."""
    config = load_yaml("relation_extraction.yaml")
    return config.get("relation_extraction", {})


def get_reranking_config() -> dict[str, Any]:
    """Get reranking configuration."""
    config = load_yaml("reranking.yaml")
    return config.get("reranking", {})


def get_adaptive_retrieval_config() -> dict[str, Any]:
    """Get adaptive retrieval configuration."""
    config = load_yaml("adaptive_retrieval.yaml")
    return config.get("adaptive_retrieval", {})


def get_ingestion_config() -> dict[str, Any]:
    """Get ingestion configuration."""
    config = load_yaml("ingestion.yaml")
    return config.get("ingestion", {})


def create_embedding_model():
    """
    Factory function to create an EmbeddingModel with configured parameters.

    Returns:
        Initialized EmbeddingModel instance

    Raises:
        ConfigError: If configuration is invalid or model cannot load
    """
    from embeddings.model import EmbeddingModel

    config = get_embeddings_config()
    model_config = config.get("model", {})

    checkpoint = model_config.get("checkpoint", "all-MiniLM-L6-v2")
    device = model_config.get("device", "cpu")
    normalize = model_config.get("normalize_embeddings", True)

    logger.info(f"Loading embedding model: {checkpoint} on device {device}")
    try:
        return EmbeddingModel(
            model_name_or_path=checkpoint,
            device=device,
            normalize_embeddings=normalize,
        )
    except Exception as e:
        raise ConfigError(f"Failed to create EmbeddingModel: {e}") from e


def create_vector_store():
    """
    Factory function to create a VectorStore with configured parameters.

    Returns:
        Initialized VectorStore instance
    """
    from vector_db.chroma_store import VectorStore

    config = get_vector_db_config()
    chroma_config = config.get("chroma", {})

    collection_name = chroma_config.get("collection_name", "enterprise_docs")
    persist_directory = chroma_config.get("persist_directory")

    logger.info(f"Initializing vector store: {collection_name}")
    if persist_directory:
        logger.info(f"  Persistence directory: {persist_directory}")
    else:
        logger.warning("  Using ephemeral (in-memory) storage - data will not persist!")

    return VectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )


def create_neo4j_connection():
    """
    Factory function to create a Neo4jConnection with configured parameters.

    Returns:
        Initialized Neo4jConnection instance

    Raises:
        ConfigError: If Neo4j connection cannot be established
    """
    from knowledge_graph.connection import Neo4jConnection

    config = get_knowledge_graph_config()
    neo4j_config = config.get("connection", {})

    uri = neo4j_config.get("uri", "bolt://localhost:7687")
    username = neo4j_config.get("username", "neo4j")
    password = neo4j_config.get("password", "neo4j")
    database = neo4j_config.get("database", "neo4j")

    logger.info(f"Connecting to Neo4j at {uri} (database: {database})")
    try:
        return Neo4jConnection(
            uri=uri,
            username=username,
            password=password,
            database=database,
        )
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        raise ConfigError(f"Neo4j connection failed: {e}") from e


def create_llm_client():
    """
    Factory function to create an LLMClient with configured parameters.

    Returns:
        Initialized LLMClient instance
    """
    from llm.client import LLMClient

    config = get_llm_config()
    client_config = config.get("client", {})

    base_url = client_config.get("base_url", "http://localhost:11434")
    model_name = client_config.get("model_name", "llama3")
    timeout = client_config.get("timeout_seconds", 60.0)
    temperature = client_config.get("temperature", 0.2)

    logger.info(f"Creating LLM client: {model_name} at {base_url}")
    return LLMClient(
        base_url=base_url,
        model_name=model_name,
        timeout=timeout,
        temperature=temperature,
    )


def create_lora_adapter_manager():
    """
    Factory function to create a LoRAAdapterManager with configured parameters.

    Returns:
        Initialized LoRAAdapterManager instance
    """
    from llm.lora_manager import LoRAAdapterManager

    config = get_llm_config()
    lora_config = config.get("lora", {})

    base_model = lora_config.get("base_model") or "Qwen/Qwen2.5-3B-Instruct"
    device = lora_config.get("device", "cpu")
    adapters = lora_config.get("adapters", {})
    active_adapter = lora_config.get("active_adapter")

    logger.info(f"Creating LoRA adapter manager with base model: {base_model}")
    manager = LoRAAdapterManager(base_model_name_or_path=base_model, device=device)
    for name, path in adapters.items():
        if os.path.exists(path):
            manager.register_adapter(name, path)

    if active_adapter and active_adapter in manager.list_adapters():
        manager.load_adapter(active_adapter)
    return manager


def create_entity_extractor():
    """
    Factory function to create an EntityExtractor with configured parameters.

    Returns:
        Initialized EntityExtractor instance or None if not configured
    """
    from ner.extractor import EntityExtractor

    config = get_ner_config()
    checkpoint = config.get("model_name_or_path", "dslim/bert-base-NER")
    device = config.get("device", "cpu")
    use_generic_label_mapping = config.get("use_generic_label_mapping", False)

    logger.info(f"Loading NER model: {checkpoint}")
    try:
        return EntityExtractor(
            model_name_or_path=checkpoint,
            device=device,
            use_generic_label_mapping=use_generic_label_mapping,
        )
    except Exception as e:
        logger.warning(f"Could not initialize EntityExtractor: {e}")
        return None


def create_relation_classifier():
    """
    Factory function to create a RelationClassifier with configured parameters.

    Returns:
        Initialized RelationClassifier instance or None if not configured
    """
    from relation_extraction.model import RelationClassifier

    config = get_relation_extraction_config()
    model_config = config.get("model", {})
    if not model_config.get("enabled", True):
        logger.info("Learned relation classifier disabled by configuration")
        return None

    checkpoint = model_config.get("checkpoint")
    if not checkpoint:
        logger.warning("No relation extraction checkpoint configured; skipping")
        return None

    device = model_config.get("device", "cpu")

    logger.info(f"Loading relation extraction model: {checkpoint}")
    try:
        return RelationClassifier(model_name_or_path=checkpoint, device=device)
    except Exception as e:
        logger.warning(f"Could not initialize RelationClassifier: {e}")
        return None



def create_alpha_classifier():
    """Create the configured adaptive-retrieval alpha classifier."""
    from adaptive_retrieval.rules import RuleBasedAlphaClassifier

    config = get_adaptive_retrieval_config()
    classifier_type = str(
        config.get("classifier_type", "rule_based")
    ).strip().lower()

    if classifier_type == "learned":
        try:
            import joblib
            from adaptive_retrieval.classifier import LearnedAlphaClassifier

            learned_config = config.get("learned", {})
            checkpoint_path = learned_config.get(
                "checkpoint_path",
                "models/alpha_classifier/model.joblib",
            )

            model = joblib.load(checkpoint_path)

            classifier = LearnedAlphaClassifier.from_sklearn_model(model)

            if not classifier.is_fitted:
                raise ValueError(
                    f"Loaded alpha classifier at {checkpoint_path!r} is not fitted."
                )

            logger.info(
                "Loaded learned alpha classifier from %s",
                checkpoint_path,
            )

            return classifier

        except Exception as exc:
            logger.warning(
                "Could not load learned alpha classifier; "
                "falling back to rule-based classifier: %s",
                exc,
            )

            return RuleBasedAlphaClassifier()

    if classifier_type != "rule_based":
        logger.warning(
            "Unknown adaptive classifier_type=%r; "
            "falling back to rule_based.",
            classifier_type,
        )

    return RuleBasedAlphaClassifier()


def create_chunker():
    """
    Factory function to create a Chunker with configured parameters.

    Returns:
        Initialized Chunker instance
    """
    from chunking.strategies import create_chunker as create_chunker_impl

    config = get_chunking_config()
    strategy = config.get("strategy", "structure_aware")
    target_size = config.get("target_size", 512)
    overlap = config.get("overlap", 64)

    logger.info(f"Creating {strategy} chunker with target_size={target_size}, overlap={overlap}")
    return create_chunker_impl(
        strategy=strategy,
        target_size=target_size,
        overlap=overlap,
    )


def create_cross_encoder_reranker():
    """
    Factory function to create a CrossEncoderReranker with configured parameters.

    Returns:
        Initialized CrossEncoderReranker instance
    """
    from reranking.model import CrossEncoderReranker

    config = get_reranking_config()
    model_config = config.get("model", {})

    checkpoint = model_config.get("checkpoint", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    device = model_config.get("device", "cpu")

    logger.info(f"Loading reranker model: {checkpoint}")
    return CrossEncoderReranker(model_name_or_path=checkpoint, device=device)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Config loader module. Use the factory functions to create components.")
