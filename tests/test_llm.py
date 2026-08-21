"""
Unit tests for Module 12 - LLM Response Generation.

Covers llm/schema.py, prompt_builder.py, client.py, and lora_manager.py.
client.py tests use the fake_ollama_server fixture (a real local HTTP
server, not a mocked transport). lora_manager.py tests use the
tiny_causal_lm_dir / make_tiny_lora_adapter fixtures (see conftest.py)
to exercise the real transformers/peft/torch code paths fully offline.
"""

from __future__ import annotations

import json
import logging

import pytest

from llm.client import LLMClient, LLMClientError, generate
from llm.lora_manager import LoRAAdapterManager, LoRAAdapterManagerError
from llm.prompt_builder import (
    PromptBuilder,
    PromptBuilderError,
    build_prompt,
    extract_citations,
)
from llm.schema import GeneratedAnswer, SchemaValidationError
from retrieval.schema import CandidateSource, FusedCandidate
from reranking.schema import RankedCandidate

logging.disable(logging.CRITICAL)


def _ranked(candidate_id: str, text: str, rank: int = 1) -> RankedCandidate:
    original = FusedCandidate(
        candidate_id=candidate_id, source=CandidateSource.VECTOR,
        vector_score=0.9, graph_score=None, fused_score=0.9, text=text,
    )
    return RankedCandidate(
        candidate_id=candidate_id, text=text, rerank_score=0.9,
        original_candidate=original, rank=rank,
    )


# --------------------------------------------------------------------------
# schema.py
# --------------------------------------------------------------------------


class TestGeneratedAnswer:
    def test_valid_construction(self) -> None:
        answer = GeneratedAnswer(
            text="Casual leave is 12 days per year [chunk-1].",
            citations=("chunk-1",), latency_ms=42.5, model_name="llama3",
        )
        assert answer.has_citations is True

    def test_no_citations(self) -> None:
        answer = GeneratedAnswer(text="I don't know.", model_name="llama3")
        assert answer.has_citations is False

    def test_empty_text_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            GeneratedAnswer(text="   ", model_name="llama3")

    def test_negative_latency_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            GeneratedAnswer(text="answer", latency_ms=-1.0, model_name="llama3")

    def test_nan_latency_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            GeneratedAnswer(text="answer", latency_ms=float("nan"), model_name="llama3")

    def test_empty_model_name_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            GeneratedAnswer(text="answer", model_name="")

    def test_empty_citation_entry_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            GeneratedAnswer(text="answer", citations=("",), model_name="llama3")


# --------------------------------------------------------------------------
# prompt_builder.py
# --------------------------------------------------------------------------


class TestPromptBuilder:
    def test_build_prompt_includes_query_and_evidence(self) -> None:
        builder = PromptBuilder()
        evidence = [_ranked("chunk-1", "Casual leave is 12 days per year.")]
        prompt = builder.build_prompt("What is the leave policy?", evidence)
        assert "What is the leave policy?" in prompt
        assert "[chunk-1]" in prompt
        assert "Casual leave is 12 days per year." in prompt

    def test_build_prompt_empty_query_raises(self) -> None:
        builder = PromptBuilder()
        with pytest.raises(PromptBuilderError):
            builder.build_prompt("   ", [])

    def test_build_prompt_empty_evidence_notes_no_evidence(self) -> None:
        builder = PromptBuilder()
        prompt = builder.build_prompt("some question", [])
        assert "No relevant evidence" in prompt

    def test_custom_system_instruction(self) -> None:
        builder = PromptBuilder(system_instruction="Custom instruction here.")
        prompt = builder.build_prompt("query", [])
        assert "Custom instruction here." in prompt

    def test_module_level_wrapper(self) -> None:
        evidence = [_ranked("c1", "some text")]
        prompt = build_prompt("query", evidence)
        assert "[c1]" in prompt


class TestExtractCitations:
    def test_extracts_valid_citation(self) -> None:
        evidence = [_ranked("chunk-1", "text one"), _ranked("chunk-2", "text two")]
        citations = extract_citations("The answer is X [chunk-1].", evidence)
        assert citations == ["chunk-1"]

    def test_extracts_multiple_citations_in_order(self) -> None:
        evidence = [_ranked("chunk-1", "text one"), _ranked("chunk-2", "text two")]
        citations = extract_citations("First [chunk-2], then [chunk-1].", evidence)
        assert citations == ["chunk-2", "chunk-1"]

    def test_deduplicates_repeated_citation(self) -> None:
        evidence = [_ranked("chunk-1", "text one")]
        citations = extract_citations("[chunk-1] and again [chunk-1].", evidence)
        assert citations == ["chunk-1"]

    def test_ignores_bracket_not_in_evidence(self) -> None:
        evidence = [_ranked("chunk-1", "text one")]
        citations = extract_citations("This cites [chunk-999] which was never given.", evidence)
        assert citations == []

    def test_empty_answer_returns_empty_list(self) -> None:
        evidence = [_ranked("chunk-1", "text one")]
        assert extract_citations("", evidence) == []

    def test_no_brackets_returns_empty_list(self) -> None:
        evidence = [_ranked("chunk-1", "text one")]
        assert extract_citations("An answer with no citations at all.", evidence) == []

    def test_graph_fact_id_shape_citation(self) -> None:
        # GraphFact ids look like "asha rao|belongs_to|finance" -- no
        # brackets inside, so the citation regex handles them fine.
        evidence = [_ranked("asha rao|belongs_to|finance", "Asha Rao belongs to Finance.")]
        citations = extract_citations("As shown [asha rao|belongs_to|finance].", evidence)
        assert citations == ["asha rao|belongs_to|finance"]


# --------------------------------------------------------------------------
# client.py
# --------------------------------------------------------------------------


class TestLLMClient:
    def test_generate_returns_response_text(self, fake_ollama_server) -> None:
        server, base_url = fake_ollama_server
        server.response_spec = {
            "status": 200,
            "body": json.dumps({"response": "Casual leave is 12 days."}).encode("utf-8"),
        }
        client = LLMClient(base_url=base_url, model_name="test-model", timeout=5.0)
        text = client.generate("What is the leave policy?")
        assert text == "Casual leave is 12 days."

    def test_generate_sends_expected_payload(self, fake_ollama_server) -> None:
        server, base_url = fake_ollama_server
        client = LLMClient(base_url=base_url, model_name="my-model", temperature=0.3, timeout=5.0)
        client.generate("hello")
        assert server.last_request["model"] == "my-model"
        assert server.last_request["prompt"] == "hello"
        assert server.last_request["stream"] is False
        assert server.last_request["options"]["temperature"] == pytest.approx(0.3)

    def test_generate_empty_prompt_raises(self, fake_ollama_server) -> None:
        _server, base_url = fake_ollama_server
        client = LLMClient(base_url=base_url, timeout=5.0)
        with pytest.raises(ValueError):
            client.generate("   ")

    def test_generate_http_error_status_raises(self, fake_ollama_server) -> None:
        server, base_url = fake_ollama_server
        server.response_spec = {"status": 500, "body": b"Internal Server Error"}
        client = LLMClient(base_url=base_url, timeout=5.0)
        with pytest.raises(LLMClientError):
            client.generate("hello")

    def test_generate_malformed_json_raises(self, fake_ollama_server) -> None:
        server, base_url = fake_ollama_server
        server.response_spec = {"status": 200, "body": b"not valid json"}
        client = LLMClient(base_url=base_url, timeout=5.0)
        with pytest.raises(LLMClientError):
            client.generate("hello")

    def test_generate_missing_response_field_raises(self, fake_ollama_server) -> None:
        server, base_url = fake_ollama_server
        server.response_spec = {
            "status": 200,
            "body": json.dumps({"unexpected": "field"}).encode("utf-8"),
        }
        client = LLMClient(base_url=base_url, timeout=5.0)
        with pytest.raises(LLMClientError):
            client.generate("hello")

    def test_connection_refused_raises_llm_client_error(self) -> None:
        client = LLMClient(base_url="http://127.0.0.1:1", timeout=2.0)
        with pytest.raises(LLMClientError):
            client.generate("hello")

    def test_module_level_generate(self, fake_ollama_server) -> None:
        _server, base_url = fake_ollama_server
        client = LLMClient(base_url=base_url, timeout=5.0)
        text = generate(client, "hello")
        assert isinstance(text, str)

    def test_generate_answer_full_pipeline(self, fake_ollama_server) -> None:
        server, base_url = fake_ollama_server
        server.response_spec = {
            "status": 200,
            "body": json.dumps(
                {"response": "Casual leave is 12 days per year [chunk-1]."}
            ).encode("utf-8"),
        }
        client = LLMClient(base_url=base_url, model_name="test-model", timeout=5.0)
        evidence = [_ranked("chunk-1", "Casual leave is 12 days per year.")]

        answer = client.generate_answer("What is the leave policy?", evidence)

        assert isinstance(answer, GeneratedAnswer)
        assert answer.citations == ("chunk-1",)
        assert answer.model_name == "test-model"
        assert answer.latency_ms >= 0.0

    def test_generate_answer_filters_hallucinated_citation(self, fake_ollama_server) -> None:
        server, base_url = fake_ollama_server
        server.response_spec = {
            "status": 200,
            "body": json.dumps(
                {"response": "According to [chunk-999], leave is unlimited."}
            ).encode("utf-8"),
        }
        client = LLMClient(base_url=base_url, timeout=5.0)
        evidence = [_ranked("chunk-1", "Casual leave is 12 days per year.")]

        answer = client.generate_answer("query", evidence)
        assert answer.citations == ()

    def test_generate_answer_empty_query_raises(self, fake_ollama_server) -> None:
        _server, base_url = fake_ollama_server
        client = LLMClient(base_url=base_url, timeout=5.0)
        with pytest.raises(PromptBuilderError):
            client.generate_answer("   ", [])


# --------------------------------------------------------------------------
# lora_manager.py
# --------------------------------------------------------------------------


class TestLoRAAdapterManager:
    def test_loads_base_model(self, tiny_causal_lm_dir: str) -> None:
        manager = LoRAAdapterManager(tiny_causal_lm_dir)
        assert manager.active_adapter is None
        assert manager.list_adapters() == []

    def test_invalid_base_model_raises(self, tmp_path) -> None:
        with pytest.raises(LoRAAdapterManagerError):
            LoRAAdapterManager(str(tmp_path / "does_not_exist"))

    def test_register_adapter(self, tiny_causal_lm_dir: str, make_tiny_lora_adapter) -> None:
        manager = LoRAAdapterManager(tiny_causal_lm_dir)
        adapter_path = make_tiny_lora_adapter("a")
        manager.register_adapter("a", adapter_path)
        assert manager.list_adapters() == ["a"]

    def test_register_adapter_empty_name_raises(self, tiny_causal_lm_dir: str, make_tiny_lora_adapter) -> None:
        manager = LoRAAdapterManager(tiny_causal_lm_dir)
        adapter_path = make_tiny_lora_adapter("a")
        with pytest.raises(ValueError):
            manager.register_adapter("", adapter_path)

    def test_register_adapter_bad_path_raises(self, tiny_causal_lm_dir: str, tmp_path) -> None:
        manager = LoRAAdapterManager(tiny_causal_lm_dir)
        with pytest.raises(LoRAAdapterManagerError):
            manager.register_adapter("a", str(tmp_path / "does_not_exist"))

    def test_load_unregistered_adapter_raises(self, tiny_causal_lm_dir: str) -> None:
        manager = LoRAAdapterManager(tiny_causal_lm_dir)
        with pytest.raises(LoRAAdapterManagerError):
            manager.load_adapter("nonexistent")

    def test_load_adapter_sets_active(self, tiny_causal_lm_dir: str, make_tiny_lora_adapter) -> None:
        manager = LoRAAdapterManager(tiny_causal_lm_dir)
        adapter_path = make_tiny_lora_adapter("a")
        manager.register_adapter("a", adapter_path)
        manager.load_adapter("a")
        assert manager.active_adapter == "a"

    def test_swap_between_two_adapters(self, tiny_causal_lm_dir: str, make_tiny_lora_adapter) -> None:
        manager = LoRAAdapterManager(tiny_causal_lm_dir)
        manager.register_adapter("a", make_tiny_lora_adapter("a"))
        manager.register_adapter("b", make_tiny_lora_adapter("b"))

        manager.load_adapter("a")
        assert manager.active_adapter == "a"

        manager.load_adapter("b")
        assert manager.active_adapter == "b"

        # Swapping back to "a" should not require re-registering or
        # error out -- its weights are already attached.
        manager.load_adapter("a")
        assert manager.active_adapter == "a"

    def test_unload_adapter_reverts_to_none(self, tiny_causal_lm_dir: str, make_tiny_lora_adapter) -> None:
        manager = LoRAAdapterManager(tiny_causal_lm_dir)
        manager.register_adapter("a", make_tiny_lora_adapter("a"))
        manager.load_adapter("a")
        manager.unload_adapter()
        assert manager.active_adapter is None

    def test_generate_with_base_model_only(self, tiny_causal_lm_dir: str) -> None:
        manager = LoRAAdapterManager(tiny_causal_lm_dir)
        text = manager.generate("Asha Rao", max_new_tokens=3)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_generate_with_active_adapter(self, tiny_causal_lm_dir: str, make_tiny_lora_adapter) -> None:
        manager = LoRAAdapterManager(tiny_causal_lm_dir)
        manager.register_adapter("a", make_tiny_lora_adapter("a"))
        manager.load_adapter("a")
        text = manager.generate("Asha Rao", max_new_tokens=3)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_generate_empty_prompt_raises(self, tiny_causal_lm_dir: str) -> None:
        manager = LoRAAdapterManager(tiny_causal_lm_dir)
        with pytest.raises(ValueError):
            manager.generate("   ")

    def test_generate_non_positive_max_new_tokens_raises(self, tiny_causal_lm_dir: str) -> None:
        manager = LoRAAdapterManager(tiny_causal_lm_dir)
        with pytest.raises(ValueError):
            manager.generate("prompt", max_new_tokens=0)
