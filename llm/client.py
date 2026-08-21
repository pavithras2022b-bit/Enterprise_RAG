"""
Module 12 - LLM Response Generation: LLM inference client.

Provides generate(...) and the LLMClient class per the Phase 4 module
contract: a thin HTTP wrapper around an Ollama-compatible generation
endpoint (Section 12.2: "Ollama/vLLM ... allow running Llama/Qwen/
Mistral entirely locally -- critical since enterprise data cannot be
sent to a third-party API").

Reference: Capstone Project Guide, Section 3M ("LLM Generation"),
Phase 4 Module 12 contract, Phase 5 file structure (llm/client.py).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import requests

from llm.prompt_builder import PromptBuilder, extract_citations
from llm.schema import GeneratedAnswer
from reranking.schema import RankedCandidate

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Raised when the LLM server request fails or returns an unusable response."""


class LLMClient:
    """
    Wraps an Ollama-compatible HTTP generation endpoint
    (POST {base_url}/api/generate). Deliberately talks to a locally-
    served model over HTTP rather than loading model weights directly
    in this process -- matching the project's chosen deployment shape
    (Section 11.3's component-level view: a separate LLM inference
    server process, not embedded in the retrieval API).

    For LoRA-adapted models, the adapter is expected to already be
    merged/registered into the model the server is serving under
    `model_name` (Section 8.8/8.9: LoRAAdapterManager, llm/lora_manager.py,
    handles adapter training/management as a separate, offline
    preparation step -- this client only calls whatever model tag the
    server currently has loaded).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "llama3",
        timeout: float = 60.0,
        temperature: float = 0.2,
        http_session: Optional[requests.Session] = None,
    ) -> None:
        """
        Args:
            base_url: Base URL of the Ollama-compatible server.
            model_name: Model tag to request generations from.
            timeout: Request timeout, in seconds.
            temperature: Sampling temperature. Section 3M recommends
                low-temperature/near-greedy decoding for factual
                enterprise Q&A, favoring accuracy over creativity.
            http_session: Injectable requests.Session, for testing
                against a local fake server without patching global
                state.
        """
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._timeout = timeout
        self._temperature = temperature
        self._session = http_session if http_session is not None else requests.Session()

    def generate(self, prompt: str) -> str:
        """
        Send a single generation request to the server.

        Args:
            prompt: The complete prompt text (typically from
                PromptBuilder.build_prompt()).

        Returns:
            The generated text.

        Raises:
            ValueError: If prompt is empty/whitespace-only.
            LLMClientError: If the HTTP request fails, the server
                returns a non-2xx status, the response body is not
                valid JSON, or the expected "response" field is
                missing.
        """
        if not prompt or not prompt.strip():
            raise ValueError("generate() requires a non-empty prompt.")

        url = f"{self._base_url}/api/generate"
        payload: dict[str, Any] = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self._temperature},
        }

        try:
            response = self._session.post(url, json=payload, timeout=self._timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LLMClientError(f"LLM generation request to {url} failed: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMClientError(
                f"LLM server at {url} returned a non-JSON response body: {exc}"
            ) from exc

        text = data.get("response")
        if text is None:
            raise LLMClientError(
                f"LLM server response is missing the expected 'response' field: {data!r}"
            )
        return text

    def generate_answer(
        self,
        query: str,
        evidence: list[RankedCandidate],
        prompt_builder: Optional[PromptBuilder] = None,
    ) -> GeneratedAnswer:
        """
        Full pipeline: build the grounded prompt, generate, extract
        and validate citations, and time the round trip.

        Args:
            query: The user's question.
            evidence: Top-N re-ranked candidates from Module 11.
            prompt_builder: Overrides the default PromptBuilder.

        Returns:
            A GeneratedAnswer with the generated text, validated
            citations, measured latency, and this client's model_name.

        Raises:
            ValueError: If query is empty/whitespace-only.
            LLMClientError: If generation fails.
        """
        builder = prompt_builder if prompt_builder is not None else PromptBuilder()
        prompt = builder.build_prompt(query, evidence)

        start_time = time.monotonic()
        text = self.generate(prompt)
        latency_ms = (time.monotonic() - start_time) * 1000.0

        citations = extract_citations(text, evidence)

        return GeneratedAnswer(
            text=text,
            citations=tuple(citations),
            latency_ms=latency_ms,
            model_name=self._model_name,
        )


def generate(client: LLMClient, prompt: str) -> str:
    """
    Module-level convenience wrapper around LLMClient.generate(),
    matching the Phase 4 module contract's function signature
    (generate(prompt) -> str).

    Args:
        client: A configured LLMClient.
        prompt: The complete prompt text.

    Returns:
        The generated text.
    """
    return client.generate(prompt)
