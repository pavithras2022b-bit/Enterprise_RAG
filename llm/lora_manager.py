"""
Module 12 - LLM Response Generation: LoRA adapter management.

Provides LoRAAdapterManager per the Phase 4 module contract: loads a
base causal LM and manages LoRA adapters on top of it via peft,
supporting registering, loading/swapping, and unloading adapters
without reloading the base model.

This operates at the training/preparation layer, separate from
LLMClient's Ollama-served runtime path (client.py): a fine-tuned
adapter produced/managed here would be merged and exported for real
serving via Ollama/vLLM in a production deployment (Section 12.2),
not queried directly through this class at answer-generation time.
Its purpose is Section 8.8's adapter lifecycle: "multiple LoRA
adapters can even be trained for different purposes and swapped
in/out without retraining the base model."

Reference: Capstone Project Guide, Section 8.6/8.8 ("Transfer
Learning Applied to the LLM" / "LoRA"), Phase 4 Module 12 contract,
Phase 5 file structure (llm/lora_manager.py).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class LoRAAdapterManagerError(Exception):
    """Raised when a base model/adapter cannot be loaded or an unknown adapter is requested."""


class LoRAAdapterManager:
    """
    Manages a base causal LM plus zero or more registered LoRA
    adapters, with at most one active at a time. Registering an
    adapter only records its path; the adapter's weights are not
    loaded into memory until load_adapter() is called for it,
    consistent with lazy loading being the more memory-efficient
    default for a manager that may know about several adapters trained
    for different purposes but only serve one at a time.
    """

    def __init__(
        self,
        base_model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Args:
            base_model_name_or_path: A local checkpoint directory or
                Hugging Face Hub model id for the base causal LM
                (Section 12.3: Llama 3 / Qwen 2.5 / Mistral). Defaults
                to "Qwen/Qwen2.5-3B-Instruct".
            device: torch device string ("cpu" or "cuda"). If None,
                automatically uses "cuda" if available, else "cpu".
            torch_dtype: Optional torch data type for weights. Defaults
                to float16 on CUDA and float32 on CPU.

        Raises:
            LoRAAdapterManagerError: If the base model or tokenizer
                cannot be loaded.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        if torch_dtype is None:
            torch_dtype = torch.float16 if self._device.type == "cuda" else torch.float32

        try:
            self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                base_model_name_or_path,
                trust_remote_code=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise LoRAAdapterManagerError(
                f"Failed to load tokenizer from {base_model_name_or_path!r}: {exc}"
            ) from exc

        try:
            model_kwargs: dict[str, object] = {
                "torch_dtype": torch_dtype,
                "trust_remote_code": True,
            }
            if self._device.type == "cpu":
                model_kwargs["low_cpu_mem_usage"] = True

            self._base_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                base_model_name_or_path,
                **model_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            raise LoRAAdapterManagerError(
                f"Failed to load base model from {base_model_name_or_path!r}: {exc}"
            ) from exc

        self._base_model.to(self._device)

        self._adapter_paths: dict[str, str] = {}
        self._peft_model: Optional[PeftModel] = None
        self._active_adapter: Optional[str] = None

    def register_adapter(self, name: str, adapter_path: str) -> None:
        """
        Record an adapter's name and on-disk path for later loading.
        Does not load any weights.

        Args:
            name: Logical name for this adapter (e.g. "hr-policy-v1").
            adapter_path: Directory containing the adapter's
                adapter_config.json/adapter_model.safetensors (i.e.
                the output_dir of a prior peft training run).

        Raises:
            ValueError: If name is empty.
            LoRAAdapterManagerError: If adapter_path does not exist.
        """
        if not name:
            raise ValueError("register_adapter() requires a non-empty name.")
        if not os.path.isdir(adapter_path):
            raise LoRAAdapterManagerError(
                f"Adapter path {adapter_path!r} does not exist or is not a directory."
            )
        self._adapter_paths[name] = adapter_path
        logger.debug("Registered adapter %r -> %s", name, adapter_path)

    def list_adapters(self) -> list[str]:
        """Returns the names of every registered adapter, sorted."""
        return sorted(self._adapter_paths.keys())

    @property
    def active_adapter(self) -> Optional[str]:
        """The name of the currently active adapter, or None if unloaded/base-only."""
        return self._active_adapter

    def load_adapter(self, name: str) -> None:
        """
        Activate a registered adapter. If this is the first adapter
        ever loaded, wraps the base model in a PeftModel; subsequent
        calls (for a different registered adapter) attach it alongside
        any already-loaded adapters and switch the active one --
        neither path reloads the base model's weights from disk.

        Args:
            name: A name previously passed to register_adapter().

        Raises:
            LoRAAdapterManagerError: If name was never registered, or
                the adapter's weights fail to load.
        """
        if name not in self._adapter_paths:
            raise LoRAAdapterManagerError(
                f"Unknown adapter {name!r}; call register_adapter() first. "
                f"Registered adapters: {self.list_adapters()}"
            )
        adapter_path = self._adapter_paths[name]

        try:
            if self._peft_model is None:
                self._peft_model = PeftModel.from_pretrained(
                    self._base_model, adapter_path, adapter_name=name
                )
            elif name not in self._peft_model.peft_config:
                self._peft_model.load_adapter(adapter_path, adapter_name=name)
        except Exception as exc:  # noqa: BLE001
            raise LoRAAdapterManagerError(f"Failed to load adapter {name!r}: {exc}") from exc

        self._peft_model.set_adapter(name)
        self._active_adapter = name
        logger.info("Active adapter set to %r", name)

    def unload_adapter(self) -> None:
        """
        Deactivate the current adapter, reverting generation to the
        base model's unmodified behavior. Loaded adapter weights stay
        in memory (available for a fast load_adapter() call again);
        this only disables their effect.
        """
        if self._peft_model is not None:
            self._peft_model.disable_adapter_layers()
        self._active_adapter = None
        logger.info("Adapter unloaded; reverted to base model.")

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        """
        Generate text from the currently active model (the PeftModel
        with its active adapter, or the plain base model if no adapter
        has ever been loaded).

        Args:
            prompt: The input prompt text.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            The generated text (decoded, including the prompt --
            callers needing only the continuation should slice it off
            using the input length, since decoding behavior varies by
            tokenizer).

        Raises:
            ValueError: If prompt is empty/whitespace-only, or
                max_new_tokens is not positive.
            LoRAAdapterManagerError: If generation fails.
        """
        if not prompt or not prompt.strip():
            raise ValueError("generate() requires a non-empty prompt.")
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")

        model = self._peft_model if self._peft_model is not None else self._base_model
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

        try:
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        except Exception as exc:  # noqa: BLE001
            raise LoRAAdapterManagerError(f"Generation failed: {exc}") from exc

        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def load_default_rag_adapter(
        self,
        name: str = "qwen2.5-3b-lora-ragqa",
        adapter_path: Optional[str] = None,
    ) -> str:
        """
        Convenience method to register and load the primary fine-tuned LoRA adapter
        from outputs/qwen2.5-3b-lora-ragqa/final_adapter/ or outputs/qwen2.5-3b-lora-ragqa/.

        Args:
            name: Logical name to assign to the loaded adapter.
            adapter_path: Explicit directory path. If None, resolves from standard paths.

        Returns:
            The loaded adapter name.
        """
        if adapter_path is None:
            candidates = [
                os.path.join("outputs", "qwen2.5-3b-lora-ragqa", "final_adapter"),
                os.path.join("outputs", "qwen2.5-3b-lora-ragqa"),
            ]
            for cand in candidates:
                if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "adapter_config.json")):
                    adapter_path = cand
                    break

        if adapter_path is None or not os.path.isdir(adapter_path):
            raise LoRAAdapterManagerError(
                "Fine-tuned LoRA adapter checkpoint not found in outputs/qwen2.5-3b-lora-ragqa/"
            )

        self.register_adapter(name, adapter_path)
        self.load_adapter(name)
        return name
