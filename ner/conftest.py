"""
Shared pytest fixtures.

tiny_bert_checkpoint_dir builds a miniature BERT-architecture
token-classification checkpoint entirely from local objects (no
Hugging Face Hub download), so ner/model.py and ner/train.py can be
exercised end-to-end against a real tokenizer + real transformer
forward/backward pass, fully offline.
"""

from __future__ import annotations

import pytest
from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast

from ner.schema import build_bio_label_maps

_VOCAB_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "asha",
    "rao",
    "belongs",
    "to",
    "the",
    "finance",
    "department",
    "employee",
    "handbook",
    "policy",
    "vpn",
    "access",
    "requires",
    "manager",
    "approval",
    "leave",
    "casual",
    "sick",
    "days",
    "year",
    "project",
    "payments",
    "modernization",
    "uses",
    "sap",
    "fi",
    "role",
    "analyst",
    "network",
    "gateway",
    "software",
    "engineering",
    ".",
    ",",
]


@pytest.fixture(scope="session")
def tiny_bert_checkpoint_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """
    Builds and saves a tiny, randomly-initialized BERT token-classification
    checkpoint to a temp directory, using a hand-written vocab (no
    network access). Session-scoped since building/saving it is not
    free and its content is never mutated by tests.
    """
    vocab_dir = tmp_path_factory.mktemp("tiny_bert_vocab")
    vocab_file = vocab_dir / "vocab.txt"
    vocab_file.write_text("\n".join(_VOCAB_TOKENS), encoding="utf-8")

    tokenizer = BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=True)

    id2label, label2id = build_bio_label_maps()
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=128,
        num_labels=len(id2label),
    )
    config.id2label = id2label
    config.label2id = label2id
    model = BertForTokenClassification(config)

    checkpoint_dir = tmp_path_factory.mktemp("tiny_bert_checkpoint")
    tokenizer.save_pretrained(str(checkpoint_dir))
    model.save_pretrained(str(checkpoint_dir))
    return str(checkpoint_dir)
