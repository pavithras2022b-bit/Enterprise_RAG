# -*- coding: utf-8 -*-
"""
LoRA fine-tuning script for Qwen2.5-3B-Instruct on data/labeled/lora_train.jsonl /
lora_val.jsonl (format: {"prompt": ..., "completion": ...} per line).

Sized for CPU-only training:
  - bfloat16 base weights (~6GB instead of ~12GB in fp32)
  - Small per-device batch size + gradient accumulation to keep peak memory low
  - LoRA adapters only on attention projections (q_proj, k_proj, v_proj, o_proj)
    to keep trainable parameter count small
  - No device_map="auto" / no bitsandbytes (bnb quantization needs a GPU build);
    this is plain CPU bf16 + peft, which works but will be slow (expect hours,
    not minutes, for a few epochs over a few hundred examples on CPU).

Usage:
    python train_lora.py \
        --base-model Qwen/Qwen2.5-3B-Instruct \
        --train-file data/labeled/lora_train.jsonl \
        --val-file data/labeled/lora_val.jsonl \
        --output-dir outputs/qwen2.5-3b-lora-ragqa \
        --epochs 3 --batch-size 1 --grad-accum 16 --lr 2e-4
"""
import argparse
import json
import os

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_dataset(rows, tokenizer, max_length):
    """Concatenate prompt + completion, mask the prompt tokens in the labels so
    loss is only computed on the completion (standard causal-LM SFT setup)."""

    def _tokenize(example):
        prompt = example["prompt"]
        completion = example["completion"]
        # Ensure a separating space/newline so the completion tokenizes cleanly.
        full_text = prompt + completion + tokenizer.eos_token

        full = tokenizer(full_text, truncation=True, max_length=max_length)
        prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"]
        prompt_len = len(prompt_ids)

        labels = list(full["input_ids"])
        # Mask out the prompt portion so the model isn't trained to "predict" the
        # question/context, only the answer that follows it.
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        full["labels"] = labels
        return full

    ds = Dataset.from_list(rows)
    ds = ds.map(_tokenize, remove_columns=ds.column_names)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--train-file", default="data/labeled/lora_train.jsonl")
    ap.add_argument("--val-file", default="data/labeled/lora_val.jsonl")
    ap.add_argument("--output-dir", default="outputs/qwen2.5-3b-lora-ragqa")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1,
                     help="Per-device batch size. Keep at 1 for CPU training.")
    ap.add_argument("--grad-accum", type=int, default=16,
                     help="Gradient accumulation steps. Effective batch size = "
                          "batch-size * grad-accum.")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--logging-steps", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading base model {args.base_model} in bfloat16 (CPU)...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False  # required for gradient checkpointing / training

    print("Applying LoRA adapters (q_proj, k_proj, v_proj, o_proj)...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading training data from {args.train_file} and {args.val_file}...")
    train_rows = load_jsonl(args.train_file)
    val_rows = load_jsonl(args.val_file)
    print(f"  {len(train_rows)} train examples, {len(val_rows)} val examples")

    train_ds = build_dataset(train_rows, tokenizer, args.max_length)
    val_ds = build_dataset(val_rows, tokenizer, args.max_length)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=False,          # CPU can't use the bf16 autocast training path reliably;
        use_cpu=True,        # weights are already bf16, this just avoids CUDA calls
        report_to=[],
        gradient_checkpointing=True,
        weight_decay=0.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    print("Starting training. This will be slow on CPU — expect it to take a "
          "while for a 3B model even with LoRA; consider running overnight or "
          "in the background.")
    trainer.train()

    print(f"Saving final LoRA adapter to {args.output_dir}/final_adapter")
    model.save_pretrained(os.path.join(args.output_dir, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_adapter"))

    print("Done. This adapter directory is what llm/lora_manager.py's "
          "register_adapter() expects as 'the output_dir of a prior peft training run'.")


if __name__ == "__main__":
    main()
