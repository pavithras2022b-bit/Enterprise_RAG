"""
Step 3: Export approved candidates into a LoRA training file.

VERIFIED AGAINST llm/prompt_builder.py:
- SYSTEM_INSTRUCTION below is copied verbatim from
  PromptBuilder._DEFAULT_SYSTEM_INSTRUCTION -- if you ever change the
  production system instruction, update it here too, or the LoRA will
  train against a prompt the live system no longer sends.
- Evidence block format matches build_prompt() exactly:
  "[candidate_id] text" per item, joined with "\n\n".
- Final prompt shape matches build_prompt() exactly: system instruction,
  blank line, "Context:\n<evidence_block>", blank line,
  "Question: <query>\nAnswer:".

CITATION HANDLING:
Answers from step 1 do NOT contain citation brackets (a 3B model
copying a hash-like id verbatim is unreliable). This script appends
the citation deterministically: " [candidate_id]" after the answer
text for positive examples. This guarantees every positive training
example has a citation in the exact bracket format extract_citations()
parses -- directly targets the citations: {} symptom seen in your
failed "annual leave" query, where the model answered wrong AND cited
nothing.

Negative ("insufficient context") examples get NO citation appended,
matching correct real-system behavior: no valid answer -> no citation
to extract.

MULTI-CHUNK CONTEXT (realistic retrieval simulation):
Your failed query returned 4 source_candidates, not 1 -- production
context is multi-chunk. If you pass --chunks-pool, this script mixes
in a configurable number of distractor chunks alongside the gold chunk
per training example, at a random position (your real failure case had
the gold chunk at fused_score 1.0, i.e. first, and still failed -- so
don't assume gold is always first). Without --chunks-pool it falls
back to single-chunk context (still useful for an early smoke test,
but less representative of production).

Usage:
    python 03_export_training_data.py --candidates candidates.jsonl \
        --out-train train.jsonl --out-val val.jsonl \
        --chunks-pool data/chunks/chunks.jsonl --num-distractors 2 \
        [--negatives-file negatives.jsonl]
"""

import argparse
import json
import os
import random

# Copied verbatim from llm/prompt_builder.py::PromptBuilder._DEFAULT_SYSTEM_INSTRUCTION.
# Keep in sync by hand -- do not paraphrase.
SYSTEM_INSTRUCTION = (
    "Using only the context below, answer the question. Each piece of context "
    "is labeled with an id like [pdf_ff8f53d904329460_chunk_0000]. After each "
    "fact you state, copy that exact bracketed id from the context -- do not "
    "write the literal word source_id. If the context does not contain enough "
    "information to answer, say so rather than guessing."
)


def build_evidence_block(items):
    """items: list of {"candidate_id": ..., "text": ...}, matches
    PromptBuilder.build_prompt()'s "\n\n".join(f"[{id}] {text}" ...)."""
    return "\n\n".join(f"[{it['candidate_id']}] {it['text']}" for it in items)


def build_prompt(query, items):
    evidence_block = build_evidence_block(items) if items else \
        "(No relevant evidence was found for this query.)"
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Context:\n{evidence_block}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out-train", default="train.jsonl")
    ap.add_argument("--out-val", default="val.jsonl")
    ap.add_argument("--negatives-file", default=None,
                     help="Optional JSONL of {question, candidate_id, text} where "
                          "text does NOT answer question")
    ap.add_argument("--chunks-pool", default=None,
                     help="Optional full chunks JSONL to sample distractors from "
                          "(same file used in step 1)")
    ap.add_argument("--num-distractors", type=int, default=2)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    rows = load_jsonl(args.candidates)
    approved = [r for r in rows if r.get("status") == "approved"]
    if not approved:
        print("No approved rows found. Run 02_review_candidates.py first.")
        return

    pool = []
    if args.chunks_pool:
        pool = load_jsonl(args.chunks_pool)
        print(f"Loaded {len(pool)} chunks for distractor sampling.")
    else:
        print("No --chunks-pool given -- falling back to single-chunk context "
              "(fine for a smoke test, less representative of production).")

    examples = []
    for r in approved:
        gold = {"candidate_id": r["candidate_id"], "text": r["text"]}

        if pool and args.num_distractors > 0:
            others = [c for c in pool if c.get("candidate_id") != r["candidate_id"]]
            k = min(args.num_distractors, len(others))
            distractors = random.sample(others, k) if k > 0 else []
            items = [{"candidate_id": d["candidate_id"], "text": d.get("text", "")}
                      for d in distractors] + [gold]
            random.shuffle(items)
        else:
            items = [gold]

        prompt = build_prompt(r["question"], items)
        completion = f" {r['answer'].strip()} [{r['candidate_id']}]"
        examples.append({"prompt": prompt, "completion": completion})

    if args.negatives_file:
        neg_rows = load_jsonl(args.negatives_file)
        for neg in neg_rows:
            items = [{"candidate_id": neg["candidate_id"], "text": neg["text"]}]
            if pool and args.num_distractors > 0:
                others = [c for c in pool if c.get("candidate_id") != neg["candidate_id"]]
                k = min(args.num_distractors, len(others))
                items += [{"candidate_id": d["candidate_id"], "text": d.get("text", "")}
                          for d in random.sample(others, k)] if k > 0 else []
                random.shuffle(items)
            prompt = build_prompt(neg["question"], items)
            # No citation appended -- correct behavior for a genuinely
            # unanswerable question is no citation, matching extract_citations
            # finding nothing to extract.
            completion = " The provided context does not contain enough information to answer this question."
            examples.append({"prompt": prompt, "completion": completion})
        print(f"Added {len(neg_rows)} insufficient-context negative examples.")
        frac = len(neg_rows) / len(examples)
        if frac < 0.08 or frac > 0.20:
            print(f"NOTE: negatives are {frac:.0%} of the dataset -- roadmap target was ~10-15%.")
    else:
        print("WARNING: no --negatives-file provided. Training without negatives risks "
              "losing the 'correctly decline when context is insufficient' behavior.")

    random.shuffle(examples)
    split = int(len(examples) * (1 - args.val_fraction))
    train, val = examples[:split], examples[split:]

    for out_path in (args.out_train, args.out_val):
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    with open(args.out_train, "w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(args.out_val, "w", encoding="utf-8") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(train)} train examples -> {args.out_train}")
    print(f"Wrote {len(val)} val examples -> {args.out_val}")
    if len(train) < 85:
        print(f"NOTE: {len(train)} training examples is below the roadmap's 100-300 target.")


if __name__ == "__main__":
    main()
    #Get-Service -Name "Oracle*"
    #Stop-Service -Name "<exact-service-name>" #raa
    #Stop-Process -Name "chrome" -Force
    #Get-CimInstance Win32_OperatingSystem | Select-Object FreePhysicalMemory, TotalVisibleMemorySize
    #wmic pagefileset where name="C:\\pagefile.sys" set InitialSize=16384,MaximumSize=16384 #raa
#Stop-Service -Name "OracleServiceXE" -Force
#Stop-Service -Name "OracleOraDB21Home1TNSListener" -Force
#Stop-Service -Name "OracleOraDB21Home1MTSRecoveryService" -Force
