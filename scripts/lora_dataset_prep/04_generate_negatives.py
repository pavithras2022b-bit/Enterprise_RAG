"""
Step 4 (LoRA dataset prep) - Generate negative training examples.

Pairs each of a sample of approved (question, own_department) pairs with the
chunk text of an APPROVED candidate from a DIFFERENT department. Since the
paired chunk is drawn from an unrelated department, it is guaranteed by
construction not to answer the question -- no manual judgment calls needed.

Output matches exactly what 03_export_training_data.py's --negatives-file
expects: JSONL rows of {question, candidate_id, text}.

Usage:
    python 04_generate_negatives.py \
        --candidates data/labeled/lora_candidates.jsonl \
        --out data/labeled/negatives.jsonl \
        --count 40 --seed 42
"""

import argparse
import json
import random
from collections import defaultdict


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
    ap.add_argument("--candidates", required=True, help="Path to data/labeled/lora_candidates.jsonl")
    ap.add_argument("--out", default="data/labeled/negatives.jsonl")
    ap.add_argument("--count", type=int, default=40, help="Number of negative examples to generate")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    rows = load_jsonl(args.candidates)
    approved = [r for r in rows if r.get("status") == "approved" and r.get("question")]
    if not approved:
        raise SystemExit(f"No approved rows with a question found in {args.candidates}")

    # Unique (candidate_id, text, department) chunks, for picking mismatched-department negatives.
    chunks_by_dept = defaultdict(dict)  # dept -> {candidate_id: text}
    for r in approved:
        dept = (r.get("metadata") or {}).get("department", "Unknown")
        cid = r.get("candidate_id")
        text = r.get("text")
        if cid and text:
            chunks_by_dept[dept][cid] = text

    all_depts = list(chunks_by_dept.keys())
    if len(all_depts) < 2:
        raise SystemExit(
            "Only one department found across approved candidates -- cannot build "
            "guaranteed-mismatched negatives. Check --candidates content."
        )

    # Sample distinct questions to pair with negatives, up to --count.
    sample_size = min(args.count, len(approved))
    sampled_rows = random.sample(approved, sample_size)

    negatives = []
    skipped = 0
    for row in sampled_rows:
        own_dept = (row.get("metadata") or {}).get("department", "Unknown")
        other_depts = [d for d in all_depts if d != own_dept]
        if not other_depts:
            skipped += 1
            continue
        neg_dept = random.choice(other_depts)
        neg_cid, neg_text = random.choice(list(chunks_by_dept[neg_dept].items()))
        negatives.append({
            "question": row["question"],
            "candidate_id": neg_cid,
            "text": neg_text,
        })

    with open(args.out, "w", encoding="utf-8") as f:
        for neg in negatives:
            f.write(json.dumps(neg, ensure_ascii=False) + "\n")

    frac_of_approved = len(negatives) / len(approved)
    print(f"Approved candidates available: {len(approved)}")
    print(f"Departments represented: {len(all_depts)}")
    print(f"Wrote {len(negatives)} negative examples -> {args.out}")
    if skipped:
        print(f"Skipped {skipped} rows (no other department available to pair against).")
    print(f"Negatives as a fraction of approved positives: {frac_of_approved:.1%}")
    print("(Combine with 03_export_training_data.py's --negatives-file to hit the ~10-15% roadmap target relative to the FINAL exported dataset, not this raw approved count.)")


if __name__ == "__main__":
    main()