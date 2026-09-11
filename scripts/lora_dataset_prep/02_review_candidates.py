"""
Step 2: Human verification pass over candidates.jsonl.

Same review workflow as before, updated only for the verified field
names (candidate_id, metadata.department/title instead of the earlier
chunk_id/source_doc guesses).

Usage:
    python 02_review_candidates.py --file candidates.jsonl
"""

import argparse
import json


def load(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    args = ap.parse_args()

    rows = load(args.file)
    pending_idx = [i for i, r in enumerate(rows) if r.get("status") == "pending"]

    if not pending_idx:
        approved = sum(1 for r in rows if r.get("status") == "approved")
        print(f"No pending rows. {approved} approved so far out of {len(rows)} total.")
        return

    print(f"{len(pending_idx)} pending rows to review. 'q' to stop anytime -- progress saves as you go.\n")

    for count, i in enumerate(pending_idx, 1):
        r = rows[i]
        meta = r.get("metadata", {}) or {}
        print("=" * 70)
        print(f"[{count}/{len(pending_idx)}]  candidate_id: {r['candidate_id']}")
        print(f"doc: {meta.get('title', meta.get('document_id', 'unknown'))}  "
              f"dept: {meta.get('department', 'unknown')}  "
              f"category: {meta.get('document_category', 'unknown')}")
        print("-" * 70)
        print(f"TEXT:\n{r['text'][:500]}{'...' if len(r['text']) > 500 else ''}")
        print(f"\nQ: {r['question']}")
        print(f"A: {r['answer']}")
        print("-" * 70)
        choice = input("[a]pprove / [e]dit / [r]eject / [s]kip / [q]uit: ").strip().lower()

        if choice == "q":
            save(args.file, rows)
            print("Saved. Resume anytime with the same command.")
            return
        elif choice == "a":
            r["status"] = "approved"
        elif choice == "e":
            new_q = input("New question (blank = keep): ").strip()
            new_a = input("New answer (blank = keep): ").strip()
            if new_q:
                r["question"] = new_q
            if new_a:
                r["answer"] = new_a
            r["status"] = "approved"
        elif choice == "r":
            r["status"] = "rejected"
        else:
            print("(skipped)")
            continue

        save(args.file, rows)

    approved = sum(1 for r in rows if r.get("status") == "approved")
    rejected = sum(1 for r in rows if r.get("status") == "rejected")
    print(f"\nReview complete for this pass. Approved: {approved}, Rejected: {rejected}, Total: {len(rows)}")
    print("Next: run 03_export_training_data.py to build the LoRA training file.")


if __name__ == "__main__":
    main()
