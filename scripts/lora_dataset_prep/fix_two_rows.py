# -*- coding: utf-8 -*-
"""
One-off fixer for two known-bad approved answers in lora_candidates.jsonl:

1. candidate_id pdf_09933584f8b49f70_chunk_0000
   Q: "What is the name of the department that maintains the employee roster?"
   Old (wrong) A: "The Information Technology department"
   The source text says the roster is "maintained in Workday HRMS by the
   Human Resources department" -- HR maintains it, not IT.

2. candidate_id docx_0d5991863da15ac6_chunk_0000
   Q: "How many people does Meera Krishnan report to in the Information
       Security department?"
   Old (wrong) A: implies Meera reports to 3 people (inverted direction)
   The source text says those 3 people report TO Meera, not the other way
   around. Meera herself reports to Vikram Suresh (outside this department).

Usage:
    python fix_two_rows.py --file data\\labeled\\lora_candidates.jsonl
"""
import argparse
import json


FIXES = {
    "pdf_09933584f8b49f70_chunk_0000": {
        "match_question": "What is the name of the department that maintains the employee roster?",
        "new_answer": "The Human Resources department maintains the employee roster (in Workday HRMS) for the Information Technology department.",
    },
    "docx_0d5991863da15ac6_chunk_0000": {
        "match_question": "How many people does Meera Krishnan report to in the Information Security department?",
        "new_question": "How many people report directly to Meera Krishnan in the Information Security department?",
        "new_answer": "Three people report directly to Meera Krishnan in the Information Security department: Anjali Krishnamurthy, Tarun Sethi, and Ishita Banerjee. Meera Krishnan herself reports to Vikram Suresh, Chief Technology Officer.",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/labeled/lora_candidates.jsonl")
    args = ap.parse_args()

    rows = []
    fixed_count = 0
    with open(args.file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj.get("candidate_id")
            if cid in FIXES:
                fix = FIXES[cid]
                if obj.get("question") == fix["match_question"]:
                    if "new_question" in fix:
                        print(f"Fixing question for {cid}")
                        obj["question"] = fix["new_question"]
                    print(f"Fixing answer for {cid}")
                    obj["answer"] = fix["new_answer"]
                    fixed_count += 1
                else:
                    print(f"WARNING: candidate_id {cid} found but question text "
                          f"did not match expected -- left unchanged. "
                          f"Found question: {obj.get('question')!r}")
            rows.append(obj)

    with open(args.file, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Done. Fixed {fixed_count} of {len(FIXES)} targeted rows in {args.file}")


if __name__ == "__main__":
    main()
