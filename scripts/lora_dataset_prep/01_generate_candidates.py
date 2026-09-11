"""
Step 1: Candidate answer generation from your existing chunks.

VERIFIED AGAINST REAL EXPORTED DATA (data/labeled/chunks_export.jsonl,
149 records) AND data/raw/metadata.json (140-doc manifest):

- Chunk metadata is actually {"source_path": "data\\raw\\documents\\
  document_001.pdf", "source_document_id": "...", "document_type":
  "unknown"} -- NO department/title/document_category at chunk level.
  This replaces my earlier wrong assumption that those fields lived
  directly on the chunk.
- The manifest at data/raw/metadata.json has filename -> title/
  department/document_category. source_path's basename
  ("document_001.pdf") matches the manifest's "filename" field
  exactly, so this script loads the manifest once and joins on that.
- Chunk text has a mojibake artifact ("â€”" where an em dash should be,
  e.g. "Meridian Dynamics Inc. â€” Internal Document") -- a UTF-8/
  Windows-1252 double-encoding from ingestion, visible in your actual
  exported chunk and matching the "â" seen in the raw API output
  earlier. CLEAN_MOJIBAKE below fixes it before it reaches the LLM
  and before it gets baked into training data. This is a training-data
  fix only -- the real bug is upstream in ingestion/loaders.py or
  preprocessing/cleaner.py and should be fixed there too eventually.

INPUT : data/labeled/chunks_export.jsonl (from step 0) and
        data/raw/metadata.json (the document manifest).

OUTPUT: candidates.jsonl -- draft (question, answer) pairs, answer text
        ONLY (no citation marker yet -- appended deterministically in
        step 3, not left to the draft LLM to copy correctly).

Usage:
    python 01_generate_candidates.py \
        --chunks data/labeled/chunks_export.jsonl \
        --manifest data/raw/metadata.json \
        --out data/labeled/lora_candidates.jsonl \
        --model llama3.2:3b --questions-per-chunk 3 --limit 5
"""

import argparse
import json
import ntpath
import os
import sys
import time
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"

# The mojibake in your chunks is the classic pattern: original UTF-8
# text got decoded as cp1252 somewhere in ingestion (see
# ingestion/loaders.py / preprocessing/cleaner.py), producing
# multi-character garbage like "â€”" in place of a single "—".
# Fix: reverse the mis-decode by re-encoding as cp1252 (recovering the
# original UTF-8 bytes) then decoding those bytes correctly as UTF-8.
# This is a general fix, not a lookup table -- it self-verified against
# your actual pasted text: "â€”" -> "—" correctly, without needing to
# guess every garbled variant by hand (a hand-typed lookup table is
# exactly what silently failed in the previous version of this script).
def clean_mojibake(text: str) -> str:
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        # Text wasn't actually mojibake'd this way -- leave it alone
        # rather than corrupt already-clean text.
        return text


def load_manifest(path: str) -> dict:
    """Returns {filename: {title, department, document_category, ...}}."""
    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return {d["filename"]: d for d in docs if "filename" in d}


def manifest_lookup(manifest: dict, source_path: str) -> dict:
    """source_path is a Windows-style path like
    'data\\raw\\documents\\document_001.pdf' -- use ntpath so this
    works whether this script runs on Windows or not."""
    filename = ntpath.basename(source_path) if source_path else None
    return manifest.get(filename, {})


PROMPT_TEMPLATE = """You are drafting evaluation questions for an internal enterprise policy assistant.

Given the passage below, write {n} distinct question-answer pairs that:
- Can be fully answered using ONLY this passage (no outside knowledge).
- Sound like realistic employee questions (not "what does this passage say").
- Cover different facts in the passage where possible.
- Have a concise, factual answer (1-3 sentences), grounded strictly in the passage.
- Do NOT include any bracketed citation markers in the answer -- plain prose only.

Passage (document: {doc_title}, department: {department}):
\"\"\"
{text}
\"\"\"

Respond ONLY with a JSON array, no preamble, no markdown fences:
[
  {{"question": "...", "answer": "..."}},
  ...
]
"""


def call_ollama(model: str, prompt: str, timeout: int = 120) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.4},
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("response", "")


def parse_candidates(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [d for d in data if "question" in d and "answer" in d]
    except json.JSONDecodeError:
        pass
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="Path to data/labeled/chunks_export.jsonl")
    ap.add_argument("--manifest", required=True, help="Path to data/raw/metadata.json")
    ap.add_argument("--out", default="candidates.jsonl")
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--questions-per-chunk", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="Cap chunks for a test run")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    print(f"Loaded manifest: {len(manifest)} documents.")

    chunks = []
    with open(args.chunks, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    if args.limit:
        chunks = chunks[: args.limit]

    print(f"Loaded {len(chunks)} chunks. Generating candidates with {args.model}...")

    written = 0
    unmatched = 0
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as out_f:
        for i, chunk in enumerate(chunks):
            candidate_id = chunk.get("candidate_id")
            if not candidate_id:
                print(f"  [skip] chunk {i} missing 'candidate_id' field", file=sys.stderr)
                continue
            text = clean_mojibake(chunk.get("text", ""))
            if not text.strip():
                continue

            chunk_meta = chunk.get("metadata", {}) or {}
            doc_meta = manifest_lookup(manifest, chunk_meta.get("source_path", ""))
            if not doc_meta:
                unmatched += 1

            doc_title = doc_meta.get("title", "unknown")
            department = doc_meta.get("department", "unknown")

            # Merge chunk-level + manifest metadata into one dict for
            # downstream steps (hard-negative mining in the reranker
            # stage can use department/document_category from here).
            merged_metadata = {**chunk_meta, **doc_meta}

            prompt = PROMPT_TEMPLATE.format(
                n=args.questions_per_chunk, doc_title=doc_title,
                department=department, text=text,
            )

            try:
                raw = call_ollama(args.model, prompt)
            except Exception as e:
                print(f"  [{i+1}/{len(chunks)}] {candidate_id}: FAILED ({e})", file=sys.stderr)
                continue

            candidates = parse_candidates(raw)
            for c in candidates:
                row = {
                    "candidate_id": candidate_id,
                    "text": text,
                    "metadata": merged_metadata,
                    "question": c["question"].strip(),
                    "answer": c["answer"].strip(),
                    "status": "pending",
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

            print(f"  [{i+1}/{len(chunks)}] {candidate_id} ({doc_title}): {len(candidates)} candidates")
            time.sleep(0.1)

    print(f"\nDone. Wrote {written} candidate pairs to {args.out}")
    if unmatched:
        print(f"NOTE: {unmatched} chunk(s) had no manifest match (doc_title/department will be 'unknown'). "
              "Check that source_path filenames line up with manifest 'filename' entries.")
    print("Next: run 02_review_candidates.py to approve/edit/reject before training.")


if __name__ == "__main__":
    main()