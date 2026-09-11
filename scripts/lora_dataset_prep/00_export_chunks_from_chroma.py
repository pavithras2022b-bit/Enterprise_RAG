"""
Step 0: Export chunks from ChromaDB into data/labeled/chunks_export.jsonl.

VERIFIED AGAINST vector_db/chroma_store.py and configs/vector_db.yaml:
- PERSIST_DIRECTORY = "data/chroma_store", COLLECTION_NAME = "enterprise_docs"
  (copied from vector_db.yaml's vector_db.chroma block).
- Collection is created with metadata={"hnsw:space": "cosine"} -- doesn't
  affect export, just confirms this is the same collection VectorStore uses.
- Critically: VectorStore.add() calls collection.upsert(ids=[chunk.chunk_id],
  ...) -- the Chroma record id IS chunk.chunk_id, there is no separate
  candidate_id assigned at this layer. The candidate_id seen in
  prompt_builder.py / your API output is this same value, renamed only
  by attribute at a later schema (RankedCandidate.candidate_id) -- the
  underlying string is identical, so exporting Chroma's record id as
  "candidate_id" below is correct, not a guess.
- Metadata: to_metadata_dict() (on EmbeddedChunk) isn't shown to me, so
  rather than guess its exact keys again, this exports whatever metadata
  dict Chroma actually returns, unfiltered. Spot-check the output to
  confirm it has the department/document_category/title fields you expect
  for later hard-negative mining -- if key names differ from what you
  expect, tell me and I'll adjust downstream scripts, not this one.

Usage:
    python 00_export_chunks_from_chroma.py --out data/labeled/chunks_export.jsonl
"""

import argparse
import json
import os

# Verified from configs/vector_db.yaml -> vector_db.chroma
PERSIST_DIRECTORY = "data/chroma_store"
COLLECTION_NAME = "enterprise_docs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/labeled/chunks_export.jsonl")
    ap.add_argument("--persist-dir", default=PERSIST_DIRECTORY)
    ap.add_argument("--collection", default=COLLECTION_NAME)
    ap.add_argument("--batch-size", type=int, default=500)
    args = ap.parse_args()

    try:
        import chromadb
    except ImportError:
        print("chromadb not installed in this environment. "
              "Run this from your project's .venv where the pipeline itself runs.")
        return

    # Mirrors VectorStore.__init__'s client construction exactly
    # (anonymized_telemetry=False), so this reads the same on-disk
    # store your pipeline writes to -- not a separate/divergent client.
    from chromadb.config import Settings
    settings = Settings(anonymized_telemetry=False)
    client = chromadb.PersistentClient(path=args.persist_dir, settings=settings)

    try:
        collection = client.get_collection(args.collection)
    except Exception as e:
        available = [c.name for c in client.list_collections()]
        print(f"Could not open collection '{args.collection}': {e}")
        print(f"Collections that DO exist at '{args.persist_dir}': {available}")
        print("Update --collection (or COLLECTION_NAME) to one of the names above and retry.")
        return

    total = collection.count()
    print(f"Collection '{args.collection}' has {total} records. Exporting...")

    written = 0
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as out_f:
        offset = 0
        while offset < total:
            batch = collection.get(
                include=["documents", "metadatas"],
                limit=args.batch_size,
                offset=offset,
            )
            ids = batch.get("ids", [])
            docs = batch.get("documents", [])
            metas = batch.get("metadatas", [])

            for _id, text, meta in zip(ids, docs, metas):
                # _id here is chunk.chunk_id as written by VectorStore.add() --
                # same underlying value as candidate_id downstream, per the
                # verification note above.
                row = {
                    "candidate_id": _id,
                    "text": text,
                    "metadata": dict(meta) if meta else {},
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

            offset += args.batch_size
            print(f"  exported {min(offset, total)}/{total}")

    print(f"\nDone. Wrote {written} chunks to {args.out}")
    if written > 0:
        print("Spot-check a line before trusting it -- especially whether 'candidate_id' "
              "matches the exact format seen in your /api/v1/query source_candidates "
              "(e.g. docx_8a299618020a5d5f_chunk_0000).")


if __name__ == "__main__":
    main()