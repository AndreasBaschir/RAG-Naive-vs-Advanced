#!/usr/bin/env python3
"""
Ingest a dataset's corpus into ChromaDB.

Both pipeline variants (naive and advanced) read from the same per-dataset
collection; the difference lies in retrieval architecture, not in data.

Run once per dataset:
    python ingest.py                  # squad (default)
    python ingest.py --dataset docred
"""

from __future__ import annotations

import argparse

import chromadb
from sentence_transformers import SentenceTransformer

from datasets_registry import CHROMA_PATH, active, available, set_active

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 256


def ingest(dataset: str) -> None:
    """Embed and store the full corpus of *dataset* into ChromaDB.

    Skips ingestion if the target collection already contains documents.
    Embeddings are computed in batches of ``BATCH_SIZE`` to bound memory usage.

    :param dataset: registered dataset key (e.g. ``"squad"``, ``"docred"``)
    """
    set_active(dataset)
    spec = active()

    print(f"Loading '{dataset}' corpus...")
    contexts = spec.load_corpus()
    print(f"Found {len(contexts)} unique documents.")

    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print("Connecting to ChromaDB...")
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        spec.collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0:
        print(f"Collection '{spec.collection_name}' already contains "
              f"{collection.count()} documents. Skipping.")
        return

    print("Embedding and ingesting...")
    for i in range(0, len(contexts), BATCH_SIZE):
        batch = contexts[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids=[f"ctx_{i + j}" for j in range(len(batch))],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"title": c["title"]} for c in batch],
        )
        print(f"  {i + len(batch)}/{len(contexts)}")

    print(f"Done. {collection.count()} documents stored in collection "
          f"'{spec.collection_name}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=available(), default="squad",
                        help="Which dataset's corpus to ingest (default: squad)")
    args = parser.parse_args()
    ingest(args.dataset)
