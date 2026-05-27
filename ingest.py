"""
Ingest SQuAD contexts into ChromaDB.

Ambele pipeline-uri (naive și advanced) citesc din aceeași colecție.
Diferența dintre ele stă în arhitectura de retrieval, nu în date.

Run once:
    python ingest.py
"""

from __future__ import annotations

import pathlib
from typing import Any, cast

import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

ROOT = pathlib.Path(__file__).parent
CHROMA_PATH = ROOT / "data" / "chroma"
COLLECTION_NAME = "squad"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 256


def ingest() -> None:
    print("Loading SQuAD dataset...")
    dataset = load_dataset("rajpurkar/squad", split="train")

    # Deduplicate: același paragraf apare de mai multe ori (câte o întrebare per rând)
    seen: set[str] = set()
    contexts: list[dict] = []
    for _row in dataset:
        row = cast(dict[str, Any], _row)
        ctx = row["context"]
        if ctx not in seen:
            seen.add(ctx)
            contexts.append({"text": ctx, "title": row["title"]})

    print(f"Found {len(contexts)} unique contexts.")

    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print("Connecting to ChromaDB...")
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0:
        print(f"Collection already contains {collection.count()} documents. Skipping.")
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

    print(f"Done. {collection.count()} documents stored in ChromaDB.")


if __name__ == "__main__":
    ingest()
