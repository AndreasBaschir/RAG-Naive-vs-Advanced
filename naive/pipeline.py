from __future__ import annotations

import pathlib
from typing import Generator

import chromadb
import ollama
import torch
from sentence_transformers import SentenceTransformer

ROOT = pathlib.Path(__file__).parent.parent
CHROMA_PATH = ROOT / "data" / "chroma"
COLLECTION_NAME = "squad"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
LLM_MODEL = "qwen3:8b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_embedder: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None


def retrieve(query: str) -> list[dict]:
    """
    Embed the query with all-MiniLM-L6-v2 and return the top-K chunks
    from ChromaDB by cosine similarity.

    Each dict has:
        "text":   str   — chunk content
        "source": str   — SQuAD article title
        "score":  float — cosine similarity (0-1)
    """
    col = _get_collection()
    if col.count() == 0:
        return []

    vec = _get_embedder().encode(query).tolist()
    results = col.query(
        query_embeddings=[vec],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"]
    metas = results["metadatas"]
    dists = results["distances"]
    if not docs or not metas or not dists:
        return []

    return [
        {
            "text": text,
            "source": meta.get("title", ""),
            "score": round(1.0 - dist, 4),
        }
        for text, meta, dist in zip(docs[0], metas[0], dists[0])
    ]


def stream(query: str, chunks: list[dict]) -> Generator[str, None, None]:
    context = "\n\n".join(c["text"] for c in chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    for part in ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ):
        yield part["message"]["content"]

def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    return _embedder


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is not None:
        return _collection
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    _collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection
