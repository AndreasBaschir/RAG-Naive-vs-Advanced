from __future__ import annotations

from typing import Generator

import chromadb
import ollama
import torch
from sentence_transformers import SentenceTransformer

from datasets_registry import CHROMA_PATH, active, active_key

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
LLM_MODEL = "qwen3:8b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_SYSTEM_PROMPT = (
    "You are a precise, factual assistant. "
    "Answer the user's question using ONLY the information in the provided context. "
    "If the context does not contain enough information to answer, reply with exactly: "
    "\"The provided context does not contain enough information to answer this question.\""
)

_embedder: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None
_collection_key: str | None = None


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

    for part in ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"},
        ],
        stream=True,
    ):
        yield part["message"]["content"]

def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    return _embedder


def _get_collection() -> chromadb.Collection:
    global _collection, _collection_key
    if _collection is not None and _collection_key == active_key():
        return _collection
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    _collection = client.get_or_create_collection(
        active().collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    _collection_key = active_key()
    return _collection
