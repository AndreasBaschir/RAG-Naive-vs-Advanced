"""
Hybrid retrieval components for the Advanced RAG pipeline.

Dense (ChromaDB cosine similarity) + Sparse (BM25) + RRF + Cross-encoder reranking.
"""

from __future__ import annotations

import pathlib

import chromadb
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = pathlib.Path(__file__).parent.parent
CHROMA_PATH = ROOT / "data" / "chroma"
COLLECTION_NAME = "squad"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_embedder: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None
_reranker: CrossEncoder | None = None
_bm25_index: BM25Okapi | None = None
_bm25_corpus: list[dict] | None = None


def encode(texts: list[str]) -> list[list[float]]:
    """Batch-encode a list of texts into embedding vectors."""
    return _get_embedder().encode(texts, batch_size=len(texts)).tolist()


def dense_retrieve(query: str, n: int = 20) -> list[dict]:
    """Top-n chunks via dense cosine similarity in ChromaDB."""
    return dense_retrieve_vec(_get_embedder().encode(query).tolist(), n)


def dense_retrieve_vec(vec: list[float], n: int = 20) -> list[dict]:
    """Top-n chunks using a pre-computed embedding vector (skips encoding)."""
    col = _get_collection()
    if col.count() == 0:
        return []

    results = col.query(
        query_embeddings=[vec],
        n_results=min(n, col.count()),
        include=["documents", "metadatas", "distances"],
    )

    docs, metas, dists = results["documents"], results["metadatas"], results["distances"]
    if not docs or not metas or not dists:
        return []

    return [
        {
            "text": text,
            "source": meta.get("title", ""),
            "dense_score": round(1.0 - dist, 4),
        }
        for text, meta, dist in zip(docs[0], metas[0], dists[0])
    ]


def bm25_retrieve(query: str, n: int = 20) -> list[dict]:
    """Top-n chunks via BM25 keyword scoring over the full corpus."""
    index, corpus = _get_bm25()
    tokenized_query = query.lower().split()
    scores = index.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]

    return [
        {
            "text": corpus[i]["text"],
            "source": corpus[i]["title"],
            "bm25_score": round(float(scores[i]), 4),
        }
        for i in top_indices
        if scores[i] > 0
    ]


def rrf_merge(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion across multiple ranked result lists.

    Score formula: sum(1 / (k + rank)) across all lists a document appears in.
    Deduplication is by text content.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = doc["text"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in doc_map:
                doc_map[key] = doc.copy()

    merged = sorted(doc_map.values(), key=lambda d: scores[d["text"]], reverse=True)
    for doc in merged:
        doc["rrf_score"] = round(scores[doc["text"]], 6)
    return merged


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Cross-encoder reranking: scores each (query, passage) pair independently,
    providing more accurate relevance than bi-encoder similarity alone.
    """
    if not candidates:
        return []

    cross_enc = _get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = cross_enc.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = round(float(s), 4)
        c["score"] = c["rerank_score"]

    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


# --- Lazy singletons ---

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


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL, device=DEVICE)
    return _reranker


def _get_bm25() -> tuple[BM25Okapi, list[dict]]:
    global _bm25_index, _bm25_corpus
    if _bm25_index is not None and _bm25_corpus is not None:
        return _bm25_index, _bm25_corpus

    col = _get_collection()
    # Load full corpus from ChromaDB to build the BM25 index.
    # One-time cost at first query; cached for the process lifetime.
    all_docs = col.get(include=["documents", "metadatas"])

    _bm25_corpus = [
        {"text": text, "title": meta.get("title", "")}
        for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]

    tokenized = [doc["text"].lower().split() for doc in _bm25_corpus]
    _bm25_index = BM25Okapi(tokenized)
    return _bm25_index, _bm25_corpus
