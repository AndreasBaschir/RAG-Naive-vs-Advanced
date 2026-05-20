from typing import Generator


def retrieve(query: str) -> list[dict]:
    """
    Return context chunks for the query.

    Each dict should have at minimum:
        "text":   str   — the chunk content
        "source": str   — document name / ID  (shown in UI)
        "score":  float — retrieval score     (shown in UI)

    You can add extra keys (e.g. "rerank_score", "sub_query") — the UI
    will display them automatically if present.

    Return [] to hide the context panel entirely.
    """
    return []


def stream(query: str, chunks: list[dict]) -> Generator[str, None, None]:
    """
    Yield text tokens for the LLM response.

    `chunks` is the output of retrieve() above — pass them as context
    to your language model.  The UI accumulates tokens as they arrive.
    """
    yield "Advanced RAG pipeline not yet implemented."
