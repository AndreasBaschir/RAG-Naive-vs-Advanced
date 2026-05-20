import os
import queue
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import streamlit as st

st.set_page_config(
    page_title="RAG Comparison",
    layout="wide",
    page_icon="⚡",
)

st.markdown(
    """
    <style>
    .pipeline-title { font-size: 1.25rem; font-weight: 700; margin-bottom: 2px; }
    .pipeline-subtitle { font-size: 0.8rem; color: #888; margin-bottom: 0.75rem; }
    .metric-row { display: flex; gap: 1.5rem; margin-bottom: 0.5rem; }
    .metric-item { font-size: 0.78rem; color: #555; }
    .metric-value { font-weight: 600; color: #222; }
    .chunk-card {
        background: #f7f8fa;
        border-left: 3px solid #ccc;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.82rem;
        line-height: 1.5;
    }
    .chunk-card.advanced { border-left-color: #4f8ef7; }
    .chunk-meta { font-size: 0.74rem; color: #777; margin-bottom: 0.35rem; }
    .response-box {
        min-height: 80px;
        font-size: 0.92rem;
        line-height: 1.7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Pipeline imports ────────────────────────────────────────────────────────
try:
    import naive.pipeline as naive_pipeline
    import advanced.pipeline as adv_pipeline
except ImportError as exc:
    st.error(f"Could not import pipelines: {exc}")
    st.info("Make sure `naive/pipeline.py` and `advanced/pipeline.py` exist in the project root.")
    st.stop()
    raise  # unreachable; tells the type-checker this branch always exits

# ── Header ───────────────────────────────────────────────────────────────────
st.title("RAG Pipeline Comparison")
st.caption("One query — two retrieval strategies — side-by-side streaming")

st.divider()

# ── Query input ──────────────────────────────────────────────────────────────
query = st.text_input(
    "Query",
    placeholder="Ask something your knowledge base can answer…",
    label_visibility="collapsed",
)

run = st.button("⚡ Compare", type="primary", disabled=not query.strip())

# ── Run ───────────────────────────────────────────────────────────────────────
if run and query.strip():
    query = query.strip()

    # 1. Retrieve context from both pipelines
    with st.spinner("Retrieving context…"):
        t0 = time.perf_counter()
        naive_chunks: list[dict] = naive_pipeline.retrieve(query)
        naive_retrieve_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        adv_chunks: list[dict] = adv_pipeline.retrieve(query)
        adv_retrieve_ms = (time.perf_counter() - t0) * 1000

    st.divider()

    # 2. Build the two-column layout
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<p class="pipeline-title">⬜ Naive RAG</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="pipeline-subtitle">retrieval: {naive_retrieve_ms:.0f} ms '
            f'· {len(naive_chunks)} chunk{"s" if len(naive_chunks) != 1 else ""}</p>',
            unsafe_allow_html=True,
        )

        if naive_chunks:
            with st.expander("Retrieved context", expanded=False):
                for i, chunk in enumerate(naive_chunks):
                    score_str = f"score {chunk['score']:.3f}" if "score" in chunk else ""
                    source_str = chunk.get("source", "")
                    meta = " · ".join(filter(None, [source_str, score_str]))
                    st.markdown(
                        f'<div class="chunk-card">'
                        f'<div class="chunk-meta">Chunk {i + 1}'
                        + (f" · {meta}" if meta else "")
                        + f"</div>{chunk.get('text', '')}</div>",
                        unsafe_allow_html=True,
                    )

        st.markdown("**Response**")
        naive_ph = st.empty()

    with col2:
        st.markdown('<p class="pipeline-title">🔷 Advanced RAG</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="pipeline-subtitle">retrieval: {adv_retrieve_ms:.0f} ms '
            f'· {len(adv_chunks)} chunk{"s" if len(adv_chunks) != 1 else ""}</p>',
            unsafe_allow_html=True,
        )

        if adv_chunks:
            with st.expander("Retrieved context", expanded=False):
                for i, chunk in enumerate(adv_chunks):
                    score_key = "rerank_score" if "rerank_score" in chunk else "score"
                    score_str = f"score {chunk[score_key]:.3f}" if score_key in chunk else ""
                    source_str = chunk.get("source", "")
                    meta = " · ".join(filter(None, [source_str, score_str]))
                    st.markdown(
                        f'<div class="chunk-card advanced">'
                        f'<div class="chunk-meta">Chunk {i + 1}'
                        + (f" · {meta}" if meta else "")
                        + f"</div>{chunk.get('text', '')}</div>",
                        unsafe_allow_html=True,
                    )

        st.markdown("**Response**")
        adv_ph = st.empty()

    # 3. Stream both responses in parallel via background threads + queues
    q_naive: queue.Queue = queue.Queue()
    q_adv: queue.Queue = queue.Queue()

    def _stream(pipeline_fn, q: queue.Queue, chunks: list[dict]) -> None:
        try:
            for token in pipeline_fn(query, chunks):
                q.put(("text", token))
        except Exception as exc:  # noqa: BLE001
            q.put(("error", str(exc)))
        finally:
            q.put(("done", None))

    t_naive = threading.Thread(
        target=_stream, args=(naive_pipeline.stream, q_naive, naive_chunks), daemon=True
    )
    t_adv = threading.Thread(
        target=_stream, args=(adv_pipeline.stream, q_adv, adv_chunks), daemon=True
    )
    t_naive.start()
    t_adv.start()

    naive_text = ""
    adv_text = ""
    done_naive = False
    done_adv = False

    while not (done_naive and done_adv):
        changed = False

        while not q_naive.empty() and not done_naive:
            kind, data = q_naive.get_nowait()
            if kind == "done":
                done_naive = True
            elif kind == "error":
                naive_text += f"\n\n⚠️ Error: {data}"
                done_naive = True
            else:
                naive_text += data
            changed = True

        while not q_adv.empty() and not done_adv:
            kind, data = q_adv.get_nowait()
            if kind == "done":
                done_adv = True
            elif kind == "error":
                adv_text += f"\n\n⚠️ Error: {data}"
                done_adv = True
            else:
                adv_text += data
            changed = True

        if changed:
            naive_ph.markdown(
                '<div class="response-box">' + naive_text + ("▌" if not done_naive else "") + "</div>",
                unsafe_allow_html=True,
            )
            adv_ph.markdown(
                '<div class="response-box">' + adv_text + ("▌" if not done_adv else "") + "</div>",
                unsafe_allow_html=True,
            )

        time.sleep(0.03)

    # Final render (no cursor)
    naive_ph.markdown(naive_text or "*No response generated.*")
    adv_ph.markdown(adv_text or "*No response generated.*")

    t_naive.join()
    t_adv.join()
