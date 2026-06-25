import time
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

import datasets_registry
from advanced import pipeline as adv
from advanced.retrieval import _get_bm25, _get_embedder as _adv_embedder, _get_reranker
from naive import pipeline as naive


@st.cache_resource
def _warmup(dataset: str) -> None:
    """Load all lazy singletons for the selected dataset into memory.

    Keyed by *dataset* so Streamlit rebuilds singletons when the user
    switches datasets in the sidebar.

    :param dataset: dataset key used as the Streamlit cache key
    """
    datasets_registry.set_active(dataset)
    naive._get_embedder()
    _adv_embedder()
    _get_reranker()
    _get_bm25()


st.set_page_config(layout="wide", page_title="Naive vs Advanced RAG")
st.title("Naive vs Advanced RAG")

dataset = st.sidebar.selectbox("Dataset", datasets_registry.available())
datasets_registry.set_active(dataset)
_warmup(dataset)

if "messages" not in st.session_state:
    st.session_state.messages = []


def get_response(pipeline, prompt: str, chunks: list[dict]) -> tuple[str, float]:
    """Stream a full response from *pipeline* and measure wall-clock latency.

    :param pipeline: naive or advanced pipeline module (must expose ``stream``)
    :param prompt: user question
    :param chunks: retrieved context chunks passed to the pipeline
    :returns: tuple of (generated answer text, elapsed seconds)
    """
    start = time.perf_counter()
    result = "".join(pipeline.stream(prompt, chunks))
    elapsed = time.perf_counter() - start
    return result, elapsed


for msg in st.session_state.messages:
    with st.chat_message("user"):
        st.markdown(msg["prompt"])

    col1, col2 = st.columns(2)
    col1.subheader(f"Naive RAG — {msg['naive_time']:.2f}s")
    col1.markdown(msg["naive"])
    col2.subheader(f"Advanced RAG — {msg['adv_time']:.2f}s")
    col2.markdown(msg["advanced"])

if prompt := st.chat_input("Ask anything..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    naive_chunks = naive.retrieve(prompt)
    adv_chunks = adv.retrieve(prompt)

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_naive = ex.submit(get_response, naive, prompt, naive_chunks)
        f_adv = ex.submit(get_response, adv, prompt, adv_chunks)
        naive_resp, naive_time = f_naive.result()
        adv_resp, adv_time = f_adv.result()

    col1, col2 = st.columns(2)
    col1.subheader(f"Naive RAG — {naive_time:.2f}s")
    col1.markdown(naive_resp)
    col2.subheader(f"Advanced RAG — {adv_time:.2f}s")
    col2.markdown(adv_resp)

    st.session_state.messages.append({
        "prompt": prompt,
        "naive": naive_resp,
        "naive_time": naive_time,
        "advanced": adv_resp,
        "adv_time": adv_time,
    })
