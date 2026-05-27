import time
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from advanced import pipeline as adv
from naive import pipeline as naive

st.set_page_config(layout="wide", page_title="Naive vs Advanced RAG")
st.title("Naive vs Advanced RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []


def get_response(pipeline, prompt: str, chunks: list[dict]) -> tuple[str, float]:
    start = time.perf_counter()
    result = "".join(pipeline.stream(prompt, chunks))
    elapsed = time.perf_counter() - start
    return result, elapsed


# Afișare istoric
for msg in st.session_state.messages:
    with st.chat_message("user"):
        st.markdown(msg["prompt"])

    col1, col2 = st.columns(2)
    col1.subheader(f"Naive RAG — {msg['naive_time']:.2f}s")
    col1.markdown(msg["naive"])
    col2.subheader(f"Advanced RAG — {msg['adv_time']:.2f}s")
    col2.markdown(msg["advanced"])

# Input
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
