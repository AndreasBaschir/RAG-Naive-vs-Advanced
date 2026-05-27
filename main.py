import queue
import threading
import time

import streamlit as st

from advanced import pipeline as adv
from naive import pipeline as naive

st.set_page_config(layout="wide", page_title="Naive vs Advanced RAG")
st.title("Naive vs Advanced RAG")

col1, col2 = st.columns(2)
col1.subheader("Naive RAG")
col2.subheader("Advanced RAG")

def fill_queue(generator, q: queue.Queue) -> None:
    accumulated = ""
    try:
        for chunk in generator:
            accumulated += chunk
            q.put(accumulated)
    except Exception as e:
        q.put(f"[Eroare: {e}]")
    finally:
        q.put(None)  # sentinel


if prompt := st.chat_input("Ask anything..."):
    naive_chunks = naive.retrieve(prompt)
    adv_chunks = adv.retrieve(prompt)

    naive_placeholder = col1.empty()
    adv_placeholder = col2.empty()

    naive_q: queue.Queue[str | None] = queue.Queue()
    adv_q: queue.Queue[str | None] = queue.Queue()

    threading.Thread(target=fill_queue, args=(naive.stream(prompt, naive_chunks), naive_q), daemon=True).start()
    threading.Thread(target=fill_queue, args=(adv.stream(prompt, adv_chunks), adv_q), daemon=True).start()

    naive_done = adv_done = False

    while not (naive_done and adv_done):
        if not naive_done:
            try:
                item = naive_q.get_nowait()
                if item is None:
                    naive_done = True
                else:
                    naive_placeholder.markdown(item)
            except queue.Empty:
                pass

        if not adv_done:
            try:
                item = adv_q.get_nowait()
                if item is None:
                    adv_done = True
                else:
                    adv_placeholder.markdown(item)
            except queue.Empty:
                pass

        time.sleep(0.02)
