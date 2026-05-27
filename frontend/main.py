import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import streamlit as st
from naive.pipeline import retrieve, stream

st.title("Naive vs Advanced RAG")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me anything about the knowledge base."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    chunks = retrieve(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(stream(prompt, chunks))

    st.session_state.messages.append({"role": "assistant", "content": str(response)})