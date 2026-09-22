"""
Optional Streamlit UI wrapper - a text box in, a grounded and cited answer out.
Makes the agent demoable without reading code.

Run with: streamlit run app.py
"""

import streamlit as st

from src.main import get_or_build_vector_store
from src.agent import build_qa_chain, answer_question

st.title("Document Q&A Agent")

question = st.text_input("Ask a question about your documents:")

if question:
    vector_store = get_or_build_vector_store()
    chain = build_qa_chain(vector_store)
    result = answer_question(chain, question)

    st.write("### Answer")
    st.write(result["answer"])
    st.write("### Sources")
    st.write(result["sources"])
