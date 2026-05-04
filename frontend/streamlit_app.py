"""Streamlit frontend for the AI Regulatory Knowledge Assistant."""

import streamlit as st
from requests import RequestException

from frontend.api_client import ask_question, fetch_history, get_api_base_url


st.set_page_config(page_title="AI Regulatory Knowledge Assistant", layout="wide")

st.title("AI Regulatory Knowledge Assistant")
st.caption("Ask questions against the ingested regulatory/product knowledge documents.")

with st.sidebar:
    st.subheader("API")
    st.code(get_api_base_url())

    st.subheader("Recent Questions")
    try:
        history = fetch_history()
    except RequestException as exc:
        st.warning(f"Could not load query history: {exc}")
        history = []

    if history:
        for item in history[:10]:
            st.caption(item["created_at"])
            st.write(item["question"])
    else:
        st.caption("No query history yet.")


question = st.text_input("Ask a regulatory question")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked:
    cleaned_question = question.strip()
    if not cleaned_question:
        st.error("Please enter a question.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            try:
                result = ask_question(cleaned_question)
            except RequestException as exc:
                st.error(f"Request failed: {exc}")
            else:
                st.subheader("Answer")
                st.write(result["answer"])

                sources = result.get("sources", [])
                st.subheader("Sources")
                if sources:
                    for source in sources:
                        with st.expander(
                            f"[Source {source['source_id']}] {source['filename']} "
                            f"| chunk {source['chunk_index']}"
                        ):
                            st.caption(
                                f"similarity={source['similarity']:.4f} "
                                f"distance={source['distance']:.4f}"
                            )
                            if source.get("page_number") is not None:
                                st.caption(f"page={source['page_number']}")
                            st.write(source["excerpt"])
                else:
                    st.info("No sources returned.")
