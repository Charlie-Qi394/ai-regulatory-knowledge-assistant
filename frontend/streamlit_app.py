"""Streamlit frontend for the AI Regulatory Knowledge Assistant."""

import streamlit as st
from requests import RequestException

from frontend.api_client import check_excel_file, ask_question, fetch_history, get_api_base_url


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


ask_tab, excel_tab = st.tabs(["Ask documents", "Excel compliance checker"])

with ask_tab:
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

with excel_tab:
    st.subheader("Excel Compliance Checker")
    st.caption(
        "Upload a .xlsx file with columns: parameter, value, unit, and optional category. "
        "This MVP checks selected FSANZ infant-formula numeric rules with deterministic Python logic."
    )
    st.code(
        "parameter,value,unit,category\n"
        "Energy,2720,kJ/L,\n"
        "Protein,15,g/L,milk-based\n"
        "Docosahexaenoic acid,10,mg/100 kJ,\n"
        "Total trans fatty acids,3,% of total fatty acids,"
    )

    uploaded_file = st.file_uploader("Upload product data workbook", type=["xlsx"])
    if uploaded_file is not None and st.button("Check Excel", type="primary"):
        with st.spinner("Checking workbook..."):
            try:
                result = check_excel_file(uploaded_file.name, uploaded_file.getvalue())
            except RequestException as exc:
                st.error(f"Excel check failed: {exc}")
            else:
                summary = result["summary"]
                passed, failed, needs_review = st.columns(3)
                passed.metric("Passed", summary["passed"])
                failed.metric("Failed", summary["failed"])
                needs_review.metric("Needs review", summary["needs_review"])

                st.subheader("Results")
                st.dataframe(result["results"], use_container_width=True)
                st.caption(
                    "This checker is a portfolio MVP. It supports selected deterministic rules only "
                    "and is not legal, regulatory, compliance, or quality advice."
                )
