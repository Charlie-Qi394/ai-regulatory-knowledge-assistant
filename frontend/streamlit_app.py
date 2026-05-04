"""Streamlit frontend for the AI Regulatory Knowledge Assistant."""

import streamlit as st
from requests import RequestException

from frontend.api_client import (
    ask_question,
    check_excel_file,
    fetch_history,
    get_api_base_url,
    review_excel_file_with_ai,
)


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
        "Use deterministic checks for coded rules, or AI-assisted review for broader regulatory screening."
    )
    st.info(
        "The rows below are examples of the workbook format only. "
        "AI-assisted review can attempt any parameter in your uploaded file when the regulatory documents "
        "contain enough relevant context. Deterministic mode is limited to coded demonstration rules."
    )
    st.markdown("**Example workbook format**")
    st.code(
        "parameter,value,unit,category\n"
        "Energy,2720,kJ/L,\n"
        "Protein,15,g/L,milk-based\n"
        "Docosahexaenoic acid,10,mg/100 kJ,\n"
        "Total trans fatty acids,3,% of total fatty acids,"
    )

    uploaded_file = st.file_uploader("Upload product data workbook", type=["xlsx"])
    deterministic_clicked = st.button(
        "Run deterministic check",
        type="primary",
        disabled=uploaded_file is None,
    )
    ai_clicked = st.button(
        "Run AI-assisted review",
        disabled=uploaded_file is None,
        help="Uses RAG retrieval and the chat model. Slower, broader, and should be treated as screening only.",
    )

    if uploaded_file is not None and deterministic_clicked:
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

    if uploaded_file is not None and ai_clicked:
        with st.spinner("Retrieving regulatory context and running AI-assisted review..."):
            try:
                result = review_excel_file_with_ai(uploaded_file.name, uploaded_file.getvalue())
            except RequestException as exc:
                st.error(f"AI-assisted review failed: {exc}")
            else:
                summary = result["summary"]
                passed, failed, needs_review, insufficient = st.columns(4)
                passed.metric("Passed", summary["passed"])
                failed.metric("Failed", summary["failed"])
                needs_review.metric("Needs review", summary["needs_review"])
                insufficient.metric("Insufficient context", summary.get("insufficient_context", 0))

                st.subheader("AI-Assisted Review Results")
                table_rows = [
                    {
                        "row": row["row_index"],
                        "parameter": row["parameter"],
                        "value": row["input_value"],
                        "unit": row["input_unit"],
                        "category": row["category"],
                        "status": row["status"],
                        "requirement": row["requirement"],
                        "reasoning": row["reasoning"],
                    }
                    for row in result["results"]
                ]
                st.dataframe(table_rows, use_container_width=True)

                for row in result["results"]:
                    with st.expander(f"Row {row['row_index']}: {row['parameter']} | {row['status']}"):
                        st.write(row["reasoning"])
                        if row["citations"]:
                            st.caption("Citations: " + ", ".join(row["citations"]))
                        for source in row.get("sources", []):
                            st.markdown(
                                f"**[Source {source['source_id']}] {source['filename']} "
                                f"| chunk {source['chunk_index']}**"
                            )
                            if source.get("page_number") is not None:
                                st.caption(f"page={source['page_number']}")
                            st.write(source["excerpt"])

                st.caption(
                    "AI-assisted review is broader than deterministic checking, but it is screening only. "
                    "It depends on retrieval quality and should not be treated as final compliance advice."
                )
