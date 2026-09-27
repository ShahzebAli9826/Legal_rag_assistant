import streamlit as st
from typing import List, Dict, Any


def render_sources(sources: List[Dict[str, Any]]):
    """Renders formatted legal source citations and expandable content snippets."""
    if not sources:
        return

    st.markdown("### 📚 Grounded Legal Sources")
    for idx, src in enumerate(sources, start=1):
        doc_name = src.get("document_name", "Unknown Document")
        page_num = src.get("page_number", "N/A")
        snippet = src.get("snippet", "")

        with st.expander(f"Source {idx}: **{doc_name}** — Page/Record {page_num}"):
            st.markdown(f"**Document:** `{doc_name}`")
            st.markdown(f"**Page / Record Number:** `{page_num}`")
            if snippet:
                st.markdown(f"**Snippet Preview:**\n> {snippet}")
