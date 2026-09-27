import streamlit as st
import os
import json
from pathlib import Path
from config.settings import DATA_DIR, METADATA_DIR, GEMINI_API_KEY, OPENAI_API_KEY, DEFAULT_TOP_K, DEFAULT_GEMINI_MODEL


def render_sidebar():
    """Renders a clean, uncluttered Streamlit sidebar."""
    st.sidebar.title("⚖️ Legal RAG Assistant")
    st.sidebar.caption("Document-Grounded Legal QA System")
    st.sidebar.markdown("---")

    # Corpus Stats
    st.sidebar.subheader("📊 Knowledge Base Status")
    meta_file = Path(METADATA_DIR) / "corpus_metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            st.sidebar.info(
                f"📄 Documents: **{meta.get('total_documents', 0)}**\n\n"
                f"📑 Pages/Records: **{meta.get('total_pages', 0)}**\n\n"
                f"🧩 Chunks Indexed: **{meta.get('total_chunks', 0)}**"
            )
        except Exception:
            st.sidebar.warning("Could not read corpus metadata.")
    else:
        st.sidebar.warning("No vector index found. Click Rebuild Index below.")

    if st.sidebar.button("🔄 Rebuild / Refresh Knowledge Index", use_container_width=True):
        with st.spinner("Building vector & BM25 indices..."):
            try:
                from ingestion.build_index import build_knowledge_base
                build_knowledge_base()
                st.sidebar.success("Index rebuilt successfully!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Index build failed: {e}")

    st.sidebar.markdown("---")

    # Optional Collapsible Advanced Settings (Kept clean & collapsed by default)
    with st.sidebar.expander("⚙️ Advanced Model Settings", expanded=False):
        provider = st.radio("Provider", options=["Gemini", "OpenAI"], index=0).lower()
        if provider == "gemini":
            api_key_input = st.text_input("Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password")
            model_name = st.selectbox("Model", options=["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"], index=0)
        else:
            api_key_input = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
            model_name = st.selectbox("Model", options=["gpt-4o-mini", "gpt-4o"], index=0)

        top_k = st.slider("Top-K Retrieval Chunks", min_value=1, max_value=10, value=DEFAULT_TOP_K)

    return {
        "provider": provider if 'provider' in locals() else "gemini",
        "api_key": api_key_input if 'api_key_input' in locals() else os.getenv("GEMINI_API_KEY", ""),
        "model_name": model_name if 'model_name' in locals() else "gemini-2.5-flash",
        "top_k": top_k if 'top_k' in locals() else DEFAULT_TOP_K
    }
