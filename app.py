import streamlit as st
from pathlib import Path
from config.settings import VECTORSTORE_DIR
from retrieval.retriever import LegalRetrieverEngine
from rag.chain import LegalRAGChain
from ui.sidebar import render_sidebar
from ui.chat import render_chat_interface

# Streamlit Page Config
st.set_page_config(
    page_title="Legal RAG Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def get_retriever_engine():
    """Cached initialization of the LegalRetrieverEngine."""
    return LegalRetrieverEngine(vectorstore_dir=VECTORSTORE_DIR)


def main():
    # 1. Render Sidebar
    settings = render_sidebar()

    # 2. Initialize cached retriever engine & RAG chain
    retriever_engine = get_retriever_engine()
    rag_chain = LegalRAGChain(retriever_engine=retriever_engine, model_name=settings["model_name"])

    # 3. Render Main Chat UI
    render_chat_interface(rag_chain, settings)


if __name__ == "__main__":
    main()
