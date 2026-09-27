import streamlit as st
from rag.chain import LegalRAGChain
from ui.sources import render_sources
from ingestion.file_extractor import FileExtractor


def render_chat_interface(rag_chain: LegalRAGChain, settings: dict):
    """Renders the main legal QA conversational chat interface with file upload and vector DB modes."""
    st.title("⚖️ Legal RAG Assistant")
    st.markdown(
        "Ask natural-language questions grounded strictly in legal acts, cases, or your uploaded case documents."
    )

    # 1. Direct File / Image / PDF Media Uploader Box
    st.markdown("### 📎 Upload Case Document / Image / PDF")
    uploaded_file = st.file_uploader(
        "Upload a case PDF, document image (PNG/JPG), or text file to analyze directly:",
        type=["pdf", "png", "jpg", "jpeg", "txt", "json", "csv"],
        key="active_chat_uploader",
        help="If uploaded, the AI answers using YOUR document. If empty, the AI searches the Vector Database."
    )

    extracted_custom_doc = None
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        extracted_custom_doc = FileExtractor.extract_from_bytes(file_bytes, uploaded_file.name)
        st.info(f"📄 **Active Document Loaded:** `{uploaded_file.name}` ({len(extracted_custom_doc['text'])} chars extracted). All answers will be grounded strictly in this file.")

    st.markdown("---")

    # 2. Initialize Message History
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your Legal QA Assistant. Upload a legal case document/image above or ask a question directly to search our legal database!",
                "sources": []
            }
        ]

    # 3. Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])

    # 4. Chat Input Box
    user_query = st.chat_input("Ask a legal question (e.g., What is the summary and key ruling of this case?)...")

    if user_query:
        # Render User Question
        st.chat_message("user").markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query, "sources": []})

        # Process Query
        with st.chat_message("assistant"):
            if extracted_custom_doc and extracted_custom_doc["text"]:
                spinner_msg = f"Analyzing uploaded document `{extracted_custom_doc['filename']}` using Gemini 2.5 Flash..."
            else:
                spinner_msg = "Searching legal vector database & generating deep grounded analysis using Gemini 2.5 Flash..."

            with st.spinner(spinner_msg):
                if extracted_custom_doc and extracted_custom_doc["text"]:
                    # Mode A: User Uploaded Document Mode
                    result = rag_chain.query_custom_document(
                        question=user_query,
                        custom_text=extracted_custom_doc["text"],
                        filename=extracted_custom_doc["filename"],
                        api_key=settings["api_key"],
                        provider=settings["provider"],
                        model_name=settings["model_name"]
                    )
                else:
                    # Mode B: Vector Database Search Mode
                    result = rag_chain.query(
                        question=user_query,
                        top_k=settings["top_k"],
                        api_key=settings["api_key"],
                        provider=settings["provider"],
                        model_name=settings["model_name"]
                    )

                answer_text = result["answer"]
                sources = result.get("sources", [])

                st.markdown(answer_text)
                if sources:
                    render_sources(sources)

                # Save to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources
                })

    # Legal Safety Disclaimer Footer (Section 41)
    st.markdown("---")
    st.caption(
        "⚠️ **Legal Disclaimer:** This AI system is a document research assistant designed for legal document analysis. "
        "It provides responses based strictly on supplied context and is not a substitute for formal legal counsel."
    )
