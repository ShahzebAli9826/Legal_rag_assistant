from typing import List
from langchain_core.documents import Document


class LegalContextBuilder:
    """Formats retrieved document chunks into structured context strings for LLM prompts."""

    @staticmethod
    def build_context(documents: List[Document]) -> str:
        if not documents:
            return "No relevant legal document context found."

        context_blocks = []
        for idx, doc in enumerate(documents, start=1):
            meta = doc.metadata or {}
            doc_name = meta.get("document_name", "Unknown Legal Document")
            page_num = meta.get("page_number", "N/A")
            chunk_id = meta.get("chunk_id", f"chunk_{idx}")

            header = f"--- SOURCE {idx}: {doc_name} (Page/Ref: {page_num}, ID: {chunk_id}) ---"
            body = doc.page_content.strip()
            context_blocks.append(f"{header}\n{body}\n")

        return "\n".join(context_blocks)

    @staticmethod
    def extract_sources(documents: List[Document]) -> List[dict]:
        """Extracts unique source metadata dicts for display in UI."""
        sources = []
        seen = set()

        for doc in documents:
            meta = doc.metadata or {}
            doc_name = meta.get("document_name", "Unknown Document")
            page_num = meta.get("page_number", "N/A")
            chunk_id = meta.get("chunk_id", "")
            key = (doc_name, page_num)

            if key not in seen:
                seen.add(key)
                sources.append({
                    "document_name": doc_name,
                    "page_number": page_num,
                    "chunk_id": chunk_id,
                    "snippet": doc.page_content[:150] + "..."
                })

        return sources
