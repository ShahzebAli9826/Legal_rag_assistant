from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from ingestion.text_cleaner import TextCleaner


class LegalChunker:
    """Splits legal text into chunks with deterministic chunk IDs and preserved metadata."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\nSection ", "\n\nArticle ", "\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes list of documents containing {'text': str, 'metadata': dict}
        and returns list of chunk dicts with deterministic chunk IDs.
        """
        chunks = []

        for doc in documents:
            raw_text = doc.get("text", "")
            cleaned_text = TextCleaner.clean_text(raw_text)
            if not cleaned_text:
                continue

            metadata = doc.get("metadata", {}).copy()
            doc_id = metadata.get("document_id", "doc")
            page_num = metadata.get("page_number", 1)

            split_texts = self.splitter.split_text(cleaned_text)

            for chunk_idx, text_chunk in enumerate(split_texts):
                chunk_id = f"{doc_id}_p{page_num}_c{chunk_idx+1:02d}"
                chunk_meta = metadata.copy()
                chunk_meta["chunk_id"] = chunk_id
                chunk_meta["chunk_index"] = chunk_idx + 1

                chunks.append({
                    "text": text_chunk,
                    "metadata": chunk_meta
                })

        return chunks
