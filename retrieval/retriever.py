import logging
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from config.settings import VECTORSTORE_DIR, DEFAULT_TOP_K
from retrieval.embeddings import LegalEmbeddings
from retrieval.vector_store import LegalVectorStore
from retrieval.bm25 import BM25RetrieverIndex
from retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


class LegalRetrieverEngine:
    """Unified Legal Retriever managing vector store, BM25 index, and hybrid search."""

    def __init__(self, vectorstore_dir: Path = VECTORSTORE_DIR):
        self.vectorstore_dir = Path(vectorstore_dir)
        self.embedder = LegalEmbeddings()
        self.vector_store = LegalVectorStore(self.embedder)
        self.bm25_index = BM25RetrieverIndex()
        self.hybrid_retriever = None
        self.is_loaded = False

        self._load_indexes()

    def _load_indexes(self):
        """Loads both FAISS and BM25 indices from disk if available."""
        try:
            self.vector_store.load(self.vectorstore_dir)
            self.bm25_index.load(self.vectorstore_dir)
            self.hybrid_retriever = HybridRetriever(self.vector_store, self.bm25_index)
            self.is_loaded = True
            logger.info("LegalRetrieverEngine successfully loaded both FAISS and BM25 indices.")
        except Exception as e:
            logger.warning(f"Failed to load indices from {self.vectorstore_dir}: {e}")
            self.is_loaded = False

    def get_relevant_documents(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Document]:
        """Retrieves top_k relevant documents using hybrid search (or fallback to vector search)."""
        if not self.is_loaded or self.hybrid_retriever is None:
            raise RuntimeError("Retriever indices are not loaded. Please run 'python -m ingestion.build_index' first.")

        return self.hybrid_retriever.retrieve(query=query, top_k=top_k)
