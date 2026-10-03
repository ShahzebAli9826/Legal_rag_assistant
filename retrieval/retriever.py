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
        self.bm25_index = BM25RetrieverIndex()
        self.embedder = None
        self.vector_store = None
        self.hybrid_retriever = None
        self.embedding_error = None
        try:
            self.embedder = LegalEmbeddings()
            self.vector_store = LegalVectorStore(self.embedder)
        except Exception as e:
            # Keep keyword retrieval available when Windows or another runtime
            # blocks a compiled embedding dependency (for example sklearn DLLs).
            self.embedding_error = e
            logger.warning("Embedding model unavailable; using BM25 keyword retrieval only: %s", e)
        self.is_loaded = False

        self._load_indexes()

    def _load_indexes(self):
        """Load BM25 always; enable hybrid retrieval when embeddings are available."""
        try:
            self.bm25_index.load(self.vectorstore_dir)
            if self.vector_store is not None:
                self.vector_store.load(self.vectorstore_dir)
                self.hybrid_retriever = HybridRetriever(self.vector_store, self.bm25_index)
            else:
                self.hybrid_retriever = None
            self.is_loaded = True
            mode = "hybrid PyTorch vectors + BM25" if self.hybrid_retriever else "BM25 keyword-only"
            logger.info("LegalRetrieverEngine loaded in %s mode.", mode)
        except Exception as e:
            logger.warning(f"Failed to load indices from {self.vectorstore_dir}: {e}")
            self.is_loaded = False

    def get_relevant_documents(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Document]:
        """Retrieves top_k relevant documents using hybrid search (or fallback to vector search)."""
        if not self.is_loaded:
            raise RuntimeError("Retriever indices are not loaded. Please run 'python -m ingestion.build_index' first.")

        if self.hybrid_retriever is not None:
            return self.hybrid_retriever.retrieve(query=query, top_k=top_k)
        return [doc for doc, _score in self.bm25_index.search(query=query, k=top_k)]
