import logging
from typing import List, Dict, Tuple
from langchain_core.documents import Document
from retrieval.vector_store import LegalVectorStore
from retrieval.bm25 import BM25RetrieverIndex
from config.settings import RRF_K

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines BM25 keyword retrieval and FAISS vector retrieval using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, vector_store: LegalVectorStore, bm25_index: BM25RetrieverIndex, rrf_k: int = RRF_K):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5, candidate_k: int = 20) -> List[Document]:
        """
        Executes hybrid BM25 + FAISS retrieval using Reciprocal Rank Fusion (RRF).
        """
        # 1. Vector similarity search
        vector_results = self.vector_store.similarity_search_with_score(query, k=candidate_k)
        vector_docs = [doc for doc, score in vector_results]

        # 2. BM25 keyword search
        bm25_results = self.bm25_index.search(query, k=candidate_k)
        bm25_docs = [doc for doc, score in bm25_results]

        # 3. Apply Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # Process vector ranks
        for rank, doc in enumerate(vector_docs):
            chunk_id = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
            doc_map[chunk_id] = doc
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (self.rrf_k + (rank + 1)))

        # Process BM25 ranks
        for rank, doc in enumerate(bm25_docs):
            chunk_id = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
            doc_map[chunk_id] = doc
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (self.rrf_k + (rank + 1)))

        # Sort by RRF score descending
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        # Select top-k documents
        final_docs = [doc_map[cid] for cid in sorted_chunk_ids[:top_k]]
        logger.info(f"Hybrid retrieval retrieved {len(final_docs)} chunks using RRF.")
        return final_docs
