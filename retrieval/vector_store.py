import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as functional
from langchain_core.documents import Document

from retrieval.embeddings import LegalEmbeddings

logger = logging.getLogger(__name__)
INDEX_FILENAME = "legal_vector_index.pt"


class LegalVectorStore:
    """Local cosine-similarity index backed by PyTorch (no native FAISS DLL)."""

    def __init__(self, embedder: LegalEmbeddings):
        self.embedder = embedder
        self.documents: List[Document] = []
        self.vectors = torch.empty((0, 0), dtype=torch.float32)

    def build_index(self, chunks: List[Dict[str, Any]]):
        self.documents = [
            Document(page_content=chunk["text"], metadata=chunk["metadata"])
            for chunk in chunks
        ]
        logger.info("Embedding %d chunks for the local vector index...", len(self.documents))
        self.vectors = torch.tensor(
            self.embedder.embed_documents([doc.page_content for doc in self.documents]),
            dtype=torch.float32,
        )

    def save(self, directory: Path | str):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "texts": [doc.page_content for doc in self.documents],
            "metadatas": [doc.metadata for doc in self.documents],
            "vectors": self.vectors,
            "model_name": self.embedder.model_name,
        }
        torch.save(payload, path / INDEX_FILENAME)
        logger.info("Saved PyTorch vector index to %s", path / INDEX_FILENAME)

    def _migrate_existing_index(self, directory: Path):
        """Re-embed the documents stored beside the existing, policy-blocked FAISS index."""
        metadata_path = directory / "index.pkl"
        if metadata_path.exists():
            # This is the project-generated LangChain document metadata file.
            with metadata_path.open("rb") as file:
                docstore, _id_map = pickle.load(file)
            stored_docs = list(docstore._dict.values())
            chunks = [
                {"text": doc.page_content, "metadata": doc.metadata}
                for doc in stored_docs
            ]
        else:
            chunks_path = directory.parent / "processed" / "processed_chunks.json"
            if not chunks_path.exists():
                raise FileNotFoundError(f"No vector index or processed chunks found in {directory}")
            with chunks_path.open("r", encoding="utf-8") as file:
                chunks = json.load(file)
        self.build_index(chunks)
        self.save(directory)

    def load(self, directory: Path | str):
        path = Path(directory)
        index_path = path / INDEX_FILENAME
        if index_path.exists():
            payload = torch.load(index_path, map_location="cpu", weights_only=True)
            self.documents = [
                Document(page_content=text, metadata=metadata)
                for text, metadata in zip(payload["texts"], payload["metadatas"])
            ]
            self.vectors = payload["vectors"].to(dtype=torch.float32)
            logger.info("Loaded PyTorch vector index with %d documents.", len(self.documents))
        else:
            logger.info("Migrating existing legal vector index to PyTorch similarity search.")
            self._migrate_existing_index(path)

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        if not self.documents or self.vectors.numel() == 0:
            raise ValueError("Vector index is not initialized or loaded.")
        query_vector = torch.tensor(self.embedder.embed_query(query), dtype=torch.float32)
        scores = functional.normalize(self.vectors, p=2, dim=1) @ functional.normalize(query_vector, p=2, dim=0)
        count = min(k, len(self.documents))
        values, indices = torch.topk(scores, k=count)
        return [(self.documents[idx], float(score)) for score, idx in zip(values.tolist(), indices.tolist())]
