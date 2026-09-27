import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from retrieval.embeddings import LegalEmbeddings

logger = logging.getLogger(__name__)


class LegalVectorStore:
    """FAISS vector database wrapper for document embedding storage and retrieval."""

    def __init__(self, embedder: LegalEmbeddings):
        self.embedder = embedder
        self.vectorstore = None

    def build_index(self, chunks: List[Dict[str, Any]]):
        """Creates FAISS vectorstore from chunk dictionary objects."""
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk["text"],
                metadata=chunk["metadata"]
            )
            documents.append(doc)

        logger.info(f"Indexing {len(documents)} chunks into FAISS vector store...")
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embedder.get_langchain_embeddings()
        )
        logger.info("FAISS vector store built successfully.")

    def save(self, directory: Path | str):
        """Saves FAISS index files to disk."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        if self.vectorstore is not None:
            self.vectorstore.save_local(folder_path=str(path))
            logger.info(f"Saved FAISS index to {path}")
        else:
            raise ValueError("No FAISS vector store to save!")

    def load(self, directory: Path | str):
        """Loads FAISS index files from disk."""
        path = Path(directory)
        if not path.exists():
            raise FileNotFoundError(f"Vectorstore directory does not exist: {path}")

        logger.info(f"Loading FAISS vector store from {path}...")
        self.vectorstore = FAISS.load_local(
            folder_path=str(path),
            embeddings=self.embedder.get_langchain_embeddings(),
            allow_dangerous_deserialization=True
        )
        logger.info("FAISS vector store loaded successfully.")

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Returns top k similar Document objects along with distance scores."""
        if self.vectorstore is None:
            raise ValueError("FAISS vector store is not initialized or loaded.")
        return self.vectorstore.similarity_search_with_score(query, k=k)
