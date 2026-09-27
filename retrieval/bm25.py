import pickle
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from config.settings import BM25_K1, BM25_B

logger = logging.getLogger(__name__)


def tokenize(text: str) -> List[str]:
    """Lowercases and tokenizes legal text into alphanumeric terms."""
    return re.findall(r'\w+', text.lower())


class BM25RetrieverIndex:
    """BM25 keyword retrieval index for exact keyword matching in legal texts."""

    def __init__(self):
        self.bm25 = None
        self.documents: List[Document] = []

    def build_index(self, chunks: List[Dict[str, Any]]):
        """Builds BM25 index from processed text chunks."""
        self.documents = []
        corpus_tokens = []

        for chunk in chunks:
            doc = Document(
                page_content=chunk["text"],
                metadata=chunk["metadata"]
            )
            self.documents.append(doc)
            tokens = tokenize(chunk["text"])
            corpus_tokens.append(tokens)

        logger.info(f"Building BM25 index over {len(self.documents)} document chunks...")
        self.bm25 = BM25Okapi(corpus_tokens, k1=BM25_K1, b=BM25_B)
        logger.info("BM25 index built successfully.")

    def save(self, directory: Path | str):
        """Saves BM25 index and documents to disk."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / "bm25_index.pkl"

        data = {
            "documents": self.documents,
            "bm25": self.bm25
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved BM25 index to {filepath}")

    def load(self, directory: Path | str):
        """Loads BM25 index from disk."""
        filepath = Path(directory) / "bm25_index.pkl"
        if not filepath.exists():
            raise FileNotFoundError(f"BM25 index file not found: {filepath}")

        logger.info(f"Loading BM25 index from {filepath}...")
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.documents = data["documents"]
        self.bm25 = data["bm25"]
        logger.info("BM25 index loaded successfully.")

    def search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Performs BM25 keyword search for given query string."""
        if self.bm25 is None or not self.documents:
            raise ValueError("BM25 index is not initialized or loaded.")

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        # Get top-k indices sorted by score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for idx in top_indices:
            score = scores[idx]
            doc = self.documents[idx]
            if score > 0:
                results.append((doc, float(score)))
            elif score == 0:
                # In small corpora (e.g. N=2), BM25Okapi IDF can be exactly 0.0 even when terms match.
                # Check if document actually contains any of the query tokens.
                doc_text_lower = doc.page_content.lower()
                if any(t in doc_text_lower for t in tokens):
                    results.append((doc, 0.001))

        return results
