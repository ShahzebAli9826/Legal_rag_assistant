import logging
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL_NAME, NORMALIZE_EMBEDDINGS

logger = logging.getLogger(__name__)


class LegalEmbeddings:
    """Provides document and query embeddings using HuggingFace sentence-transformers."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        logger.info(f"Loading HuggingFace Embedding model: {model_name}")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": NORMALIZE_EMBEDDINGS}
            )
        except Exception as e:
            logger.warning(f"Failed to load {model_name}, falling back to all-MiniLM-L6-v2: {e}")
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": NORMALIZE_EMBEDDINGS}
            )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

    def get_langchain_embeddings(self) -> HuggingFaceEmbeddings:
        return self.embeddings
