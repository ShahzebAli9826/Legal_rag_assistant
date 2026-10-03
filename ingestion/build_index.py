import json
import logging
import sys
from pathlib import Path
from config.settings import DATA_DIR, VECTORSTORE_DIR, PROCESSED_DIR, METADATA_DIR
from ingestion.pdf_loader import load_all_documents
from ingestion.chunker import LegalChunker
from ingestion.metadata import MetadataManager
from retrieval.embeddings import LegalEmbeddings
from retrieval.vector_store import LegalVectorStore
from retrieval.bm25 import BM25RetrieverIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_knowledge_base():
    logger.info("==================================================")
    logger.info("LEGAL RAG INDEX BUILDER")
    logger.info("==================================================")

    # 1. Load documents
    logger.info(f"Scanning for legal documents in: {DATA_DIR}")
    documents = load_all_documents(DATA_DIR)
    if not documents:
        logger.error("No legal documents found in data folder!")
        sys.exit(1)

    logger.info(f"Total document records/pages loaded: {len(documents)}")

    # 2. Chunk documents
    logger.info("Chunking document texts...")
    chunker = LegalChunker()
    chunks = chunker.chunk_documents(documents)
    logger.info(f"Generated {len(chunks)} chunks with metadata.")

    # 3. Save processed chunks
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_chunks_file = PROCESSED_DIR / "processed_chunks.json"
    with open(processed_chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    logger.info(f"Saved processed chunks to: {processed_chunks_file}")

    # 4. Save metadata summary
    metadata_mgr = MetadataManager(METADATA_DIR)
    metadata_mgr.save_corpus_metadata(documents, chunks)

    # 5. Build local PyTorch vector index
    logger.info("Initializing embedding model...")
    embedder = LegalEmbeddings()
    vector_store = LegalVectorStore(embedder)

    logger.info("Building PyTorch cosine-similarity index...")
    vector_store.build_index(chunks)
    vector_store.save(VECTORSTORE_DIR)

    # 6. Build BM25 keyword index
    logger.info("Building BM25 keyword index...")
    bm25_index = BM25RetrieverIndex()
    bm25_index.build_index(chunks)
    bm25_index.save(VECTORSTORE_DIR)

    logger.info("==================================================")
    logger.info("INDEX BUILD COMPLETE SUCCESSFULLY")
    logger.info("==================================================")


if __name__ == "__main__":
    build_knowledge_base()
