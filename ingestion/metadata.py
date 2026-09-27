import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from config.settings import METADATA_DIR

logger = logging.getLogger(__name__)


class MetadataManager:
    """Manages document and chunk metadata tracking."""

    def __init__(self, metadata_dir: Path = METADATA_DIR):
        self.metadata_dir = Path(metadata_dir)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.summary_file = self.metadata_dir / "corpus_metadata.json"

    def save_corpus_metadata(self, documents: List[Dict[str, Any]], chunks: List[Dict[str, Any]]):
        """Saves corpus summary statistics and chunk indices."""
        doc_stats = {}
        for doc in documents:
            meta = doc.get("metadata", {})
            doc_name = meta.get("document_name", "unknown")
            if doc_name not in doc_stats:
                doc_stats[doc_name] = {
                    "document_id": meta.get("document_id"),
                    "source_path": meta.get("source_path"),
                    "pages_loaded": 0
                }
            doc_stats[doc_name]["pages_loaded"] += 1

        summary = {
            "total_documents": len(doc_stats),
            "total_pages": len(documents),
            "total_chunks": len(chunks),
            "documents": doc_stats
        }

        try:
            with open(self.summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Saved corpus metadata summary to {self.summary_file}")
        except Exception as e:
            logger.error(f"Failed to save metadata summary: {e}")

        return summary
