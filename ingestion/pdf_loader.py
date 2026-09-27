import os
import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any
import pymupdf  # PyMuPDF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFLoader:
    """Extracts text and metadata from PDF files using PyMuPDF."""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

    def load(self) -> List[Dict[str, Any]]:
        """
        Reads the PDF page by page and returns a list of dictionaries with text and metadata.
        """
        documents = []
        doc_name = self.file_path.name
        doc_id = self.file_path.stem.lower().replace(" ", "_")

        try:
            doc = pymupdf.open(self.file_path)
            logger.info(f"Loading PDF: {doc_name} ({len(doc)} pages)")

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")

                if not text or not text.strip():
                    logger.warning(f"Page {page_num + 1} of {doc_name} is empty or unreadable.")
                    continue

                documents.append({
                    "text": text,
                    "metadata": {
                        "document_id": doc_id,
                        "document_name": doc_name,
                        "source_path": str(self.file_path),
                        "page_number": page_num + 1,
                        "total_pages": len(doc)
                    }
                })

            doc.close()
        except Exception as e:
            logger.error(f"Error reading PDF {self.file_path}: {e}")
            raise e

        return documents


class DatasetLoader:
    """Loads legal case data from JSON or CSV files in the data directory."""

    @staticmethod
    def load_json(file_path: str | Path) -> List[Dict[str, Any]]:
        path = Path(file_path)
        if not path.exists():
            return []

        documents = []
        doc_name = path.name
        doc_id = path.stem.lower().replace(" ", "_")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for idx, item in enumerate(data):
                    case_name = item.get("case_name") or item.get("title") or f"Case #{item.get('id', idx + 1)}"
                    text_content = item.get("text") or item.get("content") or ""
                    summary = item.get("summary", "")

                    full_text = text_content
                    if summary:
                        full_text = f"Case Summary:\n{summary}\n\nFull Text:\n{text_content}"

                    if full_text.strip():
                        documents.append({
                            "text": full_text.strip(),
                            "metadata": {
                                "document_id": f"{doc_id}_{idx+1}",
                                "document_name": case_name,
                                "source_path": str(path),
                                "page_number": idx + 1,
                                "summary": summary[:200] if summary else ""
                            }
                        })
        except Exception as e:
            logger.error(f"Error loading JSON dataset {file_path}: {e}")

        return documents

    @staticmethod
    def load_csv(file_path: str | Path) -> List[Dict[str, Any]]:
        path = Path(file_path)
        if not path.exists():
            return []

        documents = []
        doc_name = path.name
        doc_id = path.stem.lower().replace(" ", "_")

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    case_name = row.get("case_name") or row.get("title") or f"Case #{row.get('id', idx + 1)}"
                    text_content = row.get("text") or row.get("content") or ""
                    summary = row.get("summary", "")

                    full_text = text_content
                    if summary:
                        full_text = f"Case Summary:\n{summary}\n\nFull Text:\n{text_content}"

                    if full_text.strip():
                        documents.append({
                            "text": full_text.strip(),
                            "metadata": {
                                "document_id": f"{doc_id}_{idx+1}",
                                "document_name": case_name,
                                "source_path": str(path),
                                "page_number": idx + 1,
                                "summary": summary[:200] if summary else ""
                            }
                        })
        except Exception as e:
            logger.error(f"Error loading CSV dataset {file_path}: {e}")

        return documents


def load_all_documents(data_dir: str | Path) -> List[Dict[str, Any]]:
    """Loads all PDFs, JSONs, and CSVs found in data_dir."""
    data_path = Path(data_dir)
    all_docs = []

    # Load PDFs
    for pdf_file in data_path.glob("*.pdf"):
        loader = PDFLoader(pdf_file)
        all_docs.extend(loader.load())

    # Load JSON files (e.g., sample_cases.json)
    for json_file in data_path.glob("*.json"):
        if "metadata" not in json_file.name and "evaluation" not in json_file.name:
            all_docs.extend(DatasetLoader.load_json(json_file))

    # Load CSV files (e.g., sample_cases.csv)
    for csv_file in data_path.glob("*.csv"):
        all_docs.extend(DatasetLoader.load_csv(csv_file))

    logger.info(f"Total document pages/records loaded: {len(all_docs)}")
    return all_docs
