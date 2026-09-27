import io
import json
import csv
import logging
from typing import Dict, Any
import pymupdf  # PyMuPDF for PDFs and image text extraction

logger = logging.getLogger(__name__)


class FileExtractor:
    """Utility for extracting text from uploaded PDFs, Images, TXT, JSON, and CSV files."""

    @staticmethod
    def extract_from_bytes(file_bytes: bytes, filename: str) -> Dict[str, Any]:
        ext = filename.lower().split('.')[-1]
        text_content = ""
        page_count = 1

        try:
            if ext == "pdf":
                doc = pymupdf.open(stream=file_bytes, filetype="pdf")
                page_count = len(doc)
                pages_text = []
                for i, page in enumerate(doc):
                    t = page.get_text("text").strip()
                    if t:
                        pages_text.append(f"[Page {i+1}]\n{t}")
                doc.close()
                text_content = "\n\n".join(pages_text)

            elif ext in ["png", "jpg", "jpeg", "webp"]:
                # Open image using PyMuPDF doc stream
                doc = pymupdf.open(stream=file_bytes, filetype=ext)
                pages_text = []
                for i, page in enumerate(doc):
                    t = page.get_text("text").strip()
                    if t:
                        pages_text.append(t)
                doc.close()
                text_content = "\n\n".join(pages_text)
                if not text_content:
                    text_content = f"Uploaded image document: {filename} (Content received)"

            elif ext == "txt":
                text_content = file_bytes.decode("utf-8", errors="ignore")

            elif ext == "json":
                data = json.loads(file_bytes.decode("utf-8", errors="ignore"))
                text_content = json.dumps(data, indent=2)

            elif ext == "csv":
                text_content = file_bytes.decode("utf-8", errors="ignore")

            else:
                text_content = file_bytes.decode("utf-8", errors="ignore")

        except Exception as e:
            logger.error(f"Error extracting text from {filename}: {e}")
            text_content = f"Failed to extract text from {filename}: {str(e)}"

        return {
            "filename": filename,
            "text": text_content.strip(),
            "page_count": page_count
        }
