import pytest
from pathlib import Path
from ingestion.pdf_loader import DatasetLoader, load_all_documents


def test_dataset_loader_sample_json():
    sample_json = Path(__file__).resolve().parent.parent / "data" / "sample_cases.json"
    if sample_json.exists():
        docs = DatasetLoader.load_json(sample_json)
        assert len(docs) > 0
        assert "text" in docs[0]
        assert "metadata" in docs[0]
        assert "document_name" in docs[0]["metadata"]


def test_load_all_documents():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    docs = load_all_documents(data_dir)
    assert isinstance(docs, list)
