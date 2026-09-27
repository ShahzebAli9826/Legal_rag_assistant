from ingestion.metadata import MetadataManager


def test_metadata_manager_save(tmp_path):
    mgr = MetadataManager(metadata_dir=tmp_path)
    docs = [{"metadata": {"document_name": "Act1", "document_id": "act1", "source_path": "path"}}]
    chunks = [{"text": "t", "metadata": {"chunk_id": "c1"}}]

    summary = mgr.save_corpus_metadata(docs, chunks)
    assert summary["total_documents"] == 1
    assert summary["total_chunks"] == 1
    assert (tmp_path / "corpus_metadata.json").exists()
