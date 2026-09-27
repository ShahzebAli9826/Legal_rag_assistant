from ingestion.chunker import LegalChunker


def test_chunker_deterministic_id():
    chunker = LegalChunker(chunk_size=100, chunk_overlap=20)
    raw_doc = {
        "text": "Section 1. Short title and extent. This Act may be called the Legal QA Act. It extends to the whole of India.",
        "metadata": {
            "document_id": "test_act",
            "document_name": "Test Act 2024",
            "page_number": 1
        }
    }

    chunks = chunker.chunk_documents([raw_doc])
    assert len(chunks) > 0
    assert chunks[0]["metadata"]["chunk_id"] == "test_act_p1_c01"
    assert chunks[0]["metadata"]["document_name"] == "Test Act 2024"
