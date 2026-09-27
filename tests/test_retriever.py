from langchain_core.documents import Document
from retrieval.bm25 import BM25RetrieverIndex, tokenize


def test_tokenize():
    text = "Section 106 of the Transfer of Property Act, 1882."
    tokens = tokenize(text)
    assert "section" in tokens
    assert "106" in tokens
    assert "property" in tokens


def test_bm25_index():
    chunks = [
        {
            "text": "Landlord eviction under Delhi Rent Control Act Section 106.",
            "metadata": {"chunk_id": "c1", "document_name": "Case 1"}
        },
        {
            "text": "Arbitration proceedings under Section 34 of Arbitration Act.",
            "metadata": {"chunk_id": "c2", "document_name": "Case 2"}
        }
    ]
    bm25 = BM25RetrieverIndex()
    bm25.build_index(chunks)
    results = bm25.search("eviction rent control", k=1)
    assert len(results) == 1
    assert results[0][0].metadata["chunk_id"] == "c1"
