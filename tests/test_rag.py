from rag.answer_validator import AnswerValidator
from langchain_core.documents import Document


def test_answer_validator_insufficient():
    empty_docs = []
    assert AnswerValidator.validate_context(empty_docs) is False

    result = AnswerValidator.post_process_answer("Some guess", empty_docs)
    assert result["is_grounded"] is False
    assert AnswerValidator.INSUFFICIENT_EVIDENCE_MSG in result["answer"]


def test_answer_validator_sufficient():
    docs = [Document(page_content="Valid legal content text for testing grounding validation logic.", metadata={})]
    assert AnswerValidator.validate_context(docs) is True

    result = AnswerValidator.post_process_answer("Grounded answer text", docs)
    assert result["is_grounded"] is True
    assert result["answer"] == "Grounded answer text"
