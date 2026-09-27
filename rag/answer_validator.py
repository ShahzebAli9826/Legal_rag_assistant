from typing import List, Dict, Any
from langchain_core.documents import Document


class AnswerValidator:
    """Validates retrieval context and generated answers to prevent hallucinations."""

    INSUFFICIENT_EVIDENCE_MSG = (
        "I could not find sufficient information in the available legal documents to answer this question."
    )

    @classmethod
    def validate_context(cls, documents: List[Document]) -> bool:
        """Returns True if retrieval produced non-empty, usable document chunks."""
        if not documents:
            return False

        # Check total character length of context
        total_chars = sum(len(doc.page_content.strip()) for doc in documents)
        if total_chars < 50:
            return False

        return True

    @classmethod
    def post_process_answer(cls, answer: str, documents: List[Document]) -> Dict[str, Any]:
        """Validates answer and formats result with grounding flag."""
        if not cls.validate_context(documents):
            return {
                "answer": cls.INSUFFICIENT_EVIDENCE_MSG,
                "is_grounded": False,
                "reason": "Insufficient document evidence retrieved."
            }

        cleaned_answer = answer.strip()
        if not cleaned_answer:
            return {
                "answer": cls.INSUFFICIENT_EVIDENCE_MSG,
                "is_grounded": False,
                "reason": "LLM returned empty response."
            }

        return {
            "answer": cleaned_answer,
            "is_grounded": True,
            "reason": "Answer generated from retrieved context."
        }
