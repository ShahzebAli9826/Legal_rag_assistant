from rag.answer_validator import AnswerValidator

class ValidationAgent:
    """Agent responsible for checking grounding and evidence alignment."""

    def validate(self, answer: str, documents: list) -> dict:
        return AnswerValidator.post_process_answer(answer, documents)
