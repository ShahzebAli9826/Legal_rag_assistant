import logging
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from config.settings import PROMPTS_DIR

logger = logging.getLogger(__name__)


def get_legal_qa_prompt() -> PromptTemplate:
    prompt_file = Path(PROMPTS_DIR) / "legal_qa.txt"
    if prompt_file.exists():
        template = prompt_file.read_text(encoding="utf-8")
    else:
        template = (
            "You are a Legal QA Assistant. Answer using ONLY context.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\nAnswer:"
        )

    return PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )


def get_validation_prompt() -> PromptTemplate:
    prompt_file = Path(PROMPTS_DIR) / "validation.txt"
    if prompt_file.exists():
        template = prompt_file.read_text(encoding="utf-8")
    else:
        template = (
            "Validate if answer is supported by context:\nContext:\n{context}\nAnswer:\n{answer}\nResult:"
        )

    return PromptTemplate(
        input_variables=["context", "answer"],
        template=template
    )
