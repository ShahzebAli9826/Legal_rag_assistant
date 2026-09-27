from rag.context_builder import LegalContextBuilder
from rag.prompt import get_legal_qa_prompt
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

class AnswerAgent:
    """Agent responsible for generating grounded answers using LLM."""

    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model_name = model_name
        self.prompt = get_legal_qa_prompt()

    def generate_answer(self, query: str, documents: list) -> str:
        context_str = LegalContextBuilder.build_context(documents)
        llm = ChatOpenAI(api_key=self.api_key, model=self.model_name, temperature=0.1)
        chain = self.prompt | llm | StrOutputParser()
        return chain.invoke({"context": context_str, "question": query})
