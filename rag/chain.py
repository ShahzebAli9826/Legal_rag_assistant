import logging
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from config.settings import GEMINI_API_KEY, OPENAI_API_KEY, DEFAULT_MODEL_PROVIDER, DEFAULT_GEMINI_MODEL, DEFAULT_OPENAI_MODEL, DEFAULT_TOP_K
from retrieval.retriever import LegalRetrieverEngine
from rag.context_builder import LegalContextBuilder
from rag.prompt import get_legal_qa_prompt
from rag.answer_validator import AnswerValidator

logger = logging.getLogger(__name__)

GREETINGS = {"hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "hi there", "hello there", "help"}


class LegalRAGChain:
    """Core RAG Orchestrator supporting both Vector Database Retrieval and Direct Document Upload Analysis."""

    def __init__(self, retriever_engine: Optional[LegalRetrieverEngine] = None, model_provider: str = DEFAULT_MODEL_PROVIDER, model_name: str = DEFAULT_GEMINI_MODEL):
        self.model_provider = model_provider.lower()
        self.model_name = model_name
        self.retriever_engine = retriever_engine or LegalRetrieverEngine()
        self.prompt = get_legal_qa_prompt()

    def _get_llm(self, api_key: str, provider: str, model_name: str):
        provider = provider.lower()
        if provider == "gemini":
            keyToUse = api_key or GEMINI_API_KEY
            if not keyToUse or keyToUse == "your_gemini_api_key_here":
                raise ValueError("GEMINI_API_KEY is not configured. Please enter your Gemini API key in .env.")
            logger.info(f"Initializing ChatGoogleGenerativeAI with model {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=keyToUse,
                model=model_name,
                temperature=0.1
            )
        elif provider == "openai":
            keyToUse = api_key or OPENAI_API_KEY
            if not keyToUse or keyToUse == "your_openai_api_key_here":
                raise ValueError("OPENAI_API_KEY is not configured. Please enter your OpenAI API key in .env.")
            logger.info(f"Initializing ChatOpenAI with model {model_name}")
            return ChatOpenAI(
                api_key=keyToUse,
                model=model_name,
                temperature=0.1
            )
        else:
            raise ValueError(f"Unsupported model provider: {provider}")

    def query_custom_document(self, question: str, custom_text: str, filename: str, api_key: str = "", provider: str = "gemini", model_name: str = "gemini-2.5-flash") -> Dict[str, Any]:
        """
        Executes query specifically grounded on an uploaded case/document/image file.
        """
        if not question or not question.strip():
            return {
                "answer": "Please enter a valid legal question.",
                "sources": [],
                "is_grounded": False
            }

        clean_q = question.strip().lower().rstrip("!?.")
        if clean_q in GREETINGS:
            return {
                "answer": f"Hello! I am ready to analyze your uploaded document **({filename})**. Ask any question about its contents!",
                "sources": [],
                "is_grounded": True
            }

        if not custom_text or not custom_text.strip():
            return {
                "answer": f"The uploaded document **{filename}** contains no extractable text. Please ensure it is a valid PDF, image, or text file.",
                "sources": [],
                "is_grounded": False
            }

        # Create Document object from uploaded file
        doc = Document(
            page_content=custom_text[:12000],  # Max chunk for prompt context
            metadata={
                "document_name": filename,
                "page_number": 1,
                "chunk_id": f"upload_{filename}"
            }
        )

        retrieved_docs = [doc]
        context_str = LegalContextBuilder.build_context(retrieved_docs)
        sources = LegalContextBuilder.extract_sources(retrieved_docs)

        try:
            llm = self._get_llm(api_key=api_key, provider=provider, model_name=model_name)
            chain = self.prompt | llm | StrOutputParser()
            raw_answer = chain.invoke({"context": context_str, "question": question})

            return {
                "answer": raw_answer.strip(),
                "sources": sources,
                "is_grounded": True,
                "retrieved_docs": retrieved_docs
            }
        except Exception as e:
            logger.error(f"Error querying custom document: {e}")
            return {
                "answer": f"An error occurred while analyzing {filename}: {str(e)}",
                "sources": sources,
                "is_grounded": False
            }

    def query(self, question: str, top_k: int = DEFAULT_TOP_K, api_key: str = "", provider: str = "gemini", model_name: str = "gemini-2.5-flash") -> Dict[str, Any]:
        """
        Executes full RAG query workflow using the prebuilt Vector Database.
        """
        if not question or not question.strip():
            return {
                "answer": "Please enter a valid legal question.",
                "sources": [],
                "is_grounded": False
            }

        clean_q = question.strip().lower().rstrip("!?.")
        if clean_q in GREETINGS:
            return {
                "answer": "Hello! I am your Legal Document QA Assistant. How can I help you analyze your indexed legal documents today?",
                "sources": [],
                "is_grounded": True
            }

        # Check if retriever engine needs loading
        if not self.retriever_engine.is_loaded:
            self.retriever_engine._load_indexes()

        if not self.retriever_engine.is_loaded:
            return {
                "answer": "Knowledge base index is missing or being initialized. Please click '🔄 Rebuild / Refresh Knowledge Index' in the sidebar.",
                "sources": [],
                "is_grounded": False
            }

        # 1. Retrieval
        try:
            retrieved_docs = self.retriever_engine.get_relevant_documents(question, top_k=top_k)
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return {
                "answer": f"Retrieval failed: {e}",
                "sources": [],
                "is_grounded": False
            }

        # 2. Check context
        if not AnswerValidator.validate_context(retrieved_docs):
            return {
                "answer": AnswerValidator.INSUFFICIENT_EVIDENCE_MSG,
                "sources": [],
                "is_grounded": False
            }

        # 3. Context & Sources
        context_str = LegalContextBuilder.build_context(retrieved_docs)
        sources = LegalContextBuilder.extract_sources(retrieved_docs)

        # 4. LLM Generation
        try:
            llm = self._get_llm(api_key=api_key, provider=provider, model_name=model_name)
            chain = self.prompt | llm | StrOutputParser()
            raw_answer = chain.invoke({"context": context_str, "question": question})

            # 5. Validation
            validation_result = AnswerValidator.post_process_answer(raw_answer, retrieved_docs)
            return {
                "answer": validation_result["answer"],
                "sources": sources if validation_result["is_grounded"] else [],
                "is_grounded": validation_result["is_grounded"],
                "retrieved_docs": retrieved_docs
            }
        except ValueError as ve:
            return {
                "answer": str(ve),
                "sources": sources,
                "is_grounded": False
            }
        except Exception as e:
            logger.error(f"LLM chain execution error: {e}")
            return {
                "answer": f"An error occurred while generating the answer: {str(e)}",
                "sources": sources,
                "is_grounded": False
            }
