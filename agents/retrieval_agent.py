from retrieval.retriever import LegalRetrieverEngine

class RetrievalAgent:
    """Agent responsible for executing hybrid document retrieval."""

    def __init__(self, retriever_engine: LegalRetrieverEngine):
        self.engine = retriever_engine

    def execute_retrieval(self, query: str, top_k: int = 5):
        return self.engine.get_relevant_documents(query, top_k=top_k)
