class QueryPlannerAgent:
    """Agent responsible for classifying query intent and planning retrieval strategy."""

    def plan(self, query: str) -> dict:
        is_legal_query = len(query.strip()) > 5
        needs_retrieval = True
        return {
            "query": query,
            "is_legal_query": is_legal_query,
            "needs_retrieval": needs_retrieval,
            "recommended_top_k": 5
        }
