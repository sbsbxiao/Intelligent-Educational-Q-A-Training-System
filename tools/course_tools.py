from __future__ import annotations

from typing import Any

from services.query_understanding import QueryUnderstandingChains
from services.retrievers import VectorKnowledgeRetriever
from services.vector_store import VectorStoreService
from tools.base import BaseTool, ToolResult


class CourseMaterialSearchTool(BaseTool):
    name = "course_material_search"
    description = "Search course materials, lecture notes, outlines, and teaching documents."

    def __init__(self, vector_store: VectorStoreService) -> None:
        self.vector_store = vector_store
        self.query_understanding = QueryUnderstandingChains()

    async def arun(self, query: str, top_k: int = 5, **kwargs: Any) -> ToolResult:
        retriever = VectorKnowledgeRetriever(vector_store=self.vector_store, top_k=top_k)
        try:
            payload = await self.query_understanding.build_retrieval_payload(query)
        except Exception:
            payload = {"question": query, "queries": [query], "entities": [], "keywords": []}

        documents = await retriever.ainvoke(payload)
        return ToolResult(
            success=True,
            data=[
                {
                    "content": getattr(doc, "page_content", ""),
                    "source": getattr(doc, "metadata", {}).get("source", ""),
                    "score": getattr(doc, "metadata", {}).get("score", 0.0),
                    "metadata": getattr(doc, "metadata", {}),
                }
                for doc in documents
            ],
            metadata={
                "tool": self.name,
                "query": query,
                "top_k": top_k,
                "rewritten_queries": payload.get("queries", []),
                "keywords": payload.get("keywords", []),
            },
        )
