from __future__ import annotations

from typing import Any

from services.vector_store import VectorStoreService
from tools.base import BaseTool, ToolResult


class StudentServicePolicySearchTool(BaseTool):
    name = "student_service_policy_search"
    description = "Search student service rules, course arrangements, homework rules, exam policies, and certificate policies."

    def __init__(self, vector_store: VectorStoreService) -> None:
        self.vector_store = vector_store

    async def arun(self, query: str, top_k: int = 5, **kwargs: Any) -> ToolResult:
        results = await self.vector_store.search(query=query, top_k=top_k)
        return ToolResult(
            success=True,
            data=[
                {
                    "content": doc.get("content", ""),
                    "source": doc.get("source", ""),
                    "score": score,
                    "metadata": doc.get("metadata", {}),
                }
                for doc, score in results
            ],
            metadata={"tool": self.name, "query": query, "top_k": top_k},
        )
