from __future__ import annotations

from typing import Any

from services.knowledge_graph import KnowledgeGraphService
from tools.base import BaseTool, ToolResult


class KnowledgeGraphQueryTool(BaseTool):
    name = "knowledge_graph_query"
    description = "Query the education knowledge graph by entity neighbors or explicit Cypher."

    def __init__(self, knowledge_graph: KnowledgeGraphService) -> None:
        self.knowledge_graph = knowledge_graph

    async def arun(
        self,
        entity_name: str = "",
        hops: int = 2,
        cypher: str = "",
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        if cypher:
            records = await self.knowledge_graph.execute_cypher(cypher, params or {})
            return ToolResult(
                success=True,
                data=records,
                metadata={"tool": self.name, "mode": "cypher"},
            )

        if not entity_name:
            return ToolResult(
                success=False,
                error="entity_name or cypher is required",
                metadata={"tool": self.name},
            )

        records = await self.knowledge_graph.get_neighbors(entity_name=entity_name, hops=hops)
        return ToolResult(
            success=True,
            data=records,
            metadata={"tool": self.name, "mode": "neighbors", "entity_name": entity_name, "hops": hops},
        )
