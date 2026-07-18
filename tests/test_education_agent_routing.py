from __future__ import annotations

import pytest

from agents.education_agent import EducationAgent
from skills.registry import _SKILL_REGISTRY
from tools.registry import _TOOL_REGISTRY


class FakeVectorStore:
    async def search(self, query: str, top_k: int = 5):
        return [
            (
                {
                    "content": f"matched: {query}",
                    "source": "demo",
                    "metadata": {"top_k": top_k},
                },
                0.9,
            )
        ]


class FakeKnowledgeGraph:
    async def get_neighbors(self, entity_name: str, hops: int = 2):
        return [{"entity": entity_name, "hops": hops}]

    async def execute_cypher(self, cypher: str, params: dict | None = None):
        return [{"cypher": cypher, "params": params or {}}]


@pytest.mark.asyncio
async def test_education_agent_routes_to_skill_and_returns_tools_used():
    _TOOL_REGISTRY.clear()
    _SKILL_REGISTRY.clear()

    agent = EducationAgent(
        vector_store=FakeVectorStore(),
        knowledge_graph=FakeKnowledgeGraph(),
    )

    result = await agent.answer("这道题为什么选 B？")

    assert result.skill == "question_analysis"
    assert "question_bank_search" in result.tools_used
    assert "course_material_search" in result.tools_used
    assert result.sources
