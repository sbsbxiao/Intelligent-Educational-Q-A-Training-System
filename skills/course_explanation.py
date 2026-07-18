from __future__ import annotations

from typing import Any

from skills.base import BaseSkill, SkillResult
from tools.registry import get_tool


class CourseExplanationSkill(BaseSkill):
    name = "course_explanation"
    description = "Explain course content or teaching knowledge points using course materials and graph context."

    async def arun(self, question: str, top_k: int = 5, **kwargs: Any) -> SkillResult:
        tools_used: list[str] = []
        sources: list[dict[str, Any]] = []
        data: dict[str, Any] = {}

        course_tool = get_tool("course_material_search")
        course_result = await course_tool.arun(query=question, top_k=top_k)
        tools_used.append(course_tool.name)
        data["course_materials"] = course_result.data
        sources.extend(_sources_from_items(course_result.data))

        graph_items: Any = []
        entity_name = kwargs.get("entity_name", "")
        if entity_name:
            graph_tool = get_tool("knowledge_graph_query")
            graph_result = await graph_tool.arun(entity_name=entity_name, hops=kwargs.get("hops", 2))
            tools_used.append(graph_tool.name)
            graph_items = graph_result.data
        data["graph_context"] = graph_items

        return SkillResult(
            success=True,
            answer="Course material context has been retrieved for explanation.",
            data=data,
            tools_used=tools_used,
            sources=sources,
            metadata={"skill": self.name, "question": question},
        )


def _sources_from_items(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    return [
        {"source": item.get("source", ""), "score": item.get("score"), "metadata": item.get("metadata", {})}
        for item in items
        if isinstance(item, dict)
    ]
