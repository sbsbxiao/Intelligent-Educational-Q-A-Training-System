from __future__ import annotations

from typing import Any

from skills.base import BaseSkill, SkillResult
from tools.registry import get_tool


class QuestionAnalysisSkill(BaseSkill):
    name = "question_analysis"
    description = "Analyze questions, answers, explanations, related knowledge points, and common mistakes."

    async def arun(self, question: str, top_k: int = 5, **kwargs: Any) -> SkillResult:
        tools_used: list[str] = []
        sources: list[dict[str, Any]] = []
        data: dict[str, Any] = {}

        question_tool = get_tool("question_bank_search")
        question_result = await question_tool.arun(query=question, top_k=top_k)
        tools_used.append(question_tool.name)
        data["question_matches"] = question_result.data
        sources.extend(_sources_from_items(question_result.data))

        course_tool = get_tool("course_material_search")
        course_result = await course_tool.arun(query=question, top_k=top_k)
        tools_used.append(course_tool.name)
        data["course_context"] = course_result.data
        sources.extend(_sources_from_items(course_result.data))

        entity_name = kwargs.get("entity_name", "")
        if entity_name:
            graph_tool = get_tool("knowledge_graph_query")
            graph_result = await graph_tool.arun(entity_name=entity_name, hops=kwargs.get("hops", 2))
            tools_used.append(graph_tool.name)
            data["graph_context"] = graph_result.data
        else:
            data["graph_context"] = []

        return SkillResult(
            success=True,
            answer="Question bank, course material, and optional graph context have been retrieved for analysis.",
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
