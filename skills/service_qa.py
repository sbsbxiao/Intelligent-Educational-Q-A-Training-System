from __future__ import annotations

from typing import Any

from skills.base import BaseSkill, SkillResult
from tools.registry import get_tool


class ServiceQASkill(BaseSkill):
    name = "service_qa"
    description = "Answer student service questions using service rules and policy documents."

    async def arun(self, question: str, top_k: int = 5, **kwargs: Any) -> SkillResult:
        policy_tool = get_tool("student_service_policy_search")
        policy_result = await policy_tool.arun(query=question, top_k=top_k)

        return SkillResult(
            success=True,
            answer="Student service policy context has been retrieved for service QA.",
            data={"service_policies": policy_result.data},
            tools_used=[policy_tool.name],
            sources=_sources_from_items(policy_result.data),
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
