from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.education_agent import EducationAgentResult
from orchestrator.graph import _build_education_graph
from skills.base import SkillResult


class DemoEducationAgent:
    def __init__(self) -> None:
        self.answer_called = 0
        self.run_skill_called = 0
        self.last_skill_name = ""

    def load_history(self, session_id: str = "default") -> str:
        return ""

    async def route_question(self, question: str, history_text: str = "") -> str:
        return "study_plan"

    async def run_skill(self, skill_name: str, question: str, **kwargs: Any) -> SkillResult:
        self.run_skill_called += 1
        self.last_skill_name = skill_name
        return SkillResult(
            success=True,
            answer="study plan context retrieved",
            data={"course_materials": [{"content": "Python basic course", "source": "demo.md"}]},
            tools_used=["course_material_search"],
            sources=[{"source": "demo.md", "score": 0.9, "metadata": {"chunk_index": 0}}],
            metadata={"skill": skill_name},
        )

    async def generate_final_answer(
        self,
        question: str,
        skill_name: str,
        skill_result: SkillResult,
        history_text: str = "",
    ) -> str:
        return f"Generated answer for {skill_name}: {question}"

    def build_result(
        self,
        question: str,
        skill_name: str,
        skill_result: SkillResult,
        answer: str,
    ) -> EducationAgentResult:
        return EducationAgentResult(
            question=question,
            answer=answer,
            skill=skill_name,
            tools_used=skill_result.tools_used,
            sources=skill_result.sources,
            data=skill_result.data if isinstance(skill_result.data, dict) else {"result": skill_result.data},
            metadata={"iteration_checked": True},
        )

    async def answer(self, question: str, **kwargs: Any) -> EducationAgentResult:
        self.answer_called += 1
        return EducationAgentResult(
            question=question,
            answer=f"Normal education answer: {question}",
            skill="course_explanation",
            tools_used=[],
            sources=[],
        )


async def main() -> None:
    agent = DemoEducationAgent()
    workflow = _build_education_graph(agent)

    plan_result_state = await workflow.ainvoke({"question": "请生成学习路径设计。目标：零基础学习 Python。学习周期：30 天。"})
    plan_result = plan_result_state["result"]
    print("=== plan design question ===")
    print("answer:", plan_result.answer)
    print("skill:", plan_result.skill)
    print("tools_used:", plan_result.tools_used)
    print("sources:", plan_result.sources)
    print("iterations:", plan_result_state.get("iteration"))
    print("entered_react:", agent.run_skill_called > 0)

    normal_result_state = await workflow.ainvoke({"question": "请解释 Python 函数的作用。"})
    normal_result = normal_result_state["result"]
    print("\n=== normal education question ===")
    print("answer:", normal_result.answer)
    print("skill:", normal_result.skill)
    print("tools_used:", normal_result.tools_used)
    print("sources:", normal_result.sources)
    print("used_normal_answer:", agent.answer_called > 0)


if __name__ == "__main__":
    asyncio.run(main())

