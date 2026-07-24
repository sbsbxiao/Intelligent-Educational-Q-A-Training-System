from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import settings
from services.conversation_memory import DEFAULT_SESSION_ID, conversation_memory
from services.knowledge_graph import KnowledgeGraphService
from services.vector_store import VectorStoreService
from skills.base import SkillResult
from skills.course_explanation import CourseExplanationSkill
from skills.question_analysis import QuestionAnalysisSkill
from skills.registry import get_skill, register_skill
from skills.service_qa import ServiceQASkill
from skills.study_plan import StudyPlanSkill
from tools.course_tools import CourseMaterialSearchTool
from tools.graph_tools import KnowledgeGraphQueryTool
from tools.policy_tools import StudentServicePolicySearchTool
from tools.question_tools import QuestionBankSearchTool
from tools.registry import register_tool


ANSWER_WITH_CONTEXT_PROMPT = """你是教育培训机构知识助手。请结合可用上下文回答用户问题。
如果上下文不足，可以基于通用教育知识补充，但要避免编造具体机构资料。"""

DIRECT_ANSWER_PROMPT = """你是教育培训机构知识助手。当前没有可用 RAG 上下文、工具结果或知识库资料。
请直接基于大模型能力回答用户问题；如果问题需要机构内部资料，请明确说明当前缺少资料。"""


@dataclass
class EducationAgentResult:
    question: str
    answer: str
    skill: str
    tools_used: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class EducationAnswerGenerationChain:
    def __init__(self, llm: ChatOpenAI) -> None:
        parser = StrOutputParser()
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", ANSWER_WITH_CONTEXT_PROMPT),
            (
                "human",
                "问题类型/Skill: {skill_name}\n"
                "工具调用: {tools_used}\n"
                "上下文信息:\n{context_text}\n\n"
                "历史对话:\n{history_text}\n\n"
                "用户问题: {question}",
            ),
        ])
        self.direct_answer_prompt = ChatPromptTemplate.from_messages([
            ("system", DIRECT_ANSWER_PROMPT),
            (
                "human",
                "问题类型/Skill: {skill_name}\n"
                "历史对话:\n{history_text}\n\n"
                "用户问题: {question}",
            ),
        ])
        self.answer_chain = self.answer_prompt | llm | parser
        self.direct_answer_chain = self.direct_answer_prompt | llm | parser

    async def generate(
        self,
        *,
        question: str,
        skill_name: str,
        history_text: str,
        context_text: str,
        tools_used: list[str],
    ) -> str:
        payload = {
            "question": question,
            "skill_name": skill_name,
            "history_text": history_text or "无",
            "context_text": context_text,
            "tools_used": ", ".join(tools_used) if tools_used else "无",
        }
        if context_text:
            return await self.answer_chain.ainvoke(payload)
        return await self.direct_answer_chain.ainvoke(payload)


class EducationAgent:
    _VALID_SKILLS = {
        "course_explanation",
        "question_analysis",
        "study_plan",
        "service_qa",
    }

    _ROUTING_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "course_explanation",
                "description": "Use for course explanations, teaching material lookup, chapter summaries, and knowledge point explanations. Uses course_material_search and optionally knowledge_graph_query.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "question_analysis",
                "description": "Use for question analysis, answer explanation, similar question lookup, wrong-question review, and exam point analysis. Uses question_bank_search, course_material_search, and optionally knowledge_graph_query.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "study_plan",
                "description": "Use for study plans, learning paths, exam preparation plans, review order, and beginner learning guidance. Uses course_material_search and optionally knowledge_graph_query.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "service_qa",
                "description": "Use for student service questions, leave requests, homework rules, registration, certificates, course schedules, refunds, and class-hour policies. Uses student_service_policy_search.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
    ]

    def __init__(
        self,
        vector_store: VectorStoreService,
        knowledge_graph: KnowledgeGraphService | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
        )
        self.answer_generation = EducationAnswerGenerationChain(llm=self.llm)
        self._register_tools()
        self._register_skills()

    async def answer(self, question: str, session_id: str = DEFAULT_SESSION_ID, **kwargs: Any) -> EducationAgentResult:
        history_text = self.load_history(session_id=session_id)
        skill_name = await self.route_question(question, history_text)
        skill_result = await self.run_skill(skill_name, question, **kwargs)
        answer = await self.generate_final_answer(question, skill_name, skill_result, history_text)
        self.save_memory(question, answer, skill_name, session_id=session_id)
        result = self.build_result(question, skill_name, skill_result, answer)
        self.save_record(question, answer, skill_name, skill_result, session_id=session_id)
        await conversation_memory.refresh_short_memory(session_id=session_id)
        await conversation_memory.refresh_long_memory(session_id=session_id)
        return result

    def load_history(self, session_id: str = DEFAULT_SESSION_ID) -> str:
        return conversation_memory.format_history_with_short_memory(session_id=session_id)

    async def route_question(self, question: str, history_text: str = "") -> str:
        routed = await self.select_skill_with_llm(question, history_text)
        if self.is_valid_skill(routed):
            return routed
        return self.route_by_rules(question)

    async def select_skill_with_llm(self, question: str, history_text: str = "") -> str:
        try:
            return await self._llm_select_skill(question, history_text)
        except Exception:
            return ""

    def is_valid_skill(self, skill_name: str) -> bool:
        return skill_name in self._VALID_SKILLS

    def route_by_rules(self, question: str) -> str:
        return self._route_by_rules(question)

    async def run_skill(self, skill_name: str, question: str, **kwargs: Any) -> SkillResult:
        try:
            skill = get_skill(skill_name)
            return await skill.arun(question=question, **kwargs)
        except Exception as exc:
            return SkillResult(
                success=False,
                answer="",
                data={},
                tools_used=[],
                sources=[],
                error=str(exc),
                metadata={"skill": skill_name, "fallback_reason": "skill_or_tool_failed"},
            )

    async def generate_final_answer(
        self,
        question: str,
        skill_name: str,
        skill_result: SkillResult,
        history_text: str = "",
    ) -> str:
        context_text = self._format_context(skill_result)
        return await self.answer_generation.generate(
            question=question,
            skill_name=skill_name,
            history_text=history_text,
            context_text=context_text,
            tools_used=skill_result.tools_used,
        )

    def save_memory(
        self,
        question: str,
        answer: str,
        skill_name: str,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> None:
        conversation_memory.append_turn(
            question=question,
            answer=answer,
            session_id=session_id,
            metadata={"agent": "education", "skill": skill_name},
        )

    def save_record(
        self,
        question: str,
        answer: str,
        skill_name: str,
        skill_result: SkillResult,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> None:
        conversation_memory.append_record(
            question=question,
            answer=answer,
            agent="education",
            session_id=session_id,
            metadata={
                "skill": skill_name,
                "tools_used": skill_result.tools_used,
                "sources": skill_result.sources,
                "success": skill_result.success,
                "error": skill_result.error,
            },
        )

    def build_result(
        self,
        question: str,
        skill_name: str,
        skill_result: SkillResult,
        answer: str,
    ) -> EducationAgentResult:
        return self._build_result(question, skill_name, skill_result, answer)

    async def _run_skill_safely(self, skill_name: str, question: str, **kwargs: Any) -> SkillResult:
        return await self.run_skill(skill_name, question, **kwargs)

    async def _generate_final_answer(
        self,
        question: str,
        skill_name: str,
        skill_result: SkillResult,
        history_text: str = "",
    ) -> str:
        return await self.generate_final_answer(question, skill_name, skill_result, history_text)

    @staticmethod
    def _format_context(skill_result: SkillResult) -> str:
        parts: list[str] = []

        valid_sources = [source for source in skill_result.sources if source]
        if valid_sources:
            parts.append("来源信息:\n" + json.dumps(valid_sources[:5], ensure_ascii=False, default=str))

        cleaned_data = EducationAgent._remove_empty_values(skill_result.data)
        if cleaned_data:
            parts.append("工具结果:\n" + json.dumps(cleaned_data, ensure_ascii=False, default=str)[:4000])

        return "\n\n".join(parts).strip()

    @staticmethod
    def _remove_empty_values(value: Any) -> Any:
        if isinstance(value, dict):
            cleaned = {
                key: EducationAgent._remove_empty_values(item)
                for key, item in value.items()
            }
            return {key: item for key, item in cleaned.items() if item not in (None, "", [], {})}
        if isinstance(value, list):
            cleaned_list = [EducationAgent._remove_empty_values(item) for item in value]
            return [item for item in cleaned_list if item not in (None, "", [], {})]
        return value

    async def _llm_select_skill(self, question: str, history_text: str = "") -> str:
        llm_with_tools = self.llm.bind_tools(self._ROUTING_TOOLS, tool_choice="auto")
        response = await llm_with_tools.ainvoke([
            SystemMessage(
                content=(
                    "You are a router for an education knowledge assistant. "
                    "Select exactly one tool that best matches the user's question. "
                    "Do not answer the question directly."
                )
            ),
            HumanMessage(content=f"历史对话:\n{history_text or '无'}\n\n当前问题: {question}"),
        ])
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            return ""
        return tool_calls[0].get("name", "")

    def _route_by_rules(self, question: str) -> str:
        normalized = question.lower()

        if any(k in normalized for k in ["题", "选项", "答案", "解析", "错题", "考点"]):
            return "question_analysis"

        if any(k in normalized for k in ["学习路径", "学习计划", "怎么学", "复习", "备考", "零基础"]):
            return "study_plan"

        if any(k in normalized for k in ["请假", "作业", "报名", "证书", "课时", "退费", "课程安排"]):
            return "service_qa"

        return "course_explanation"

    def _register_tools(self) -> None:
        self._safe_register_tool(CourseMaterialSearchTool(self.vector_store))
        self._safe_register_tool(QuestionBankSearchTool(self.vector_store))
        self._safe_register_tool(StudentServicePolicySearchTool(self.vector_store))
        if self.knowledge_graph:
            self._safe_register_tool(KnowledgeGraphQueryTool(self.knowledge_graph))

    def _register_skills(self) -> None:
        self._safe_register_skill(CourseExplanationSkill())
        self._safe_register_skill(QuestionAnalysisSkill())
        self._safe_register_skill(StudyPlanSkill())
        self._safe_register_skill(ServiceQASkill())

    @staticmethod
    def _safe_register_tool(tool: Any) -> None:
        try:
            register_tool(tool)
        except ValueError:
            pass

    @staticmethod
    def _safe_register_skill(skill: Any) -> None:
        try:
            register_skill(skill)
        except ValueError:
            pass

    @staticmethod
    def _build_result(
        question: str,
        skill_name: str,
        skill_result: SkillResult,
        answer: str,
    ) -> EducationAgentResult:
        metadata = dict(skill_result.metadata)
        if skill_result.error:
            metadata["error"] = skill_result.error
        metadata["used_direct_llm_fallback"] = not bool(skill_result.sources or skill_result.data)

        return EducationAgentResult(
            question=question,
            answer=answer,
            skill=skill_name,
            tools_used=skill_result.tools_used,
            sources=skill_result.sources,
            data=skill_result.data if isinstance(skill_result.data, dict) else {"result": skill_result.data},
            metadata=metadata,
        )

