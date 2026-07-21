from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Annotated, Any

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agents.doc_parser_agent import DocParserAgent, DocumentChunk
from agents.education_agent import EducationAgent, EducationAgentResult
from agents.knowledge_extract_agent import ExtractionResult, KnowledgeExtractAgent
from agents.knowledge_update_agent import DocumentChange, KnowledgeUpdateAgent, UpdateResult
from agents.qa_agent import QAAgent, QAResult
from services.knowledge_graph import KnowledgeGraphService
from services.vector_store import VectorStoreService
from skills.base import SkillResult
from tools.registry import get_tool


logger = logging.getLogger("agent_hub.workflow")


class WorkflowType(str, Enum):
    INGEST = "ingest"
    QA = "qa"
    UPDATE = "update"
    EDUCATION = "education"


class IngestState(dict):
    file_paths: list[str]
    chunks: list[DocumentChunk]
    extractions: list[ExtractionResult]
    vectors_stored: int
    entities_stored: int
    messages: Annotated[list, add_messages]


class QAState(dict):
    question: str
    result: QAResult | None
    messages: Annotated[list, add_messages]


class EducationState(dict):
    question: str
    result: EducationAgentResult | None
    messages: Annotated[list, add_messages]


class PlanReactState(dict):
    question: str
    history_text: str
    plan_type: str
    thoughts: list[str]
    actions: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    verification: dict[str, Any]
    skill_name: str
    tools_used: list[str]
    sources: list[dict[str, Any]]
    answer: str
    iteration: int
    max_iterations: int
    is_satisfied: bool
    result: EducationAgentResult | None
    messages: Annotated[list, add_messages]


class UpdateState(dict):
    changes: list[DocumentChange]
    results: list[UpdateResult]
    messages: Annotated[list, add_messages]


def build_knowledge_graph_workflow(
    vector_store: VectorStoreService | None = None,
    knowledge_graph: KnowledgeGraphService | None = None,
) -> dict[str, Any]:
    doc_parser = DocParserAgent()
    extractor = KnowledgeExtractAgent()
    qa_agent = QAAgent(vector_store=vector_store, knowledge_graph=knowledge_graph)
    education_agent = (
        EducationAgent(vector_store=vector_store, knowledge_graph=knowledge_graph)
        if vector_store
        else None
    )
    update_agent = KnowledgeUpdateAgent(
        doc_parser=doc_parser,
        knowledge_extractor=extractor,
        vector_store=vector_store,
        knowledge_graph=knowledge_graph,
    )

    return {
        "ingest": _build_ingest_graph(doc_parser, extractor, vector_store, knowledge_graph),
        "qa": _build_qa_graph(qa_agent),
        "education": _build_education_graph(education_agent),
        "update": _build_update_graph(update_agent),
    }


def _build_ingest_graph(
    doc_parser: DocParserAgent,
    extractor: KnowledgeExtractAgent,
    vector_store: VectorStoreService | None,
    knowledge_graph: KnowledgeGraphService | None,
) -> StateGraph:

    async def parse_documents(state: dict) -> dict:
        file_paths = state.get("file_paths", [])
        start_time = time.perf_counter()
        logger.info("Ingest parse started: files=%s", file_paths)
        try:
            chunks = await doc_parser.parse_batch(file_paths)
            logger.info("Ingest parse finished: chunks=%s, elapsed_ms=%.2f", len(chunks), (time.perf_counter() - start_time) * 1000)
            next_state = dict(state)
            next_state["chunks"] = chunks
            return next_state
        except Exception as exc:
            logger.exception("Ingest parse failed: error=%s", exc)
            raise

    async def extract_knowledge(state: dict) -> dict:
        chunks = state.get("chunks", [])
        start_time = time.perf_counter()
        logger.info("Ingest extract started: chunks=%s", len(chunks))
        try:
            extractions = await extractor.extract(chunks)
            entities_count = sum(len(e.entities) for e in extractions)
            relations_count = sum(len(e.relations) for e in extractions)
            logger.info(
                "Ingest extract finished: extractions=%s, entities=%s, relations=%s, elapsed_ms=%.2f",
                len(extractions),
                entities_count,
                relations_count,
                (time.perf_counter() - start_time) * 1000,
            )
            next_state = dict(state)
            next_state["extractions"] = extractions
            return next_state
        except Exception as exc:
            logger.exception("Ingest extract failed: error=%s", exc)
            raise

    async def store_vectors(state: dict) -> dict:
        chunks = state.get("chunks", [])
        start_time = time.perf_counter()
        logger.info("Ingest vector store started: chunks=%s", len(chunks))
        try:
            count = 0
            if vector_store and chunks:
                count = await vector_store.add_chunks(chunks)
            logger.info("Ingest vector store finished: vectors=%s, elapsed_ms=%.2f", count, (time.perf_counter() - start_time) * 1000)
            next_state = dict(state)
            next_state["vectors_stored"] = count
            return next_state
        except Exception as exc:
            logger.exception("Ingest vector store failed: error=%s", exc)
            raise

    async def store_graph(state: dict) -> dict:
        extractions = state.get("extractions", [])
        start_time = time.perf_counter()
        total_entities = sum(len(ext.entities) for ext in extractions)
        total_relations = sum(len(ext.relations) for ext in extractions)
        logger.info(
            "Ingest graph store started: extractions=%s, entities=%s, relations=%s",
            len(extractions),
            total_entities,
            total_relations,
        )
        try:
            entity_count = 0
            relation_count = 0
            if knowledge_graph:
                for ext_index, ext in enumerate(extractions, start=1):
                    logger.info(
                        "Graph store extraction progress: %s/%s, entities=%s, relations=%s",
                        ext_index,
                        len(extractions),
                        len(ext.entities),
                        len(ext.relations),
                    )
                    for ent in ext.entities:
                        await knowledge_graph.upsert_entity(ent)
                        entity_count += 1
                        if entity_count == total_entities or entity_count % 20 == 0:
                            logger.info("Graph entity store progress: %s/%s", entity_count, total_entities)
                    for rel in ext.relations:
                        await knowledge_graph.add_relation(rel)
                        relation_count += 1
                        if relation_count == total_relations or relation_count % 20 == 0:
                            logger.info("Graph relation store progress: %s/%s", relation_count, total_relations)
            logger.info(
                "Ingest graph store finished: entities=%s, relations=%s, elapsed_ms=%.2f",
                entity_count,
                relation_count,
                (time.perf_counter() - start_time) * 1000,
            )
            next_state = dict(state)
            next_state["entities_stored"] = entity_count
            next_state["relations_stored"] = relation_count
            return next_state
        except Exception as exc:
            logger.exception("Ingest graph store failed: error=%s", exc)
            raise

    graph = StateGraph(dict)
    graph.add_node("parse", parse_documents)
    graph.add_node("extract", extract_knowledge)
    graph.add_node("store_vectors", store_vectors)
    graph.add_node("store_graph", store_graph)

    graph.set_entry_point("parse")
    graph.add_edge("parse", "extract")
    graph.add_edge("extract", "store_vectors")
    graph.add_edge("store_vectors", "store_graph")
    graph.add_edge("store_graph", END)

    return graph.compile()


def _build_qa_graph(qa_agent: QAAgent) -> StateGraph:

    async def process_question(state: dict) -> dict:
        question = state.get("question", "")
        result = await qa_agent.answer(question)
        return {"result": result}

    graph = StateGraph(dict)
    graph.add_node("answer", process_question)
    graph.set_entry_point("answer")
    graph.add_edge("answer", END)

    return graph.compile()


def _build_education_graph(education_agent: EducationAgent | None) -> StateGraph:
    plan_react_graph = _build_plan_react_graph(education_agent) if education_agent else None

    async def process_education_question(state: dict) -> dict:
        if not education_agent:
            raise RuntimeError("Education workflow requires vector_store")

        question = state.get("question", "")
        if _is_plan_design_question(question) and plan_react_graph:
            return await plan_react_graph.ainvoke(state)

        result = await education_agent.answer(question)
        return {"result": result}

    graph = StateGraph(dict)
    graph.add_node("education_answer", process_education_question)
    graph.set_entry_point("education_answer")
    graph.add_edge("education_answer", END)

    return graph.compile()


def _build_plan_react_graph(education_agent: EducationAgent | None) -> StateGraph:

    async def initialize(state: dict) -> dict:
        if not education_agent:
            raise RuntimeError("Plan ReAct workflow requires education_agent")

        question = state.get("question", "")
        next_state = dict(state)
        next_state.setdefault("history_text", education_agent.load_history())
        next_state.setdefault("plan_type", _detect_plan_type(question))
        next_state.setdefault("thoughts", [])
        next_state.setdefault("actions", [])
        next_state.setdefault("observations", [])
        next_state.setdefault("verification", {})
        next_state.setdefault("skill_name", "")
        next_state.setdefault("tools_used", [])
        next_state.setdefault("sources", [])
        next_state.setdefault("answer", "")
        next_state.setdefault("iteration", 0)
        next_state.setdefault("max_iterations", 3)
        next_state.setdefault("is_satisfied", False)
        next_state.setdefault("result", None)
        return next_state

    async def think(state: dict) -> dict:
        if not education_agent:
            raise RuntimeError("Plan ReAct workflow requires education_agent")

        question = state.get("question", "")
        history_text = state.get("history_text", "")
        skill_name = await education_agent.route_question(question, history_text)
        iteration = int(state.get("iteration", 0)) + 1
        thought = f"Iteration {iteration}: select {skill_name} for plan design context retrieval."
        action = {
            "type": "skill",
            "name": skill_name,
            "input": {"question": question},
        }

        next_state = dict(state)
        next_state["iteration"] = iteration
        next_state["skill_name"] = skill_name
        next_state["thoughts"] = [*state.get("thoughts", []), thought]
        next_state["actions"] = [*state.get("actions", []), action]
        return next_state

    async def act(state: dict) -> dict:
        if not education_agent:
            raise RuntimeError("Plan ReAct workflow requires education_agent")

        action = (state.get("actions") or [{}])[-1]
        action_type = action.get("type", "skill")
        action_name = action.get("name", "study_plan")
        action_input = action.get("input", {})
        question = action_input.get("question") or state.get("question", "")

        if action_type == "tool":
            try:
                tool = get_tool(action_name)
                tool_result = await tool.arun(**action_input)
                skill_result = SkillResult(
                    success=tool_result.success,
                    data=tool_result.data,
                    tools_used=[tool.name],
                    sources=[],
                    error=tool_result.error,
                    metadata=tool_result.metadata,
                )
            except Exception as exc:
                skill_result = SkillResult(
                    success=False,
                    data={},
                    tools_used=[action_name],
                    sources=[],
                    error=str(exc),
                    metadata={"fallback_reason": "tool_failed"},
                )
        else:
            skill_result = await education_agent.run_skill(action_name, question)

        next_state = dict(state)
        next_state["skill_result"] = skill_result
        next_state["tools_used"] = _merge_unique(state.get("tools_used", []), skill_result.tools_used)
        next_state["sources"] = [*state.get("sources", []), *skill_result.sources]
        return next_state

    async def observe(state: dict) -> dict:
        skill_result = state.get("skill_result")
        observation = {
            "iteration": state.get("iteration", 0),
            "skill": state.get("skill_name", ""),
            "success": bool(getattr(skill_result, "success", False)),
            "tools_used": list(getattr(skill_result, "tools_used", [])),
            "sources_count": len(getattr(skill_result, "sources", [])),
            "has_data": bool(getattr(skill_result, "data", None)),
            "error": getattr(skill_result, "error", ""),
        }

        next_state = dict(state)
        next_state["observations"] = [*state.get("observations", []), observation]
        return next_state

    async def verify(state: dict) -> dict:
        skill_result = state.get("skill_result")
        has_context = bool(getattr(skill_result, "sources", []) or getattr(skill_result, "data", None))
        reached_limit = int(state.get("iteration", 0)) >= int(state.get("max_iterations", 3))
        is_satisfied = has_context or reached_limit
        verification = {
            "is_satisfied": is_satisfied,
            "has_context": has_context,
            "reached_limit": reached_limit,
            "reason": "context_available" if has_context else "max_iterations_reached" if reached_limit else "need_more_context",
        }

        next_state = dict(state)
        next_state["is_satisfied"] = is_satisfied
        next_state["verification"] = verification
        return next_state

    def should_continue_plan(state: dict) -> str:
        if bool(state.get("is_satisfied", False)):
            return "final"
        if int(state.get("iteration", 0)) >= int(state.get("max_iterations", 3)):
            return "final"
        return "continue"

    async def final_answer(state: dict) -> dict:
        if not education_agent:
            raise RuntimeError("Plan ReAct workflow requires education_agent")

        skill_result = state.get("skill_result")
        if not isinstance(skill_result, SkillResult):
            skill_result = SkillResult(success=False, data={}, tools_used=state.get("tools_used", []), sources=state.get("sources", []))

        question = state.get("question", "")
        skill_name = state.get("skill_name") or "study_plan"
        history_text = state.get("history_text", "")
        answer = await education_agent.generate_final_answer(question, skill_name, skill_result, history_text)
        result = education_agent.build_result(question, skill_name, skill_result, answer)

        next_state = dict(state)
        next_state["answer"] = answer
        next_state["result"] = result
        return next_state

    graph = StateGraph(dict)
    graph.add_node("initialize", initialize)
    graph.add_node("think", think)
    graph.add_node("act", act)
    graph.add_node("observe", observe)
    graph.add_node("verify", verify)
    graph.add_node("final_answer", final_answer)

    graph.set_entry_point("initialize")
    graph.add_edge("initialize", "think")
    graph.add_edge("think", "act")
    graph.add_edge("act", "observe")
    graph.add_edge("observe", "verify")
    graph.add_conditional_edges("verify", should_continue_plan, {"continue": "think", "final": "final_answer"})
    graph.add_edge("final_answer", END)

    return graph.compile()


def _is_plan_design_question(question: str) -> bool:
    return any(
        keyword in question
        for keyword in [
            "学习路径",
            "学习计划",
            "课程大纲",
            "备考计划",
            "复习计划",
            "知识点讲解方案",
            "题目解析方案",
            "方案设计",
            "请生成",
        ]
    )


def _detect_plan_type(question: str) -> str:
    if "课程大纲" in question:
        return "course_outline"
    if "备考" in question or "复习" in question:
        return "exam_preparation"
    if "知识点讲解方案" in question:
        return "knowledge_explanation_plan"
    if "题目解析方案" in question:
        return "question_analysis_plan"
    return "study_path"


def _merge_unique(existing: list[str], incoming: list[str]) -> list[str]:
    merged: list[str] = []
    for item in [*existing, *incoming]:
        if item and item not in merged:
            merged.append(item)
    return merged

def _build_update_graph(update_agent: KnowledgeUpdateAgent) -> StateGraph:

    async def process_updates(state: dict) -> dict:
        changes = state.get("changes", [])
        results = await update_agent.process_batch(changes)
        return {"results": results}

    def should_continue(state: dict) -> str:
        results = state.get("results", [])
        failed = [r for r in results if not r.success]
        if failed:
            return "retry"
        return "done"

    async def retry_failed(state: dict) -> dict:
        results = state.get("results", [])
        failed_changes = [r.change for r in results if not r.success]
        retried = await update_agent.process_batch(failed_changes)
        all_results = [r for r in results if r.success] + retried
        return {"results": all_results}

    graph = StateGraph(dict)
    graph.add_node("process", process_updates)
    graph.add_node("retry", retry_failed)

    graph.set_entry_point("process")
    graph.add_conditional_edges("process", should_continue, {"retry": "retry", "done": END})
    graph.add_edge("retry", END)

    return graph.compile()









