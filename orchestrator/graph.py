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
from agents.qa_agent import QAAgent, QAResult, QueryIntent
from services.knowledge_graph import KnowledgeGraphService
from services.vector_store import VectorStoreService
from services.conversation_memory import DEFAULT_SESSION_ID, conversation_memory
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
    session_id: str
    memory_context: str
    intent: QueryIntent | None
    rewritten: dict[str, Any]
    vector_contexts: list[Any]
    graph_contexts: list[Any]
    contexts: list[Any]
    answer: str
    reasoning_steps: list[str]
    confidence: float
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

    async def load_memory(state: dict) -> dict:
        session_id = state.get("session_id", DEFAULT_SESSION_ID)
        return {
            "session_id": session_id,
            "memory_context": qa_agent.load_memory_context(session_id=session_id),
        }

    async def classify_intent(state: dict) -> dict:
        question = state.get("question", "")
        intent = await qa_agent.classify_intent(question)
        return {"intent": intent}

    async def rewrite_query(state: dict) -> dict:
        question = state.get("question", "")
        rewritten = await qa_agent.rewrite_query(question)
        return {"rewritten": rewritten}

    async def retrieve_vector_contexts(state: dict) -> dict:
        question = state.get("question", "")
        rewritten = state.get("rewritten", {})
        contexts = await qa_agent.retrieve_vector_contexts(question, rewritten)
        return {"vector_contexts": contexts}

    async def retrieve_graph_contexts(state: dict) -> dict:
        question = state.get("question", "")
        rewritten = state.get("rewritten", {})
        contexts = await qa_agent.retrieve_graph_contexts(question, rewritten)
        return {"graph_contexts": contexts}

    async def fuse_contexts(state: dict) -> dict:
        vector_contexts = state.get("vector_contexts", [])
        graph_contexts = state.get("graph_contexts", [])
        contexts = qa_agent.fuse_contexts(vector_contexts, graph_contexts)
        return {"contexts": contexts}

    async def generate_answer(state: dict) -> dict:
        question = state.get("question", "")
        session_id = state.get("session_id", DEFAULT_SESSION_ID)
        memory_context = state.get("memory_context", "")
        contexts = state.get("contexts", [])
        intent = state.get("intent")
        if not isinstance(intent, QueryIntent):
            intent = QueryIntent.EXPLORATORY

        answer_text, reasoning_steps = await qa_agent.generate_answer(
            question,
            contexts,
            intent,
            session_id=session_id,
            memory_context=memory_context,
        )
        return {
            "intent": intent,
            "answer": answer_text,
            "reasoning_steps": reasoning_steps,
            "confidence": qa_agent.build_result(
                question=question,
                answer=answer_text,
                contexts=contexts,
                intent=intent,
                reasoning_steps=reasoning_steps,
            ).confidence,
        }

    async def save_memory(state: dict) -> dict:
        question = state.get("question", "")
        answer = state.get("answer", "")
        intent = state.get("intent")
        if not isinstance(intent, QueryIntent):
            intent = QueryIntent.EXPLORATORY

        await qa_agent.save_answer_memory(
            question=question,
            answer=answer,
            intent=intent,
            contexts=state.get("contexts", []),
            reasoning_steps=state.get("reasoning_steps", []),
            session_id=state.get("session_id", DEFAULT_SESSION_ID),
        )
        return {"intent": intent}

    async def build_result(state: dict) -> dict:
        intent = state.get("intent")
        if not isinstance(intent, QueryIntent):
            intent = QueryIntent.EXPLORATORY

        result = qa_agent.build_result(
            question=state.get("question", ""),
            answer=state.get("answer", ""),
            contexts=state.get("contexts", []),
            intent=intent,
            reasoning_steps=state.get("reasoning_steps", []),
        )
        return {"result": result}

    graph = StateGraph(dict)
    graph.add_node("load_memory", load_memory)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve_vector_contexts", retrieve_vector_contexts)
    graph.add_node("retrieve_graph_contexts", retrieve_graph_contexts)
    graph.add_node("fuse_contexts", fuse_contexts)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("save_memory", save_memory)
    graph.add_node("build_result", build_result)

    graph.set_entry_point("load_memory")
    graph.add_edge("load_memory", "classify_intent")
    graph.add_edge("classify_intent", "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve_vector_contexts")
    graph.add_edge("rewrite_query", "retrieve_graph_contexts")
    graph.add_edge("retrieve_vector_contexts", "fuse_contexts")
    graph.add_edge("retrieve_graph_contexts", "fuse_contexts")
    graph.add_edge("fuse_contexts", "generate_answer")
    graph.add_edge("generate_answer", "save_memory")
    graph.add_edge("save_memory", "build_result")
    graph.add_edge("build_result", END)

    return graph.compile()


def _build_education_graph(education_agent: EducationAgent | None) -> StateGraph:
    plan_react_graph = _build_plan_react_graph(education_agent) if education_agent else None

    async def initialize_education_request(state: dict) -> dict:
        if not education_agent:
            raise RuntimeError("Education workflow requires vector_store")

        question = state.get("question", "")
        next_state = dict(state)
        next_state["question"] = question
        next_state["is_plan_design"] = _is_plan_design_question(question)
        return next_state

    def route_education_request(state: dict) -> str:
        if bool(state.get("is_plan_design", False)) and plan_react_graph:
            return "plan_design"
        return "standard_answer"

    async def process_standard_education_question(state: dict) -> dict:
        if not education_agent:
            raise RuntimeError("Education workflow requires vector_store")

        question = state.get("question", "")
        result = await education_agent.answer(question)
        return {"result": result}

    async def process_plan_design_question(state: dict) -> dict:
        if not plan_react_graph:
            raise RuntimeError("Plan ReAct workflow not initialized")
        return await plan_react_graph.ainvoke(state)

    graph = StateGraph(dict)
    graph.add_node("initialize_request", initialize_education_request)
    graph.add_node("standard_answer", process_standard_education_question)
    graph.add_node("plan_design_answer", process_plan_design_question)

    graph.set_entry_point("initialize_request")
    graph.add_conditional_edges(
        "initialize_request",
        route_education_request,
        {
            "standard_answer": "standard_answer",
            "plan_design": "plan_design_answer",
        },
    )
    graph.add_edge("standard_answer", END)
    graph.add_edge("plan_design_answer", END)

    return graph.compile()


def _build_plan_react_graph(education_agent: EducationAgent | None) -> StateGraph:

    async def initialize(state: dict) -> dict:
        if not education_agent:
            raise RuntimeError("Plan ReAct workflow requires education_agent")

        question = state.get("question", "")
        next_state = dict(state)
        next_state.setdefault("history_text", "")
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

    async def load_history(state: dict) -> dict:
        if not education_agent:
            raise RuntimeError("Plan ReAct workflow requires education_agent")

        session_id = state.get("session_id", DEFAULT_SESSION_ID)
        return {"history_text": education_agent.load_history(session_id=session_id)}

    async def route_skill(state: dict) -> dict:
        if not education_agent:
            raise RuntimeError("Plan ReAct workflow requires education_agent")

        question = state.get("question", "")
        history_text = state.get("history_text", "")
        candidate_skill = await education_agent.select_skill_with_llm(question, history_text)

        next_state = dict(state)
        next_state["candidate_skill"] = candidate_skill
        next_state["thoughts"] = [
            *state.get("thoughts", []),
            f"Route candidate from LLM: {candidate_skill or 'none'}",
        ]
        return next_state

    async def validate_skill(state: dict) -> dict:
        candidate_skill = state.get("candidate_skill", "")
        is_valid = education_agent.is_valid_skill(candidate_skill) if education_agent else False

        next_state = dict(state)
        next_state["route_is_valid"] = is_valid
        if is_valid:
            next_state["skill_name"] = candidate_skill
            next_state["thoughts"] = [
                *state.get("thoughts", []),
                f"Validated skill route: {candidate_skill}",
            ]
        return next_state

    def route_after_validation(state: dict) -> str:
        if bool(state.get("route_is_valid", False)):
            return "prepare_action"
        return "fallback_route"

    async def fallback_route(state: dict) -> dict:
        if not education_agent:
            raise RuntimeError("Plan ReAct workflow requires education_agent")

        question = state.get("question", "")
        skill_name = education_agent.route_by_rules(question)

        next_state = dict(state)
        next_state["skill_name"] = skill_name
        next_state["thoughts"] = [
            *state.get("thoughts", []),
            f"Fallback rule route: {skill_name}",
        ]
        return next_state

    async def prepare_action(state: dict) -> dict:
        question = state.get("question", "")
        skill_name = state.get("skill_name", "study_plan")
        iteration = int(state.get("iteration", 0)) + 1
        action = {
            "type": "skill",
            "name": skill_name,
            "input": {"question": question},
        }

        next_state = dict(state)
        next_state["iteration"] = iteration
        next_state["actions"] = [*state.get("actions", []), action]
        next_state["thoughts"] = [
            *state.get("thoughts", []),
            f"Iteration {iteration}: select {skill_name} for plan design context retrieval.",
        ]
        return next_state

    async def select_executor(state: dict) -> dict:
        action = (state.get("actions") or [{}])[-1]
        action_type = action.get("type", "skill")
        action_name = action.get("name", "study_plan")
        action_input = action.get("input", {})
        question = action_input.get("question") or state.get("question", "")

        next_state = dict(state)
        next_state["action_type"] = action_type
        next_state["action_name"] = action_name
        next_state["action_input"] = action_input
        next_state["action_question"] = question
        return next_state

    def route_executor(state: dict) -> str:
        if state.get("action_type", "skill") == "tool":
            return "run_tool"
        return "run_skill"

    async def run_tool_action(state: dict) -> dict:
        action_name = state.get("action_name", "")
        action_input = state.get("action_input", {})
        try:
            tool = get_tool(action_name)
            tool_result = await tool.arun(**action_input)
            raw_result = SkillResult(
                success=tool_result.success,
                data=tool_result.data,
                tools_used=[tool.name],
                sources=[],
                error=tool_result.error,
                metadata=tool_result.metadata,
            )
        except Exception as exc:
            raw_result = SkillResult(
                success=False,
                data={},
                tools_used=[action_name],
                sources=[],
                error=str(exc),
                metadata={"fallback_reason": "tool_failed"},
            )

        next_state = dict(state)
        next_state["raw_action_result"] = raw_result
        return next_state

    async def run_skill_action(state: dict) -> dict:
        if not education_agent:
            raise RuntimeError("Plan ReAct workflow requires education_agent")

        action_name = state.get("action_name", "study_plan")
        question = state.get("action_question", state.get("question", ""))
        raw_result = await education_agent.run_skill(action_name, question)

        next_state = dict(state)
        next_state["raw_action_result"] = raw_result
        return next_state

    async def normalize_result(state: dict) -> dict:
        skill_result = state.get("raw_action_result")
        if not isinstance(skill_result, SkillResult):
            skill_result = SkillResult(
                success=False,
                data={},
                tools_used=[],
                sources=[],
                error="invalid_action_result",
                metadata={"fallback_reason": "normalize_failed"},
            )

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

    async def generate_final_answer(state: dict) -> dict:
        if not education_agent:
            raise RuntimeError("Plan ReAct workflow requires education_agent")

        skill_result = state.get("skill_result")
        if not isinstance(skill_result, SkillResult):
            skill_result = SkillResult(success=False, data={}, tools_used=state.get("tools_used", []), sources=state.get("sources", []))

        question = state.get("question", "")
        skill_name = state.get("skill_name") or "study_plan"
        history_text = state.get("history_text", "")
        answer = await education_agent.generate_final_answer(question, skill_name, skill_result, history_text)
        return {"answer": answer, "skill_result": skill_result}

    async def build_final_result(state: dict) -> dict:
        if not education_agent:
            raise RuntimeError("Plan ReAct workflow requires education_agent")

        skill_result = state.get("skill_result")
        if not isinstance(skill_result, SkillResult):
            skill_result = SkillResult(success=False, data={}, tools_used=state.get("tools_used", []), sources=state.get("sources", []))

        question = state.get("question", "")
        skill_name = state.get("skill_name") or "study_plan"
        answer = state.get("answer", "")
        result = education_agent.build_result(question, skill_name, skill_result, answer)
        return {"result": result, "skill_result": skill_result}

    async def save_conversation_result(state: dict) -> dict:
        if not education_agent:
            raise RuntimeError("Plan ReAct workflow requires education_agent")

        skill_result = state.get("skill_result")
        if not isinstance(skill_result, SkillResult):
            skill_result = SkillResult(success=False, data={}, tools_used=state.get("tools_used", []), sources=state.get("sources", []))

        question = state.get("question", "")
        answer = state.get("answer", "")
        skill_name = state.get("skill_name") or "study_plan"
        session_id = state.get("session_id", DEFAULT_SESSION_ID)
        education_agent.save_memory(question, answer, skill_name, session_id=session_id)
        education_agent.save_record(question, answer, skill_name, skill_result, session_id=session_id)
        return {"skill_result": skill_result}

    async def refresh_conversation_memory(state: dict) -> dict:
        session_id = state.get("session_id", DEFAULT_SESSION_ID)
        await conversation_memory.refresh_short_memory(session_id=session_id)
        await conversation_memory.refresh_long_memory(session_id=session_id)
        return {}

    graph = StateGraph(dict)
    graph.add_node("initialize", initialize)
    graph.add_node("load_history", load_history)
    graph.add_node("route_skill", route_skill)
    graph.add_node("validate_skill", validate_skill)
    graph.add_node("fallback_route", fallback_route)
    graph.add_node("prepare_action", prepare_action)
    graph.add_node("select_executor", select_executor)
    graph.add_node("run_tool", run_tool_action)
    graph.add_node("run_skill", run_skill_action)
    graph.add_node("normalize_result", normalize_result)
    graph.add_node("observe", observe)
    graph.add_node("verify", verify)
    graph.add_node("generate_final_answer", generate_final_answer)
    graph.add_node("build_final_result", build_final_result)
    graph.add_node("save_conversation_result", save_conversation_result)
    graph.add_node("refresh_conversation_memory", refresh_conversation_memory)

    graph.set_entry_point("initialize")
    graph.add_edge("initialize", "load_history")
    graph.add_edge("load_history", "route_skill")
    graph.add_edge("route_skill", "validate_skill")
    graph.add_conditional_edges("validate_skill", route_after_validation, {"prepare_action": "prepare_action", "fallback_route": "fallback_route"})
    graph.add_edge("fallback_route", "prepare_action")
    graph.add_edge("prepare_action", "select_executor")
    graph.add_conditional_edges("select_executor", route_executor, {"run_tool": "run_tool", "run_skill": "run_skill"})
    graph.add_edge("run_tool", "normalize_result")
    graph.add_edge("run_skill", "normalize_result")
    graph.add_edge("normalize_result", "observe")

    graph.add_edge("observe", "verify")
    graph.add_conditional_edges("verify", should_continue_plan, {"continue": "route_skill", "final": "generate_final_answer"})
    graph.add_edge("generate_final_answer", "build_final_result")
    graph.add_edge("build_final_result", "save_conversation_result")
    graph.add_edge("save_conversation_result", "refresh_conversation_memory")
    graph.add_edge("refresh_conversation_memory", END)

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








