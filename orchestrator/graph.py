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

    async def process_education_question(state: dict) -> dict:
        if not education_agent:
            raise RuntimeError("Education workflow requires vector_store")

        question = state.get("question", "")
        result = await education_agent.answer(question)
        return {"result": result}

    graph = StateGraph(dict)
    graph.add_node("education_answer", process_education_question)
    graph.set_entry_point("education_answer")
    graph.add_edge("education_answer", END)

    return graph.compile()


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




