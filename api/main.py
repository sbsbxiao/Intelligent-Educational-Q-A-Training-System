from __future__ import annotations

import os
import shutil
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from agents.knowledge_update_agent import ChangeType, DocumentChange
from config import settings
from orchestrator.graph import build_knowledge_graph_workflow
from services.knowledge_graph import KnowledgeGraphService
from services.token_usage import token_usage_service
from services.vector_store import VectorStoreService
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)

vector_store = VectorStoreService()
knowledge_graph = KnowledgeGraphService()
workflows: dict[str, Any] = {}
logger = logging.getLogger("agent_hub.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.upload_dir, exist_ok=True)
    try:
        await vector_store.init()
        logger.info("Vector store initialized")
    except Exception as exc:
        logger.exception("Vector store initialization failed: %s", exc)
    try:
        await knowledge_graph.init()
        logger.info("Knowledge graph initialized")
    except Exception as exc:
        logger.exception("Knowledge graph initialization failed: %s", exc)
    workflows.update(
        build_knowledge_graph_workflow(vector_store=vector_store, knowledge_graph=knowledge_graph)
    )
    yield
    await knowledge_graph.close()


app = FastAPI(
    title="智能教育问答&培训系统",
    description="Education QA and training API with RAG, knowledge graph, and agent workflows.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str


class TokenUsageResponse(BaseModel):
    task_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    llm_calls: int


class QuestionResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    intent: str
    sources: list[dict[str, Any]]
    reasoning_steps: list[str]
    token_usage: TokenUsageResponse


class EducationQuestionRequest(BaseModel):
    question: str


class EducationQuestionResponse(BaseModel):
    question: str
    answer: str
    skill: str
    tools_used: list[str]
    sources: list[dict[str, Any]]
    token_usage: TokenUsageResponse


class IngestResponse(BaseModel):
    file_name: str
    chunks_count: int
    entities_count: int
    relations_count: int
    status: str


class StatsResponse(BaseModel):
    vector_store: dict[str, Any]
    knowledge_graph: dict[str, Any]


class UpdateRequest(BaseModel):
    file_path: str
    change_type: str = "modified"


class UpdateResponse(BaseModel):
    file_path: str
    vectors_added: int
    vectors_deleted: int
    entities_added: int
    relations_added: int
    success: bool
    processing_time_ms: float


@app.post("/api/ingest/upload", response_model=IngestResponse, tags=["ingest"])
async def upload_document(file: UploadFile = File(...)):
    start_time = time.perf_counter()
    save_path = os.path.join(settings.upload_dir, file.filename or "unknown")
    logger.info("Upload started: filename=%s, save_path=%s", file.filename, save_path)
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        file_size = os.path.getsize(save_path)
        logger.info("Upload file saved: filename=%s, size=%s bytes", file.filename, file_size)
    except Exception as exc:
        logger.exception("Upload file save failed: filename=%s, error=%s", file.filename, exc)
        raise

    ingest_wf = workflows.get("ingest")
    if not ingest_wf:
        logger.error("Upload failed: ingest workflow not initialized")
        raise HTTPException(status_code=503, detail="Ingest workflow not initialized")

    logger.info("Ingest workflow started: filename=%s", file.filename)
    result = await ingest_wf.ainvoke({"file_paths": [save_path]})
    logger.info("Ingest workflow finished: filename=%s, elapsed_ms=%.2f", file.filename, (time.perf_counter() - start_time) * 1000)
    chunks = result.get("chunks", [])
    extractions = result.get("extractions", [])
    total_entities = sum(len(e.entities) for e in extractions)
    total_relations = sum(len(e.relations) for e in extractions)
    logger.info(
        "Upload completed: filename=%s, chunks=%s, entities=%s, relations=%s",
        file.filename,
        len(chunks),
        total_entities,
        total_relations,
    )

    return IngestResponse(
        file_name=file.filename or "unknown",
        chunks_count=len(chunks),
        entities_count=total_entities,
        relations_count=total_relations,
        status="success",
    )


@app.post("/api/ingest/batch", response_model=list[IngestResponse], tags=["ingest"])
async def upload_batch(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        resp = await upload_document(file)
        results.append(resp)
    return results


@app.post("/api/qa/ask", response_model=QuestionResponse, tags=["qa"])
async def ask_question(req: QuestionRequest):
    qa_wf = workflows.get("qa")
    if not qa_wf:
        raise HTTPException(status_code=503, detail="QA workflow not initialized")

    with token_usage_service.task_scope(scene="qa") as token_task:
        result = await qa_wf.ainvoke({"question": req.question})
    qa_result = result.get("result")
    if not qa_result:
        raise HTTPException(status_code=500, detail="QA failed")
    token_snapshot = token_usage_service.get_snapshot(token_task.task_id) or token_task

    return QuestionResponse(
        question=qa_result.question,
        answer=qa_result.answer,
        confidence=qa_result.confidence,
        intent=qa_result.intent.value,
        sources=[
            {"content": c.content[:200], "source": c.source, "score": c.score, "type": c.retrieval_type}
            for c in qa_result.contexts
        ],
        reasoning_steps=qa_result.reasoning_steps,
        token_usage=TokenUsageResponse(
            task_id=token_snapshot.task_id,
            prompt_tokens=token_snapshot.prompt_tokens,
            completion_tokens=token_snapshot.completion_tokens,
            total_tokens=token_snapshot.total_tokens,
            cached_tokens=token_snapshot.cached_tokens,
            reasoning_tokens=token_snapshot.reasoning_tokens,
            llm_calls=token_snapshot.llm_calls,
        ),
    )


@app.post("/api/education/ask", response_model=EducationQuestionResponse, tags=["education"])
async def ask_education_question(req: EducationQuestionRequest):
    education_wf = workflows.get("education")
    if not education_wf:
        raise HTTPException(status_code=503, detail="Education workflow not initialized")

    with token_usage_service.task_scope(scene="education") as token_task:
        result = await education_wf.ainvoke({"question": req.question})
    education_result = result.get("result")
    if not education_result:
        raise HTTPException(status_code=500, detail="Education QA failed")
    token_snapshot = token_usage_service.get_snapshot(token_task.task_id) or token_task

    return EducationQuestionResponse(
        question=education_result.question,
        answer=education_result.answer,
        skill=education_result.skill,
        tools_used=education_result.tools_used,
        sources=education_result.sources,
        token_usage=TokenUsageResponse(
            task_id=token_snapshot.task_id,
            prompt_tokens=token_snapshot.prompt_tokens,
            completion_tokens=token_snapshot.completion_tokens,
            total_tokens=token_snapshot.total_tokens,
            cached_tokens=token_snapshot.cached_tokens,
            reasoning_tokens=token_snapshot.reasoning_tokens,
            llm_calls=token_snapshot.llm_calls,
        ),
    )


@app.get("/api/admin/stats", response_model=StatsResponse, tags=["admin"])
async def get_stats():
    vs_stats = await vector_store.get_stats()
    kg_stats = await knowledge_graph.get_stats()
    return StatsResponse(vector_store=vs_stats, knowledge_graph=kg_stats)


@app.post("/api/admin/update", response_model=UpdateResponse, tags=["admin"])
async def trigger_update(req: UpdateRequest):
    update_wf = workflows.get("update")
    if not update_wf:
        raise HTTPException(status_code=503, detail="Update workflow not initialized")

    change = DocumentChange(
        file_path=req.file_path,
        change_type=ChangeType(req.change_type),
    )
    result = await update_wf.ainvoke({"changes": [change]})
    results = result.get("results", [])
    if not results:
        raise HTTPException(status_code=500, detail="Update failed")

    r = results[0]
    return UpdateResponse(
        file_path=r.change.file_path,
        vectors_added=r.vectors_added,
        vectors_deleted=r.vectors_deleted,
        entities_added=r.entities_added,
        relations_added=r.relations_added,
        success=r.success,
        processing_time_ms=r.processing_time_ms,
    )


@app.get("/api/health", tags=["admin"])
async def health():
    return {"status": "ok", "service": "智能教育问答&培训系统"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
