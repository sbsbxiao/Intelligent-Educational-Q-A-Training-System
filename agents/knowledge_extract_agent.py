"""
Knowledge extraction agent.
Extracts entities, relations, and events from document chunks for Neo4j graph storage.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from agents.doc_parser_agent import DocumentChunk
from services.local_text_generation import create_local_text_generation_model

logger = logging.getLogger("agent_hub.knowledge_extract")

EXTRACTION_SYSTEM_PROMPT = """\
你是教育培训知识图谱抽取器。请从课程资料、题库资料、服务规则文本中抽取实体和关系。

示例：
文本：Python 基础课程包含函数章节。函数章节讲解参数传递，参数传递是期末考试的常见考点。
输出：
{
  "entities": [
    {"name": "Python 基础课程", "type": "Course", "description": "课程"},
    {"name": "函数章节", "type": "Chapter", "description": "章节"},
    {"name": "参数传递", "type": "Concept", "description": "知识点"},
    {"name": "期末考试", "type": "Exam", "description": "考试"}
  ],
  "relations": [
    {"head": "Python 基础课程", "relation": "contains", "tail": "函数章节", "confidence": 0.95},
    {"head": "函数章节", "relation": "explains", "tail": "参数传递", "confidence": 0.95},
    {"head": "期末考试", "relation": "tests", "tail": "参数传递", "confidence": 0.9}
  ],
  "events": []
}

实体类型尽量使用：
- Course: 课程
- Chapter: 章节
- Concept: 知识点/概念
- Question: 题目/题型
- ExamPoint: 考点
- Exam: 考试
- Policy: 服务规则/管理规则
- Task: 学习任务/作业任务
- Tool: 工具/框架/软件
- Person, Organization, Location, Time: 人物/机构/地点/时间

关系类型尽量使用：
- contains: 包含
- part_of: 属于
- explains: 讲解/解释
- tests: 考察
- depends_on: 依赖
- prerequisite_of: 是前置知识
- related_to: 相关
- similar_to: 相似
- causes: 导致
- solves: 解决
- uses: 使用
- requires: 要求
- applies_to: 适用于
- follows: 后续步骤/后续章节
- managed_by: 由某规则管理

严格要求：
1. 只返回 JSON，不要返回 Markdown 代码块，不要解释。
2. JSON 顶层必须包含 entities、relations、events 三个字段。
3. 每个 chunk 最多抽取 6 个 entities、8 条 relations、2 个 events。
4. 课程名、章节名、知识点、考点、题目类型、学习任务都要优先抽取。
5. 不要轻易返回空数组；只要文本里有课程、章节、知识点、规则或考点，就必须抽取。
6. 如果文本确实没有可抽取内容，返回 {"entities": [], "relations": [], "events": []}。
"""


@dataclass
class Entity:
    name: str
    type: str
    description: str = ""
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def node_label(self) -> str:
        return self.type.replace(" ", "_")


@dataclass
class Relation:
    head: str
    relation: str
    tail: str
    confidence: float = 0.0
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeEvent:
    trigger: str
    type: str
    participants: list[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    entities: list[Entity]
    relations: list[Relation]
    events: list[KnowledgeEvent]
    source_chunk_id: str = ""


class KnowledgeExtractAgent:
    """Extracts graph-ready knowledge from document chunks using a local Qwen model."""

    BATCH_SIZE = 5

    def __init__(self) -> None:
        self.local_llm = create_local_text_generation_model()

    async def extract(self, chunks: list[DocumentChunk]) -> list[ExtractionResult]:
        results: list[ExtractionResult] = []
        total = len(chunks)
        logger.info("Knowledge extraction progress: 0/%s chunks", total)
        for i in range(0, total, self.BATCH_SIZE):
            batch = chunks[i : i + self.BATCH_SIZE]
            for offset, chunk in enumerate(batch, start=1):
                current = i + offset
                logger.info(
                    "Knowledge extraction chunk started: %s/%s, chunk_id=%s, chars=%s",
                    current,
                    total,
                    chunk.chunk_id,
                    len(chunk.content),
                )
                result = await self._extract_from_chunk(chunk)
                results.append(result)
                logger.info(
                    "Knowledge extraction chunk finished: %s/%s, chunk_id=%s, entities=%s, relations=%s, events=%s",
                    current,
                    total,
                    chunk.chunk_id,
                    len(result.entities),
                    len(result.relations),
                    len(result.events),
                )
        deduped = self._deduplicate(results)
        logger.info("Knowledge extraction progress: %s/%s chunks completed", total, total)
        return deduped

    async def extract_single(self, text: str, chunk_id: str = "") -> ExtractionResult:
        return await self._extract_from_text(text, chunk_id)

    async def _extract_from_chunk(self, chunk: DocumentChunk) -> ExtractionResult:
        return await self._extract_from_text(chunk.content, chunk.chunk_id)

    async def _extract_from_text(self, text: str, source_id: str) -> ExtractionResult:
        user_prompt = f"请从以下文本中抽取知识，并只返回 JSON：\n\n{text}"
        try:
            raw = await self.local_llm.agenerate(EXTRACTION_SYSTEM_PROMPT, user_prompt)
            return self._parse_response(raw, source_id)
        except Exception as exc:
            logger.exception("Local knowledge extraction failed: source_id=%s, error=%s", source_id, exc)
            return ExtractionResult(entities=[], relations=[], events=[], source_chunk_id=source_id)

    def _parse_response(self, raw: str, source_id: str) -> ExtractionResult:
        try:
            cleaned = self._clean_json_text(raw)
            data = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError, TypeError):
            logger.warning("Knowledge extraction returned invalid JSON: source_id=%s", source_id)
            return ExtractionResult(entities=[], relations=[], events=[], source_chunk_id=source_id)

        entities = [
            Entity(
                name=str(e.get("name", "")).strip(),
                type=str(e.get("type", "Concept")).strip() or "Concept",
                description=str(e.get("description", "")).strip(),
            )
            for e in data.get("entities", [])
            if isinstance(e, dict) and str(e.get("name", "")).strip()
        ]
        relations = [
            Relation(
                head=str(r.get("head", "")).strip(),
                relation=str(r.get("relation", "related_to")).strip() or "related_to",
                tail=str(r.get("tail", "")).strip(),
                confidence=self._safe_float(r.get("confidence", 0.5)),
            )
            for r in data.get("relations", [])
            if isinstance(r, dict) and str(r.get("head", "")).strip() and str(r.get("tail", "")).strip()
        ]
        events = [
            KnowledgeEvent(
                trigger=str(ev.get("trigger", "")).strip(),
                type=str(ev.get("type", "")).strip(),
                participants=ev.get("participants", []) if isinstance(ev.get("participants", []), list) else [],
            )
            for ev in data.get("events", [])
            if isinstance(ev, dict)
        ]
        return ExtractionResult(
            entities=entities,
            relations=relations,
            events=events,
            source_chunk_id=source_id,
        )

    @staticmethod
    def _clean_json_text(raw: str) -> str:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end >= start:
            cleaned = cleaned[start : end + 1]
        return cleaned.strip()

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.5

    @staticmethod
    def _deduplicate(results: list[ExtractionResult]) -> list[ExtractionResult]:
        seen_entities: dict[str, Entity] = {}
        seen_relations: set[tuple[str, str, str]] = set()
        deduped: list[ExtractionResult] = []

        for result in results:
            unique_entities: list[Entity] = []
            for ent in result.entities:
                key = f"{ent.name}::{ent.type}"
                if key not in seen_entities:
                    seen_entities[key] = ent
                    unique_entities.append(ent)

            unique_relations: list[Relation] = []
            for rel in result.relations:
                key = (rel.head, rel.relation, rel.tail)
                if key not in seen_relations:
                    seen_relations.add(key)
                    unique_relations.append(rel)

            deduped.append(ExtractionResult(
                entities=unique_entities,
                relations=unique_relations,
                events=result.events,
                source_chunk_id=result.source_chunk_id,
            ))
        return deduped


