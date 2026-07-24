"""
QA agent with hybrid retrieval (vector + graph) and answer generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings
from services.conversation_memory import DEFAULT_SESSION_ID, conversation_memory
from services.local_text_generation import create_local_text_generation_model
from services.query_understanding import QueryUnderstandingChains


class QueryIntent(str, Enum):
    FACTOID = "factoid"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    EXPLORATORY = "exploratory"


@dataclass
class RetrievedContext:
    content: str
    source: str
    score: float
    retrieval_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QAResult:
    question: str
    answer: str
    contexts: list[RetrievedContext]
    intent: QueryIntent
    confidence: float
    reasoning_steps: list[str] = field(default_factory=list)


CYPHER_GENERATION_PROMPT = """\
你是一个 Neo4j Cypher 查询生成专家。根据用户问题和提取的实体，生成 Cypher 查询。

知识图谱 Schema:
- 节点标签: Person, Organization, Technology, Product, Concept, Location
- 关系类型: belongs_to, works_at, located_in, developed_by, related_to, part_of, uses, depends_on
- 节点属性: name, type, description, created_at, version

生成 1-2 条 Cypher 查询，返回 JSON: {"queries": ["MATCH ...", "MATCH ..."]}
只返回 JSON，不要其他文字。
"""

ANSWER_PROMPT = """\
你是一个专业的企业知识问答助手。根据检索到的上下文信息回答用户问题。

要求：
1. 答案必须基于提供的上下文，不要编造
2. 如果上下文信息不足，明确告知用户
3. 引用信息来源（如 [来源: xxx]）
4. 如果涉及多个信息源，综合分析后给出结论
5. 保持专业、准确、简洁
"""


class QAAgent:
    """QA agent entrypoint."""

    def __init__(
        self,
        vector_store: Any = None,
        knowledge_graph: Any = None,
    ) -> None:
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
        )
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.local_llm = create_local_text_generation_model()
        self.query_understanding = QueryUnderstandingChains(llm=self.llm)

    async def answer(self, question: str, session_id: str = DEFAULT_SESSION_ID) -> QAResult:
        history_text = conversation_memory.format_history_with_short_memory(session_id=session_id)

        try:
            intent = await self._classify_intent(question)
        except Exception:
            intent = QueryIntent.EXPLORATORY

        try:
            rewritten = await self._rewrite_query(question)
        except Exception:
            rewritten = {"queries": [question], "entities": [], "keywords": []}

        vector_contexts = await self._vector_retrieve(rewritten)
        graph_contexts = await self._graph_retrieve(question, rewritten)

        all_contexts = self._hybrid_rerank(vector_contexts + graph_contexts)
        top_contexts = all_contexts[:8]

        answer_text, reasoning = await self._generate_answer(question, top_contexts, intent, history_text)
        conversation_memory.append_turn(
            question=question,
            answer=answer_text,
            session_id=session_id,
            metadata={"agent": "qa", "intent": intent.value},
        )
        conversation_memory.append_record(
            question=question,
            answer=answer_text,
            agent="qa",
            session_id=session_id,
            metadata={
                "intent": intent.value,
                "tools_used": [],
                "sources": [
                    {
                        "content": context.content[:200],
                        "source": context.source,
                        "score": context.score,
                        "type": context.retrieval_type,
                    }
                    for context in top_contexts
                ],
                "confidence": self._calc_confidence(top_contexts),
                "reasoning_steps": reasoning,
            },
        )

        await conversation_memory.refresh_short_memory(session_id=session_id)
        await conversation_memory.refresh_long_memory(session_id=session_id)

        return QAResult(
            question=question,
            answer=answer_text,
            contexts=top_contexts,
            intent=intent,
            confidence=self._calc_confidence(top_contexts),
            reasoning_steps=reasoning,
        )

    async def _classify_intent(self, question: str) -> QueryIntent:
        intent_name = await self.query_understanding.classify_intent(question)
        try:
            return QueryIntent(intent_name)
        except ValueError:
            return QueryIntent.FACTOID

    async def _rewrite_query(self, question: str) -> dict:
        rewritten = await self.query_understanding.rewrite_query(question)
        queries = rewritten.queries or [question]
        return {
            "queries": queries,
            "entities": rewritten.entities,
            "keywords": rewritten.keywords,
        }

    async def _vector_retrieve(self, rewritten: dict) -> list[RetrievedContext]:
        if not self.vector_store:
            return []

        contexts: list[RetrievedContext] = []
        for query in rewritten.get("queries", []):
            try:
                results = await self.vector_store.search(query, top_k=5)
            except Exception:
                continue
            for doc, score in results:
                contexts.append(RetrievedContext(
                    content=doc.get("content", ""),
                    source=doc.get("source", "vector_store"),
                    score=score,
                    retrieval_type="vector",
                    metadata=doc.get("metadata", {}),
                ))
        return contexts

    async def _graph_retrieve(self, question: str, rewritten: dict) -> list[RetrievedContext]:
        if not self.knowledge_graph:
            return []

        import json

        entities = rewritten.get("entities", [])
        try:
            raw = await self.local_llm.agenerate(
                CYPHER_GENERATION_PROMPT,
                f"问题: {question}\n实体: {entities}",
            )
            cleaned = self._clean_json_text(raw)
            cypher_data = json.loads(cleaned)
        except Exception:
            cypher_data = {"queries": []}

        contexts: list[RetrievedContext] = []
        for cypher in cypher_data.get("queries", []):
            try:
                records = await self.knowledge_graph.execute_cypher(cypher)
                for record in records:
                    contexts.append(RetrievedContext(
                        content=str(record),
                        source="knowledge_graph",
                        score=0.8,
                        retrieval_type="graph",
                        metadata={"cypher": cypher},
                    ))
            except Exception:
                continue
        return contexts

    @staticmethod
    def _hybrid_rerank(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        weight_map = {"vector": 1.0, "graph": 1.2, "hybrid": 1.1}
        for ctx in contexts:
            ctx.score *= weight_map.get(ctx.retrieval_type, 1.0)

        seen: set[str] = set()
        unique: list[RetrievedContext] = []
        for ctx in contexts:
            key = ctx.content[:100]
            if key not in seen:
                seen.add(key)
                unique.append(ctx)

        unique.sort(key=lambda c: c.score, reverse=True)
        return unique

    async def _generate_answer(
        self,
        question: str,
        contexts: list[RetrievedContext],
        intent: QueryIntent,
        history_text: str = "",
    ) -> tuple[str, list[str]]:
        if not contexts:
            messages = [
                SystemMessage(
                    content=(
                        "你是一个有帮助的智能助手。请直接回答用户问题。"
                        "如果问题需要项目知识但当前没有可用知识库，请简要说明。"
                    )
                ),
                HumanMessage(content=f"历史对话:\n{history_text or '无'}\n\n当前问题: {question}"),
            ]
            resp = await self.llm.ainvoke(messages)
            return resp.content, [
                f"识别问题意图: {intent.value}",
                "未获得检索上下文，使用大模型直接回答",
                "答案生成完成",
            ]

        context_text = "\n\n".join(
            f"[来源 {i + 1}: {c.source} | 类型: {c.retrieval_type} | 分数: {c.score:.2f}]\n{c.content}"
            for i, c in enumerate(contexts)
        )
        reasoning_steps = [
            f"识别问题意图: {intent.value}",
            f"检索到 {len(contexts)} 条相关上下文",
            f"向量检索: {sum(1 for c in contexts if c.retrieval_type == 'vector')} 条",
            f"图谱检索: {sum(1 for c in contexts if c.retrieval_type == 'graph')} 条",
        ]

        messages = [
            SystemMessage(content=ANSWER_PROMPT),
            HumanMessage(content=f"历史对话:\n{history_text or '无'}\n\n上下文信息:\n{context_text}\n\n用户问题: {question}"),
        ]
        resp = await self.llm.ainvoke(messages)
        reasoning_steps.append("答案生成完成")
        return resp.content, reasoning_steps

    @staticmethod
    def _calc_confidence(contexts: list[RetrievedContext]) -> float:
        if not contexts:
            return 0.0
        avg_score = sum(c.score for c in contexts) / len(contexts)
        return min(avg_score, 1.0)

    @staticmethod
    def _clean_json_text(raw: str) -> str:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end >= start:
            cleaned = cleaned[start : end + 1]
        return cleaned.strip()
