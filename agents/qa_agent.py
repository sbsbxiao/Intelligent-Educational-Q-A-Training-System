"""
QA agent with hybrid retrieval (vector + graph) and answer generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_openai import ChatOpenAI

from config import settings
from services.answer_generation import AnswerGenerationChain
from services.conversation_memory import DEFAULT_SESSION_ID, conversation_memory
from services.local_text_generation import create_local_text_generation_model
from services.query_understanding import QueryUnderstandingChains
from services.retrievers import GraphKnowledgeRetriever, HybridKnowledgeRetriever, VectorKnowledgeRetriever


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
        self.answer_generation = AnswerGenerationChain(
            llm=self.llm,
            history_factory=conversation_memory.get_message_history,
        )
        self.vector_retriever = (
            VectorKnowledgeRetriever(vector_store=self.vector_store, top_k=5)
            if self.vector_store
            else None
        )
        self.graph_retriever = (
            GraphKnowledgeRetriever(
                knowledge_graph=self.knowledge_graph,
                local_llm=self.local_llm,
                top_k=5,
            )
            if self.knowledge_graph
            else None
        )
        self.hybrid_retriever = (
            HybridKnowledgeRetriever(
                vector_retriever=self.vector_retriever,
                graph_retriever=self.graph_retriever,
            )
            if self.vector_retriever
            else None
        )

    async def answer(self, question: str, session_id: str = DEFAULT_SESSION_ID) -> QAResult:
        memory_context = conversation_memory.format_memory_context(session_id=session_id)

        try:
            intent = await self._classify_intent(question)
        except Exception:
            intent = QueryIntent.EXPLORATORY

        try:
            rewritten = await self._rewrite_query(question)
        except Exception:
            rewritten = {"queries": [question], "entities": [], "keywords": []}

        top_contexts = await self._retrieve_contexts(question, rewritten)

        answer_text, reasoning = await self._generate_answer(
            question,
            top_contexts,
            intent,
            session_id=session_id,
            memory_context=memory_context,
        )
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

    async def _retrieve_contexts(self, question: str, rewritten: dict[str, Any]) -> list[RetrievedContext]:
        if not self.hybrid_retriever:
            return []

        try:
            documents = await self.hybrid_retriever.ainvoke(
                {
                    "question": question,
                    "queries": rewritten.get("queries", []),
                    "entities": rewritten.get("entities", []),
                }
            )
        except Exception:
            return []

        contexts: list[RetrievedContext] = []
        for document in documents[:8]:
            metadata = dict(document.metadata)
            contexts.append(
                RetrievedContext(
                    content=document.page_content,
                    source=str(metadata.get("source", "")),
                    score=float(metadata.get("score", 0.0)),
                    retrieval_type=str(metadata.get("retrieval_type", "vector")),
                    metadata=metadata,
                )
            )
        return contexts

    async def _generate_answer(
        self,
        question: str,
        contexts: list[RetrievedContext],
        intent: QueryIntent,
        *,
        session_id: str,
        memory_context: str = "",
    ) -> tuple[str, list[str]]:
        if not contexts:
            answer = await self.answer_generation.generate(
                session_id=session_id,
                question=question,
                memory_context=memory_context,
                context_text="",
                has_context=False,
            )
            return answer, [
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

        answer = await self.answer_generation.generate(
            session_id=session_id,
            question=question,
            memory_context=memory_context,
            context_text=context_text,
            has_context=True,
        )
        reasoning_steps.append("答案生成完成")
        return answer, reasoning_steps

    @staticmethod
    def _calc_confidence(contexts: list[RetrievedContext]) -> float:
        if not contexts:
            return 0.0
        avg_score = sum(c.score for c in contexts) / len(contexts)
        return min(avg_score, 1.0)
