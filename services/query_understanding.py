from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from config import settings
from services.structured_output import QueryRewriteOutput, StructuredOutputAdapter


class IntentClassificationOutput(BaseModel):
    intent: str = "factoid"


INTENT_PROMPT = """你是查询意图分类器。请把问题归类为 factoid、analytical、comparative、procedural、exploratory 之一。"""

QUERY_REWRITE_PROMPT = """你是查询改写助手。请输出更适合检索的查询、实体和关键词。
要求：保留核心实体，不偏离原问题，不编造新事实；查询数量控制在 1-3 个。"""

_VALID_INTENTS = {"factoid", "analytical", "comparative", "procedural", "exploratory"}


class QueryUnderstandingChains:
    def __init__(self, llm: ChatOpenAI | None = None) -> None:
        self.llm = llm or ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
        )
        self.intent_parser = StructuredOutputAdapter(IntentClassificationOutput)
        self.rewrite_parser = StructuredOutputAdapter(QueryRewriteOutput)

        self.intent_prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}\n\n请严格按照以下格式输出：\n{format_instructions}"),
            ("human", "{question}"),
        ])
        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}\n\n请严格按照以下格式输出：\n{format_instructions}"),
            ("human", "{question}"),
        ])

        self.intent_chain = self.intent_prompt | self.llm | RunnableLambda(self._parse_intent_message)
        self.rewrite_chain = self.rewrite_prompt | self.llm | RunnableLambda(self._parse_rewrite_message)

    async def classify_intent(self, question: str) -> str:
        quick_intent = self._quick_classify_intent(question)
        if quick_intent:
            return quick_intent

        result = await self.intent_chain.ainvoke({
            "system_prompt": INTENT_PROMPT,
            "format_instructions": self.intent_parser.format_instructions(),
            "question": self._limit_text(question, 220),
        })
        return result.intent

    async def rewrite_query(self, question: str) -> QueryRewriteOutput:
        if self._should_skip_rewrite(question):
            return self._build_lightweight_rewrite(question)

        rewritten = await self.rewrite_chain.ainvoke({
            "system_prompt": QUERY_REWRITE_PROMPT,
            "format_instructions": self.rewrite_parser.format_instructions(),
            "question": self._limit_text(question, 280),
        })
        queries = self._dedupe_keep_order([question, *rewritten.queries])
        return QueryRewriteOutput(
            queries=queries[:3],
            entities=self._dedupe_keep_order(rewritten.entities),
            keywords=self._dedupe_keep_order(rewritten.keywords),
        )

    async def build_retrieval_payload(self, question: str) -> dict[str, Any]:
        rewritten = await self.rewrite_query(question)
        return {
            "question": question,
            "queries": rewritten.queries or [question],
            "entities": rewritten.entities,
            "keywords": rewritten.keywords,
        }

    def _parse_intent_message(self, message: BaseMessage) -> IntentClassificationOutput:
        parsed = self.intent_parser.parse_or_default(
            self._message_text(message),
            lambda: IntentClassificationOutput(intent="factoid"),
        )
        intent = str(parsed.intent).strip().lower()
        if intent not in _VALID_INTENTS:
            intent = "factoid"
        return IntentClassificationOutput(intent=intent)

    def _parse_rewrite_message(self, message: BaseMessage) -> QueryRewriteOutput:
        parsed = self.rewrite_parser.parse_or_default(
            self._message_text(message),
            lambda: QueryRewriteOutput(queries=[], entities=[], keywords=[]),
        )

        queries = self._dedupe_keep_order(str(item).strip() for item in parsed.queries if str(item).strip())
        entities = self._dedupe_keep_order(str(item).strip() for item in parsed.entities if str(item).strip())
        keywords = self._dedupe_keep_order(str(item).strip() for item in parsed.keywords if str(item).strip())

        return QueryRewriteOutput(
            queries=queries[:3],
            entities=entities,
            keywords=keywords,
        )

    @staticmethod
    def _message_text(message: BaseMessage) -> str:
        content: Any = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(str(item) for item in content)
        return str(content)

    @staticmethod
    def _dedupe_keep_order(items: Any) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for item in items:
            normalized = str(item).strip()
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            seen.add(key)
            output.append(normalized)
        return output

    @staticmethod
    def _quick_classify_intent(question: str) -> str | None:
        normalized = question.lower()
        if any(token in normalized for token in ["对比", "区别", "不同", "哪个好"]):
            return "comparative"
        if any(token in normalized for token in ["步骤", "流程", "如何", "怎么", "怎么办"]):
            return "procedural"
        if any(token in normalized for token in ["原因", "为什么", "分析", "影响"]):
            return "analytical"
        if any(token in normalized for token in ["有哪些", "总结", "全面", "介绍"]):
            return "exploratory"
        if len(question.strip()) <= 18:
            return "factoid"
        return None

    @staticmethod
    def _should_skip_rewrite(question: str) -> bool:
        compact = " ".join(question.split())
        return len(compact) <= 24 and all(mark not in compact for mark in ["，", ",", "；", ";", "？", "?", "。"])

    @classmethod
    def _build_lightweight_rewrite(cls, question: str) -> QueryRewriteOutput:
        compact = " ".join(question.split())
        keywords = [part for part in cls._dedupe_keep_order(compact.replace("，", " ").replace("。", " ").split(" ")) if len(part) >= 2]
        entities = [item for item in keywords if any(ch.isupper() for ch in item) or len(item) >= 4][:3]
        return QueryRewriteOutput(
            queries=[compact],
            entities=entities,
            keywords=keywords[:6],
        )

    @staticmethod
    def _limit_text(text: str, max_chars: int) -> str:
        cleaned = " ".join(str(text).split())
        if max_chars <= 0 or len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rstrip() + "..."
