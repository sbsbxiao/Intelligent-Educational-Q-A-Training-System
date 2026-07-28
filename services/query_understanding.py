from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from config import settings
from services.structured_output import QueryRewriteOutput, StructuredOutputAdapter
from services.token_usage import token_usage_service


class IntentClassificationOutput(BaseModel):
    intent: str = "factoid"


INTENT_PROMPT = """你是一个查询意图分类器。根据用户问题识别最合适的意图类别。
可选类别：factoid、analytical、comparative、procedural、exploratory。"""

QUERY_REWRITE_PROMPT = """你是一个查询改写专家。将用户问题改写为更适合检索的形式。
要求：
1. 提取核心实体和关键词，保留课程名、章节名、题型名、机构名等关键标识。
2. 生成 1-3 个检索查询，覆盖原问法、关键词问法、补全后的短语问法。
3. 检索查询必须更适合向量检索和知识图谱检索，但不能偏离原问题语义。
4. 不要编造不存在的事实，不要引入原问题中没有的新领域。
5. 如果原问题已经很适合检索，至少保留原问题本身作为一个查询。"""

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
        result = await self.intent_chain.ainvoke({
            "system_prompt": INTENT_PROMPT,
            "format_instructions": self.intent_parser.format_instructions(),
            "question": question,
        })
        return result.intent

    async def rewrite_query(self, question: str) -> QueryRewriteOutput:
        rewritten = await self.rewrite_chain.ainvoke({
            "system_prompt": QUERY_REWRITE_PROMPT,
            "format_instructions": self.rewrite_parser.format_instructions(),
            "question": question,
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

