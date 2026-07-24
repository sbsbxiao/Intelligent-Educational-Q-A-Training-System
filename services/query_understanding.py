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


INTENT_PROMPT = """你是一个查询意图分类器。根据用户问题识别最合适的意图类别。
可选类别：factoid、analytical、comparative、procedural、exploratory。"""

QUERY_REWRITE_PROMPT = """你是一个查询改写专家。将用户问题改写为更适合检索的形式。
要求：
1. 提取核心实体和关键词
2. 生成 1-3 个检索查询
3. 保持原问题语义，不要编造不存在的领域信息"""

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
        return await self.rewrite_chain.ainvoke({
            "system_prompt": QUERY_REWRITE_PROMPT,
            "format_instructions": self.rewrite_parser.format_instructions(),
            "question": question,
        })

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

        queries = [str(item).strip() for item in parsed.queries if str(item).strip()]
        entities = [str(item).strip() for item in parsed.entities if str(item).strip()]
        keywords = [str(item).strip() for item in parsed.keywords if str(item).strip()]

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
