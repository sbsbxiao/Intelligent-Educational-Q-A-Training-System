from __future__ import annotations

from collections.abc import Callable

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from config import settings


ANSWER_PROMPT = """你是一个专业的企业知识问答助手。根据检索到的上下文信息回答用户问题。

要求：
1. 答案必须基于提供的上下文，不要编造
2. 如果上下文信息不足，明确告知用户
3. 引用信息来源（如 [来源: xxx]）
4. 如果涉及多个信息源，综合分析后给出结论
5. 保持专业、准确、简洁"""

DIRECT_ANSWER_PROMPT = """你是一个有帮助的智能助手。请直接回答用户问题。
如果问题需要项目知识但当前没有可用知识库，请简要说明。"""


class AnswerGenerationChain:
    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        history_factory: Callable[[str], BaseChatMessageHistory] | None = None,
    ) -> None:
        self.llm = llm or ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
        )
        self.history_factory = history_factory

        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", ANSWER_PROMPT),
            ("system", "记忆增强信息:\n{memory_context}"),
            MessagesPlaceholder("history"),
            (
                "human",
                "上下文信息:\n{context_text}\n\n用户问题: {question}",
            ),
        ])
        self.direct_answer_prompt = ChatPromptTemplate.from_messages([
            ("system", DIRECT_ANSWER_PROMPT),
            ("system", "记忆增强信息:\n{memory_context}"),
            MessagesPlaceholder("history"),
            (
                "human",
                "当前问题: {question}",
            ),
        ])

        parser = StrOutputParser()
        answer_core = self.answer_prompt | self.llm | parser
        direct_answer_core = self.direct_answer_prompt | self.llm | parser

        if history_factory:
            self.answer_chain = RunnableWithMessageHistory(
                answer_core,
                history_factory,
                input_messages_key="question",
                history_messages_key="history",
            )
            self.direct_answer_chain = RunnableWithMessageHistory(
                direct_answer_core,
                history_factory,
                input_messages_key="question",
                history_messages_key="history",
            )
        else:
            self.answer_chain = answer_core
            self.direct_answer_chain = direct_answer_core

    async def generate(
        self,
        *,
        session_id: str,
        question: str,
        memory_context: str = "",
        context_text: str = "",
        has_context: bool = True,
    ) -> str:
        payload = {
            "question": question,
            "memory_context": memory_context or "无",
            "context_text": context_text,
            "history": [],
        }
        config = {"configurable": {"session_id": session_id}}
        if has_context:
            return await self.answer_chain.ainvoke(payload, config=config)
        return await self.direct_answer_chain.ainvoke(payload, config=config)
