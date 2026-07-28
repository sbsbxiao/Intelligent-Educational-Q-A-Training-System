from __future__ import annotations

from collections.abc import Callable

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from config import settings


ANSWER_PROMPT = """你是一个专业的企业知识问答助手。请基于检索到的上下文回答用户问题。

回答要求：
1. 只把检索上下文中的信息当作事实依据，记忆信息只能辅助理解，不得替代检索证据。
2. 如果上下文能够支持答案，就直接给出简洁、明确的结论。
3. 如果上下文只能支持部分答案，要明确区分“已知信息”和“信息不足”的部分。
4. 如果上下文不足以支持关键结论，必须明确说明“根据当前检索结果，信息不足以确认”，不要补造项目内部事实。
5. 回答中尽量标注来源，使用类似 [来源: xxx] 的形式；如果一条结论来自多个来源，可以合并标注。
6. 如果上下文存在冲突或表述不一致，说明这是基于当前检索结果的综合判断，不要伪造确定性结论。
7. 保持专业、准确、简洁，优先直接回答问题本身，不要展开无关内容。"""

DIRECT_ANSWER_PROMPT = """你是一个有帮助的智能助手。当前没有可用的知识库上下文。

回答要求：
1. 可以回答通用知识性问题。
2. 如果问题需要项目知识、企业内部资料、私有文档或明确依赖知识库检索，请直接说明当前缺少可用上下文，无法准确确认。
3. 不要把猜测当成事实，不要伪造来源。
4. 保持简洁、明确。"""


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
                "请严格基于以下检索上下文作答。\n"
                "如果无法从上下文确认，请明确说明信息不足。\n\n"
                "上下文信息:\n{context_text}\n\n用户问题: {question}",
            ),
        ])
        self.direct_answer_prompt = ChatPromptTemplate.from_messages([
            ("system", DIRECT_ANSWER_PROMPT),
            ("system", "记忆增强信息:\n{memory_context}"),
            MessagesPlaceholder("history"),
            (
                "human",
                "当前没有检索上下文。\n"
                "如果问题依赖知识库，请明确说明无法准确确认。\n\n"
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
