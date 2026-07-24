from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
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
    def __init__(self, llm: ChatOpenAI | None = None) -> None:
        self.llm = llm or ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
        )

        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", ANSWER_PROMPT),
            (
                "human",
                "历史对话:\n{history_text}\n\n上下文信息:\n{context_text}\n\n用户问题: {question}",
            ),
        ])
        self.direct_answer_prompt = ChatPromptTemplate.from_messages([
            ("system", DIRECT_ANSWER_PROMPT),
            (
                "human",
                "历史对话:\n{history_text}\n\n当前问题: {question}",
            ),
        ])

        parser = StrOutputParser()
        self.answer_chain = self.answer_prompt | self.llm | parser
        self.direct_answer_chain = self.direct_answer_prompt | self.llm | parser

    async def generate(
        self,
        *,
        question: str,
        history_text: str = "",
        context_text: str = "",
        has_context: bool = True,
    ) -> str:
        payload = {
            "question": question,
            "history_text": history_text or "无",
            "context_text": context_text,
        }
        if has_context:
            return await self.answer_chain.ainvoke(payload)
        return await self.direct_answer_chain.ainvoke(payload)
