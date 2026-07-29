from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool

from config import settings

DEFAULT_SESSION_ID = "default"
logger = logging.getLogger("agent_hub.conversation_memory")

_MAX_RECENT_HISTORY_MESSAGES = 2
_MAX_HISTORY_LINE_CHARS = 120
_MAX_HISTORY_CONTEXT_CHARS = 220
_MAX_SHORT_MEMORY_SUMMARY_CHARS = 320
_MAX_LONG_MEMORY_SUMMARY_CHARS = 220


LONG_MEMORY_SYSTEM_PROMPT = """
你是对话长时记忆提炼器。请把一条历史问答大幅压缩成长期关键信息。
要求：
1. 只保留用户长期目标、偏好、学习基础、持续任务、重要事实。
2. 内容必须比短时记忆更精简。
3. 输出 JSON 对象，字段为 memory。
4. 不要输出额外解释。
""".strip()


SHORT_MEMORY_SYSTEM_PROMPT = """
你是对话短时记忆提炼器。请根据最近的问答记录，提炼成最多 5 条简短问答式记忆。
要求：
1. 只保留对后续回答有帮助的信息。
2. 每条包含 question 和 answer 两个字段。
3. answer 要比原回答更短，但保留关键上下文。
4. 只输出 JSON 数组，不要输出额外解释。
""".strip()


class ConversationMemoryStore:
    def __init__(
        self,
        file_path: str | None = None,
        records_file_path: str | None = None,
        short_memory_file_path: str | None = None,
        long_memory_file_path: str | None = None,
    ) -> None:
        self.file_path = file_path or settings.conversation_history_file
        self.records_file_path = records_file_path or settings.conversation_records_file
        self.short_memory_file_path = short_memory_file_path or settings.short_memory_file
        self.long_memory_file_path = long_memory_file_path or settings.long_memory_file
        self._lock = threading.Lock()
        self._short_memory_model: Any = None
        self._long_memory_model: Any = None

    def load_messages(self, session_id: str = DEFAULT_SESSION_ID) -> list[dict[str, Any]]:
        data = self._read_all(self.file_path)
        messages = data.get(session_id, [])
        return messages if isinstance(messages, list) else []

    def append_turn(
        self,
        question: str,
        answer: str,
        session_id: str = DEFAULT_SESSION_ID,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            data = self._read_all(self.file_path)
            messages = data.get(session_id, [])
            if not isinstance(messages, list):
                messages = []
            now = int(time.time())
            messages.extend([
                {"role": "user", "content": question, "created_at": now, "metadata": metadata or {}},
                {"role": "assistant", "content": answer, "created_at": now, "metadata": metadata or {}},
            ])
            data[session_id] = messages[-settings.max_history_messages:]
            self._write_all(self.file_path, data)

    def format_history(self, session_id: str = DEFAULT_SESSION_ID, max_messages: int | None = None) -> str:
        limit = max_messages or settings.max_history_messages
        messages = self.load_messages(session_id)[-limit:]
        lines: list[str] = []
        for message in messages:
            role = "用户" if message.get("role") == "user" else "助手"
            content = self._limit_text(str(message.get("content", "")).strip(), _MAX_HISTORY_LINE_CHARS)
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def format_history_with_short_memory(self, session_id: str = DEFAULT_SESSION_ID) -> str:
        history_text = self.format_history(session_id=session_id, max_messages=_MAX_RECENT_HISTORY_MESSAGES)
        long_memory_text = self.format_long_memory_summary(session_id=session_id)
        short_memory_text = self.format_short_memory_summary(session_id=session_id)
        parts: list[str] = []
        if long_memory_text:
            parts.append(f"长时记忆摘要:\n{long_memory_text}")
        if short_memory_text:
            parts.append(f"短时记忆摘要:\n{short_memory_text}")
        if history_text:
            parts.append(f"最近历史对话:\n{history_text}")
        return self._limit_text("\n\n".join(parts), 700)

    def build_history_context(self, session_id: str = DEFAULT_SESSION_ID) -> dict[str, str]:
        return {
            "recent_history": self.format_history(session_id=session_id, max_messages=_MAX_RECENT_HISTORY_MESSAGES),
            "short_memory": self.format_short_memory_summary(session_id=session_id),
            "long_memory": self.format_long_memory_summary(session_id=session_id),
        }

    def format_memory_context(self, session_id: str = DEFAULT_SESSION_ID) -> str:
        context = self.build_history_context(session_id=session_id)
        parts: list[str] = []
        if context["long_memory"]:
            parts.append(f"长时记忆摘要:\n{context['long_memory']}")
        if context["short_memory"]:
            parts.append(f"短时记忆摘要:\n{context['short_memory']}")
        if context["recent_history"]:
            parts.append(f"最近历史摘要:\n{context['recent_history']}")
        return self._limit_text("\n\n".join(parts), 700)

    def get_message_history(self, session_id: str = DEFAULT_SESSION_ID) -> "JsonFileChatMessageHistory":
        return JsonFileChatMessageHistory(store=self, session_id=session_id)

    def load_records(self, session_id: str = DEFAULT_SESSION_ID) -> list[dict[str, Any]]:
        data = self._read_all(self.records_file_path)
        records = data.get(session_id, [])
        return records if isinstance(records, list) else []

    def append_record(
        self,
        question: str,
        answer: str,
        agent: str,
        session_id: str = DEFAULT_SESSION_ID,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            data = self._read_all(self.records_file_path)
            records = data.get(session_id, [])
            if not isinstance(records, list):
                records = []
            now = int(time.time())
            record = {
                "turn_index": self._next_turn_index(records),
                "created_at": now,
                "created_at_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                "agent": agent,
                "question": question,
                "answer": answer,
                "metadata": metadata or {},
            }
            records.append(record)
            data[session_id] = self._fifo(records, settings.max_conversation_records)
            self._write_all(self.records_file_path, data)
            return record

    def load_short_memories(self, session_id: str = DEFAULT_SESSION_ID) -> list[dict[str, Any]]:
        return self._load_memory_items(self.short_memory_file_path, session_id)

    def save_short_memories(self, items: list[dict[str, Any]], session_id: str = DEFAULT_SESSION_ID) -> None:
        self._save_memory_items(self.short_memory_file_path, session_id, items, settings.max_short_memory_items)

    def append_short_memory(self, item: dict[str, Any], session_id: str = DEFAULT_SESSION_ID) -> None:
        self._append_memory_item(self.short_memory_file_path, session_id, item, settings.max_short_memory_items)

    def format_short_memory_summary(self, session_id: str = DEFAULT_SESSION_ID, max_chars: int = _MAX_SHORT_MEMORY_SUMMARY_CHARS) -> str:
        memories = self.load_short_memories(session_id=session_id)[-settings.max_short_memory_items:]
        parts: list[str] = []
        for item in memories:
            question = self._limit_text(str(item.get("question", "")).strip(), 80)
            answer = self._limit_text(str(item.get("answer", "")).strip(), 120)
            if question or answer:
                parts.append(f"问：{question}；答：{answer}")
        return self._limit_text("；".join(parts), max_chars)

    def format_long_memory_summary(self, session_id: str = DEFAULT_SESSION_ID, max_chars: int = _MAX_LONG_MEMORY_SUMMARY_CHARS) -> str:
        memories = self.load_long_memories(session_id=session_id)[-settings.max_long_memory_items:]
        parts = [self._limit_text(str(item.get("memory", "")).strip(), 80) for item in memories if str(item.get("memory", "")).strip()]
        return self._limit_text("；".join(parts), max_chars)

    def format_short_memory(self, session_id: str = DEFAULT_SESSION_ID) -> str:
        memories = self.load_short_memories(session_id=session_id)
        lines: list[str] = []
        for index, item in enumerate(memories[-settings.max_short_memory_items:], start=1):
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if question or answer:
                lines.append(f"{index}. 问: {question}\n答: {answer}")
        return "\n".join(lines)

    async def refresh_short_memory(self, session_id: str = DEFAULT_SESSION_ID) -> None:
        records = self.load_records(session_id=session_id)[-settings.max_short_memory_items:]
        if not records:
            self.save_short_memories([], session_id=session_id)
            return

        try:
            model = self._get_short_memory_model()
            raw = await model.agenerate(
                SHORT_MEMORY_SYSTEM_PROMPT,
                json.dumps(
                    [
                        {
                            "turn_index": record.get("turn_index"),
                            "question": record.get("question", ""),
                            "answer": record.get("answer", ""),
                        }
                        for record in records
                    ],
                    ensure_ascii=False,
                ),
            )
            memories = self._parse_short_memory_output(raw, records)
        except Exception as exc:
            logger.exception("Short memory refresh failed: %s", exc)
            memories = self._fallback_short_memories(records)

        self.save_short_memories(memories, session_id=session_id)

    def load_long_memories(self, session_id: str = DEFAULT_SESSION_ID) -> list[dict[str, Any]]:
        return self._load_memory_items(self.long_memory_file_path, session_id)

    def save_long_memories(self, items: list[dict[str, Any]], session_id: str = DEFAULT_SESSION_ID) -> None:
        self._save_memory_items(self.long_memory_file_path, session_id, items, settings.max_long_memory_items)

    def append_long_memory(self, item: dict[str, Any], session_id: str = DEFAULT_SESSION_ID) -> None:
        self._append_memory_item(self.long_memory_file_path, session_id, item, settings.max_long_memory_items)

    def format_long_memory(self, session_id: str = DEFAULT_SESSION_ID) -> str:
        memories = self.load_long_memories(session_id=session_id)
        lines: list[str] = []
        for index, item in enumerate(memories[-settings.max_long_memory_items:], start=1):
            memory = str(item.get("memory", "")).strip()
            if memory:
                lines.append(f"{index}. {memory}")
        return "\n".join(lines)

    async def refresh_long_memory(self, session_id: str = DEFAULT_SESSION_ID) -> None:
        records = self.load_records(session_id=session_id)
        if len(records) <= settings.max_short_memory_items:
            return

        older_records = records[:-settings.max_short_memory_items]
        existing = self.load_long_memories(session_id=session_id)
        summarized_turns = {
            item.get("source_turn_index")
            for item in existing
            if item.get("source_turn_index") is not None
        }
        pending = [
            record
            for record in older_records
            if record.get("turn_index") not in summarized_turns
        ]
        if not pending:
            return

        memories = list(existing)
        for record in pending:
            try:
                model = self._get_long_memory_model()
                raw = await model.agenerate(
                    LONG_MEMORY_SYSTEM_PROMPT,
                    json.dumps(
                        {
                            "turn_index": record.get("turn_index"),
                            "question": record.get("question", ""),
                            "answer": record.get("answer", ""),
                            "metadata": record.get("metadata", {}),
                        },
                        ensure_ascii=False,
                    ),
                )
                memory = self._parse_long_memory_output(raw, record)
            except Exception as exc:
                logger.exception("Long memory refresh failed: %s", exc)
                memory = self._fallback_long_memory(record)
            memories.append(memory)

        self.save_long_memories(memories, session_id=session_id)

    def _get_short_memory_model(self) -> Any:
        if self._short_memory_model is None:
            from services.local_text_generation import create_local_text_generation_model

            self._short_memory_model = create_local_text_generation_model()
        return self._short_memory_model

    def _get_long_memory_model(self) -> Any:
        if self._long_memory_model is None:
            from services.local_text_generation import create_local_text_generation_model

            self._long_memory_model = create_local_text_generation_model()
        return self._long_memory_model

    def _load_memory_items(self, file_path: str, session_id: str) -> list[dict[str, Any]]:
        data = self._read_all(file_path)
        items = data.get(session_id, [])
        return items if isinstance(items, list) else []

    def _save_memory_items(self, file_path: str, session_id: str, items: list[dict[str, Any]], limit: int) -> None:
        with self._lock:
            data = self._read_all(file_path)
            data[session_id] = self._fifo(items, limit)
            self._write_all(file_path, data)

    def _append_memory_item(self, file_path: str, session_id: str, item: dict[str, Any], limit: int) -> None:
        with self._lock:
            data = self._read_all(file_path)
            items = data.get(session_id, [])
            if not isinstance(items, list):
                items = []
            now = int(time.time())
            next_item = {
                "created_at": now,
                "created_at_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                **item,
            }
            items.append(next_item)
            data[session_id] = self._fifo(items, limit)
            self._write_all(file_path, data)

    @staticmethod
    def _parse_short_memory_output(raw: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(cleaned)
        if isinstance(data, dict):
            data = data.get("memories", [])
        if not isinstance(data, list):
            return ConversationMemoryStore._fallback_short_memories(records)

        memories: list[dict[str, Any]] = []
        now = int(time.time())
        for index, item in enumerate(data[: settings.max_short_memory_items]):
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", item.get("summary", ""))).strip()
            if question or answer:
                memories.append({
                    "memory_index": index + 1,
                    "created_at": now,
                    "created_at_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                    "question": question[:220],
                    "answer": answer[:320],
                })
        return memories or ConversationMemoryStore._fallback_short_memories(records)

    @staticmethod
    def _fallback_short_memories(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        now = int(time.time())
        memories: list[dict[str, Any]] = []
        for index, record in enumerate(records[-settings.max_short_memory_items:], start=1):
            memories.append({
                "memory_index": index,
                "created_at": now,
                "created_at_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                "question": str(record.get("question", ""))[:220],
                "answer": str(record.get("answer", ""))[:320],
            })
        return memories

    @staticmethod
    def _parse_long_memory_output(raw: str, record: dict[str, Any]) -> dict[str, Any]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            return ConversationMemoryStore._fallback_long_memory(record)
        memory = str(data.get("memory", data.get("summary", ""))).strip()
        if not memory:
            return ConversationMemoryStore._fallback_long_memory(record)
        now = int(time.time())
        return {
            "source_turn_index": record.get("turn_index"),
            "created_at": now,
            "created_at_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "memory": memory[:220],
        }

    @staticmethod
    def _fallback_long_memory(record: dict[str, Any]) -> dict[str, Any]:
        now = int(time.time())
        question = str(record.get("question", ""))[:90]
        answer = str(record.get("answer", ""))[:120]
        return {
            "source_turn_index": record.get("turn_index"),
            "created_at": now,
            "created_at_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "memory": f"用户曾问：{question}；关键信息：{answer}",
        }

    @staticmethod
    def _limit_text(text: str, max_chars: int) -> str:
        cleaned = " ".join(text.split())
        if max_chars <= 0 or len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rstrip() + "..."

    @staticmethod
    def _fifo(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        return items[-limit:]

    @staticmethod
    def _next_turn_index(records: list[dict[str, Any]]) -> int:
        if not records:
            return 1
        last_index = records[-1].get("turn_index", 0)
        return int(last_index) + 1 if isinstance(last_index, int) else len(records) + 1

    @staticmethod
    def _read_all(file_path: str) -> dict[str, list[dict[str, Any]]]:
        if not os.path.exists(file_path):
            return {}
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _write_all(file_path: str, data: dict[str, list[dict[str, Any]]]) -> None:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class JsonFileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, store: ConversationMemoryStore, session_id: str = DEFAULT_SESSION_ID) -> None:
        self.store = store
        self.session_id = session_id

    @property
    def messages(self) -> list[BaseMessage]:
        raw_messages = self.store.load_messages(session_id=self.session_id)
        messages: list[BaseMessage] = []
        for item in raw_messages:
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        return messages

    def add_messages(self, messages: list[BaseMessage]) -> None:
        if not messages:
            return
        with self.store._lock:
            data = self.store._read_all(self.store.file_path)
            raw_messages = data.get(self.session_id, [])
            if not isinstance(raw_messages, list):
                raw_messages = []
            now = int(time.time())
            for message in messages:
                role = "assistant"
                if isinstance(message, HumanMessage):
                    role = "user"
                raw_messages.append({
                    "role": role,
                    "content": str(message.content),
                    "created_at": now,
                    "metadata": {},
                })
            data[self.session_id] = raw_messages[-settings.max_history_messages:]
            self.store._write_all(self.store.file_path, data)

    def clear(self) -> None:
        with self.store._lock:
            data = self.store._read_all(self.store.file_path)
            data[self.session_id] = []
            self.store._write_all(self.store.file_path, data)


conversation_memory = ConversationMemoryStore()


@tool
def load_conversation_history(session_id: str = DEFAULT_SESSION_ID) -> str:
    """Load recent conversation history for the current session."""
    return conversation_memory.format_memory_context(session_id=session_id)


@tool
def save_conversation_turn(question: str, answer: str, session_id: str = DEFAULT_SESSION_ID) -> str:
    """Save one user question and assistant answer into local conversation history."""
    conversation_memory.append_turn(question=question, answer=answer, session_id=session_id)
    return "saved"
