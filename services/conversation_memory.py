from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from langchain_core.tools import tool

from config import settings

DEFAULT_SESSION_ID = "default"


class ConversationMemoryStore:
    def __init__(self, file_path: str | None = None) -> None:
        self.file_path = file_path or settings.conversation_history_file
        self._lock = threading.Lock()

    def load_messages(self, session_id: str = DEFAULT_SESSION_ID) -> list[dict[str, Any]]:
        data = self._read_all()
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
            data = self._read_all()
            messages = data.get(session_id, [])
            if not isinstance(messages, list):
                messages = []
            now = int(time.time())
            messages.extend([
                {"role": "user", "content": question, "created_at": now, "metadata": metadata or {}},
                {"role": "assistant", "content": answer, "created_at": now, "metadata": metadata or {}},
            ])
            data[session_id] = messages[-settings.max_history_messages:]
            self._write_all(data)

    def format_history(self, session_id: str = DEFAULT_SESSION_ID, max_messages: int | None = None) -> str:
        limit = max_messages or settings.max_history_messages
        messages = self.load_messages(session_id)[-limit:]
        lines: list[str] = []
        for message in messages:
            role = "用户" if message.get("role") == "user" else "助手"
            content = str(message.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _read_all(self) -> dict[str, list[dict[str, Any]]]:
        if not os.path.exists(self.file_path):
            return {}
        try:
            with open(self.file_path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_all(self, data: dict[str, list[dict[str, Any]]]) -> None:
        directory = os.path.dirname(self.file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


conversation_memory = ConversationMemoryStore()


@tool
def load_conversation_history(session_id: str = DEFAULT_SESSION_ID) -> str:
    """Load recent conversation history for the current session."""
    return conversation_memory.format_history(session_id=session_id)


@tool
def save_conversation_turn(question: str, answer: str, session_id: str = DEFAULT_SESSION_ID) -> str:
    """Save one user question and assistant answer into local conversation history."""
    conversation_memory.append_turn(question=question, answer=answer, session_id=session_id)
    return "saved"
