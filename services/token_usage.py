from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any
from uuid import uuid4


@dataclass
class TokenUsageSnapshot:
    task_id: str
    scene: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    llm_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _TaskUsageState:
    snapshot: TokenUsageSnapshot

    def add_usage(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> None:
        self.snapshot.prompt_tokens += max(prompt_tokens, 0)
        self.snapshot.completion_tokens += max(completion_tokens, 0)
        self.snapshot.total_tokens += max(total_tokens, 0)
        self.snapshot.cached_tokens += max(cached_tokens, 0)
        self.snapshot.reasoning_tokens += max(reasoning_tokens, 0)
        self.snapshot.llm_calls += 1


class TokenUsageService:
    """Collect token usage for the current logical task."""

    def __init__(self) -> None:
        self._current_task_id: ContextVar[str | None] = ContextVar("token_usage_task_id", default=None)
        self._task_store: dict[str, _TaskUsageState] = {}
        self._lock = Lock()

    def start_task(self, scene: str, task_id: str | None = None) -> TokenUsageSnapshot:
        resolved_task_id = task_id or f"{scene}_{uuid4().hex[:12]}"
        snapshot = TokenUsageSnapshot(task_id=resolved_task_id, scene=scene)
        with self._lock:
            self._task_store[resolved_task_id] = _TaskUsageState(snapshot=snapshot)
        self._current_task_id.set(resolved_task_id)
        return snapshot

    def ensure_task(self, scene: str, task_id: str | None = None) -> TokenUsageSnapshot:
        current_task_id = self._current_task_id.get()
        if current_task_id:
            snapshot = self.get_snapshot(current_task_id)
            if snapshot:
                return snapshot
        return self.start_task(scene=scene, task_id=task_id)

    def activate_task(self, task_id: str) -> None:
        self._current_task_id.set(task_id)

    def get_current_task_id(self) -> str | None:
        return self._current_task_id.get()

    def record_usage(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
        task_id: str | None = None,
    ) -> TokenUsageSnapshot | None:
        resolved_task_id = task_id or self._current_task_id.get()
        if not resolved_task_id:
            return None

        with self._lock:
            state = self._task_store.get(resolved_task_id)
            if not state:
                return None
            state.add_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cached_tokens=cached_tokens,
                reasoning_tokens=reasoning_tokens,
            )
            return TokenUsageSnapshot(**state.snapshot.to_dict())

    def record_langchain_message(
        self,
        message: Any,
        *,
        task_id: str | None = None,
    ) -> TokenUsageSnapshot | None:
        usage = self._extract_usage(message)
        if not usage:
            return None
        return self.record_usage(task_id=task_id, **usage)

    def get_snapshot(self, task_id: str | None = None) -> TokenUsageSnapshot | None:
        resolved_task_id = task_id or self._current_task_id.get()
        if not resolved_task_id:
            return None
        with self._lock:
            state = self._task_store.get(resolved_task_id)
            if not state:
                return None
            return TokenUsageSnapshot(**state.snapshot.to_dict())

    def end_task(self, task_id: str | None = None) -> TokenUsageSnapshot | None:
        resolved_task_id = task_id or self._current_task_id.get()
        if not resolved_task_id:
            return None
        with self._lock:
            state = self._task_store.get(resolved_task_id)
            snapshot = TokenUsageSnapshot(**state.snapshot.to_dict()) if state else None
        if self._current_task_id.get() == resolved_task_id:
            self._current_task_id.set(None)
        return snapshot

    @contextmanager
    def task_scope(self, scene: str, task_id: str | None = None) -> Iterator[TokenUsageSnapshot]:
        existing_task_id = self._current_task_id.get()
        created_new_task = existing_task_id is None
        snapshot = self.ensure_task(scene=scene, task_id=task_id)
        try:
            yield snapshot
        finally:
            if created_new_task:
                self.end_task(snapshot.task_id)
            elif existing_task_id:
                self._current_task_id.set(existing_task_id)

    @staticmethod
    def _extract_usage(message: Any) -> dict[str, int] | None:
        usage_metadata = getattr(message, "usage_metadata", None) or {}
        response_metadata = getattr(message, "response_metadata", None) or {}
        token_usage = response_metadata.get("token_usage", {}) if isinstance(response_metadata, dict) else {}

        prompt_tokens = TokenUsageService._coerce_int(
            usage_metadata.get("input_tokens"),
            token_usage.get("prompt_tokens"),
        )
        completion_tokens = TokenUsageService._coerce_int(
            usage_metadata.get("output_tokens"),
            token_usage.get("completion_tokens"),
        )
        total_tokens = TokenUsageService._coerce_int(
            usage_metadata.get("total_tokens"),
            token_usage.get("total_tokens"),
            prompt_tokens + completion_tokens,
        )

        prompt_details = token_usage.get("prompt_tokens_details", {}) if isinstance(token_usage, dict) else {}
        completion_details = token_usage.get("completion_tokens_details", {}) if isinstance(token_usage, dict) else {}
        input_details = usage_metadata.get("input_token_details", {}) if isinstance(usage_metadata, dict) else {}
        output_details = usage_metadata.get("output_token_details", {}) if isinstance(usage_metadata, dict) else {}

        cached_tokens = TokenUsageService._coerce_int(
            input_details.get("cache_read"),
            prompt_details.get("cached_tokens"),
        )
        reasoning_tokens = TokenUsageService._coerce_int(
            output_details.get("reasoning"),
            completion_details.get("reasoning_tokens"),
        )

        if total_tokens <= 0 and prompt_tokens <= 0 and completion_tokens <= 0:
            return None

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cached_tokens": cached_tokens,
            "reasoning_tokens": reasoning_tokens,
        }

    @staticmethod
    def _coerce_int(*values: Any) -> int:
        for value in values:
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
        return 0


token_usage_service = TokenUsageService()
