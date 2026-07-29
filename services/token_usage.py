from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
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
    model: str = ""

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
        model: str = "",
    ) -> None:
        self.snapshot.prompt_tokens += max(prompt_tokens, 0)
        self.snapshot.completion_tokens += max(completion_tokens, 0)
        self.snapshot.total_tokens += max(total_tokens, 0)
        self.snapshot.cached_tokens += max(cached_tokens, 0)
        self.snapshot.reasoning_tokens += max(reasoning_tokens, 0)
        self.snapshot.llm_calls += 1
        if model and not self.snapshot.model:
            self.snapshot.model = model


class TokenUsageStorage:
    """Persist token usage details and hourly aggregates to local files."""

    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        self._detail_dir = project_root / "logs" / "token_usage"
        self._hourly_dir = project_root / "logs" / "token_usage_hourly"
        self._lock = Lock()

    def persist_snapshot(self, snapshot: TokenUsageSnapshot) -> None:
        now = datetime.now()
        detail_record = self._build_record(snapshot, now)
        hourly_record = self._build_record(snapshot, now)

        with self._lock:
            self._detail_dir.mkdir(parents=True, exist_ok=True)
            self._hourly_dir.mkdir(parents=True, exist_ok=True)
            self._append_detail_record(detail_record)
            self._update_hourly_record(hourly_record)

    def _append_detail_record(self, record: dict[str, Any]) -> None:
        detail_path = self._detail_dir / f"{record['date']}.jsonl"
        with detail_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _update_hourly_record(self, record: dict[str, Any]) -> None:
        hourly_path = self._hourly_dir / f"{record['date']}.json"
        if hourly_path.exists():
            with hourly_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        else:
            payload = {"date": record["date"], "hours": {}}

        hour_key = record["hour"]
        current = payload.setdefault("hours", {}).get(
            hour_key,
            {
                "date": record["date"],
                "hour": hour_key,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "llm_calls": 0,
                "tasks": 0,
                "scenes": {},
                "models": {},
            },
        )

        current["prompt_tokens"] += record["prompt_tokens"]
        current["completion_tokens"] += record["completion_tokens"]
        current["total_tokens"] += record["total_tokens"]
        current["cached_tokens"] += record["cached_tokens"]
        current["reasoning_tokens"] += record["reasoning_tokens"]
        current["llm_calls"] += record["llm_calls"]
        current["tasks"] += 1

        scene_entry = current.setdefault("scenes", {}).get(
            record["scene"],
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "llm_calls": 0,
                "tasks": 0,
            },
        )
        scene_entry["prompt_tokens"] += record["prompt_tokens"]
        scene_entry["completion_tokens"] += record["completion_tokens"]
        scene_entry["total_tokens"] += record["total_tokens"]
        scene_entry["cached_tokens"] += record["cached_tokens"]
        scene_entry["reasoning_tokens"] += record["reasoning_tokens"]
        scene_entry["llm_calls"] += record["llm_calls"]
        scene_entry["tasks"] += 1
        current["scenes"][record["scene"]] = scene_entry

        model_key = record["model"] or "unknown"
        model_entry = current.setdefault("models", {}).get(
            model_key,
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "llm_calls": 0,
                "tasks": 0,
            },
        )
        model_entry["prompt_tokens"] += record["prompt_tokens"]
        model_entry["completion_tokens"] += record["completion_tokens"]
        model_entry["total_tokens"] += record["total_tokens"]
        model_entry["cached_tokens"] += record["cached_tokens"]
        model_entry["reasoning_tokens"] += record["reasoning_tokens"]
        model_entry["llm_calls"] += record["llm_calls"]
        model_entry["tasks"] += 1
        current["models"][model_key] = model_entry

        payload["hours"][hour_key] = current
        with hourly_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    @staticmethod
    def _build_record(snapshot: TokenUsageSnapshot, now: datetime) -> dict[str, Any]:
        return {
            "timestamp": now.isoformat(timespec="seconds"),
            "date": now.strftime("%Y-%m-%d"),
            "hour": now.strftime("%H"),
            "task_id": snapshot.task_id,
            "scene": snapshot.scene,
            "prompt_tokens": snapshot.prompt_tokens,
            "completion_tokens": snapshot.completion_tokens,
            "total_tokens": snapshot.total_tokens,
            "cached_tokens": snapshot.cached_tokens,
            "reasoning_tokens": snapshot.reasoning_tokens,
            "llm_calls": snapshot.llm_calls,
            "model": snapshot.model,
        }


class TokenUsageService:
    """Collect token usage for the current logical task."""

    def __init__(self) -> None:
        self._current_task_id: ContextVar[str | None] = ContextVar("token_usage_task_id", default=None)
        self._task_store: dict[str, _TaskUsageState] = {}
        self._lock = Lock()
        self._storage = TokenUsageStorage()

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
        model: str = "",
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
                model=model,
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
            state = self._task_store.pop(resolved_task_id, None)
            snapshot = TokenUsageSnapshot(**state.snapshot.to_dict()) if state else None
        if self._current_task_id.get() == resolved_task_id:
            self._current_task_id.set(None)
        if snapshot:
            self._storage.persist_snapshot(snapshot)
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
    def _extract_usage(message: Any) -> dict[str, Any] | None:
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
        model = TokenUsageService._coerce_str(response_metadata.get("model_name")) if isinstance(response_metadata, dict) else ""

        if total_tokens <= 0 and prompt_tokens <= 0 and completion_tokens <= 0:
            return None

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cached_tokens": cached_tokens,
            "reasoning_tokens": reasoning_tokens,
            "model": model,
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

    @staticmethod
    def _coerce_str(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        return ""


token_usage_service = TokenUsageService()
