from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillResult:
    success: bool
    answer: str = ""
    data: Any = None
    tools_used: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseSkill(ABC):
    name: str
    description: str

    @abstractmethod
    async def arun(self, question: str, **kwargs: Any) -> SkillResult:
        raise NotImplementedError
