from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    success: bool
    data: Any = None
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    name: str
    description: str

    @abstractmethod
    async def arun(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError
