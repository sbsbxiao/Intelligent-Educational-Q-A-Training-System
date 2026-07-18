from __future__ import annotations

from tools.base import BaseTool


_TOOL_REGISTRY: dict[str, BaseTool] = {}


def register_tool(tool: BaseTool) -> BaseTool:
    if tool.name in _TOOL_REGISTRY:
        raise ValueError(f"Tool already registered: {tool.name}")
    _TOOL_REGISTRY[tool.name] = tool
    return tool


def get_tool(name: str) -> BaseTool:
    try:
        return _TOOL_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Tool not found: {name}") from exc


def list_tools() -> list[BaseTool]:
    return list(_TOOL_REGISTRY.values())
