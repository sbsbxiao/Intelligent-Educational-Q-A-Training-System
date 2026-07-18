from __future__ import annotations

from skills.base import BaseSkill


_SKILL_REGISTRY: dict[str, BaseSkill] = {}


def register_skill(skill: BaseSkill) -> BaseSkill:
    if skill.name in _SKILL_REGISTRY:
        raise ValueError(f"Skill already registered: {skill.name}")
    _SKILL_REGISTRY[skill.name] = skill
    return skill


def get_skill(name: str) -> BaseSkill:
    try:
        return _SKILL_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Skill not found: {name}") from exc


def list_skills() -> list[BaseSkill]:
    return list(_SKILL_REGISTRY.values())
