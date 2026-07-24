from __future__ import annotations

from typing import Callable, Generic, TypeVar

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, RootModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class QueryRewriteOutput(BaseModel):
    queries: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class CypherGenerationOutput(BaseModel):
    queries: list[str] = Field(default_factory=list)


class EntityLinkingOutput(BaseModel):
    entities: list[str] = Field(default_factory=list)


class ShortMemoryItem(BaseModel):
    question: str = ""
    answer: str = ""
    summary: str = ""


class ShortMemoryOutput(RootModel[list[ShortMemoryItem]]):
    pass


class LongMemoryOutput(BaseModel):
    memory: str = ""
    summary: str = ""


class ExtractedEntity(BaseModel):
    name: str = ""
    type: str = "Concept"
    description: str = ""


class ExtractedRelation(BaseModel):
    head: str = ""
    relation: str = "related_to"
    tail: str = ""
    confidence: float = 0.5


class ExtractedEvent(BaseModel):
    trigger: str = ""
    type: str = ""
    participants: list[str] = Field(default_factory=list)


class KnowledgeExtractionOutput(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    events: list[ExtractedEvent] = Field(default_factory=list)


class StructuredOutputAdapter(Generic[SchemaT]):
    def __init__(self, schema_type: type[SchemaT]) -> None:
        self.schema_type = schema_type
        self.parser = PydanticOutputParser(pydantic_object=schema_type)

    def parse(self, raw: str) -> SchemaT:
        cleaned = normalize_json_text(raw)
        try:
            return self.parser.parse(cleaned)
        except Exception:
            return self.schema_type.model_validate_json(cleaned)

    def parse_or_default(self, raw: str, default_factory: Callable[[], SchemaT]) -> SchemaT:
        try:
            return self.parse(raw)
        except Exception:
            return default_factory()

    def format_instructions(self) -> str:
        return self.parser.get_format_instructions()


def normalize_json_text(raw: str) -> str:
    cleaned = str(raw).strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    object_start = cleaned.find("{")
    array_start = cleaned.find("[")

    if object_start == -1 and array_start == -1:
        return cleaned

    if array_start == -1 or (object_start != -1 and object_start < array_start):
        start = object_start
        end = cleaned.rfind("}")
    else:
        start = array_start
        end = cleaned.rfind("]")

    if start >= 0 and end >= start:
        cleaned = cleaned[start : end + 1]
    return cleaned.strip()
