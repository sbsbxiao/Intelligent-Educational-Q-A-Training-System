from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.doc_parser_agent import DocType, DocumentChunk
from services.embeddings import create_embeddings


@dataclass
class MultimodalSearchResult:
    content: str
    modality: str
    score: float
    metadata: dict[str, Any]


class MultimodalService:
    MODALITY_WEIGHTS: dict[str, float] = {
        DocType.TEXT.value: 1.0,
        DocType.MARKDOWN.value: 1.0,
        DocType.PDF.value: 0.95,
        DocType.TABLE.value: 0.9,
        DocType.IMAGE.value: 0.85,
    }

    def __init__(self) -> None:
        self.embeddings = create_embeddings()

    async def embed_chunks(self, chunks: list[DocumentChunk]) -> list[list[float]]:
        texts = [c.content for c in chunks]
        return await self.embeddings.aembed_documents(texts)

    async def embed_query(self, query: str) -> list[float]:
        return await self.embeddings.aembed_query(query)

    def weighted_rerank(
        self,
        results: list[tuple[DocumentChunk, float]],
    ) -> list[MultimodalSearchResult]:
        reranked: list[MultimodalSearchResult] = []
        for chunk, score in results:
            weight = self.MODALITY_WEIGHTS.get(chunk.doc_type.value, 1.0)
            reranked.append(MultimodalSearchResult(
                content=chunk.content,
                modality=chunk.doc_type.value,
                score=score * weight,
                metadata=chunk.metadata,
            ))
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked
