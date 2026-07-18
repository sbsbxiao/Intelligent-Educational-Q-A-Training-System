from __future__ import annotations

import asyncio
import logging
import os
from typing import Protocol

import numpy as np
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger("agent_hub.embeddings")


class EmbeddingsClient(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    async def aembed_query(self, text: str) -> list[float]:
        ...


class LocalSentenceTransformerEmbeddings:
    def __init__(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Local embedding model path does not exist: {model_path}")
        logger.info("Loading local sentence-transformer embedding model: %s", model_path)
        self.model = SentenceTransformer(model_path)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        cleaned = [str(text) for text in texts if str(text).strip()]
        if not cleaned:
            return []
        vectors = self.model.encode(
            cleaned,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        cleaned = str(text).strip()
        if not cleaned:
            return []
        try:
            vector = self.model.encode(
                [cleaned],
                prompt_name="query",
                normalize_embeddings=True,
                convert_to_numpy=True,
            )[0]
        except (KeyError, ValueError):
            vector = self.model.encode(
                [cleaned],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )[0]
        return vector.tolist()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)


class LocalLlamaCppEmbeddings:
    def __init__(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Local GGUF embedding model path does not exist: {model_path}")
        from llama_cpp import Llama

        logger.info("Loading local GGUF embedding model: %s", model_path)
        self.model = Llama(model_path=model_path, embedding=True, verbose=False)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        cleaned = [str(text) for text in texts if str(text).strip()]
        if not cleaned:
            return []
        vectors = self.model.embed(cleaned)
        return self._normalize(vectors)

    def embed_query(self, text: str) -> list[float]:
        cleaned = str(text).strip()
        if not cleaned:
            return []
        vectors = self.model.embed([cleaned])
        normalized = self._normalize(vectors)
        return normalized[0] if normalized else []

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)

    @staticmethod
    def _normalize(vectors: object) -> list[list[float]]:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (arr / norms).tolist()


def create_embeddings() -> EmbeddingsClient:
    if settings.embedding_provider == "local":
        if settings.local_embedding_path.lower().endswith(".gguf"):
            return LocalLlamaCppEmbeddings(settings.local_embedding_path)
        return LocalSentenceTransformerEmbeddings(settings.local_embedding_path)

    logger.info("Using online embedding model: %s", settings.embedding_model)
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
