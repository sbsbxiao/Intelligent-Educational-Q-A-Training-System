from __future__ import annotations

import logging
from typing import Any

from agents.doc_parser_agent import DocumentChunk
from config import settings
from services.embeddings import EmbeddingsClient, create_embeddings

logger = logging.getLogger("agent_hub.vector_store")


class VectorStoreService:
    """Unified vector store facade for ChromaDB and PGVector."""

    def __init__(self) -> None:
        self.embeddings: EmbeddingsClient | None = None
        self.collection_name = settings.vector_collection_name
        self._store: Any = None
        self._backend = settings.vector_store_type

    async def init(self) -> None:
        self.embeddings = create_embeddings()
        if self._backend == "chroma":
            await self._init_chroma()
        else:
            await self._init_pgvector()

    async def _init_chroma(self) -> None:
        import chromadb

        client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        self._store = client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def _init_pgvector(self) -> None:
        from langchain_community.vectorstores import PGVector

        if not self.embeddings:
            raise RuntimeError("Embedding client is not initialized")
        self._store = PGVector(
            connection_string=settings.pgvector_dsn,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )

    async def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0
        if not self._store or not self.embeddings:
            logger.warning("Vector store is unavailable, skip adding chunks")
            return 0

        texts = [c.content for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "doc_id": c.doc_id,
                "doc_type": c.doc_type.value,
                "source": c.metadata.get("source", ""),
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]

        try:
            logger.info("Embedding started: chunks=%s, backend=%s", len(chunks), self._backend)
            if self._backend == "chroma":
                vectors = await self.embeddings.aembed_documents(texts)
                logger.info("Embedding finished: chunks=%s, vectors=%s", len(chunks), len(vectors))
                if not vectors:
                    logger.warning("Embedding returned empty vectors, skip adding chunks")
                    return 0
                logger.info("Vector upsert started: collection=%s, vectors=%s", self.collection_name, len(vectors))
                self._store.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
                logger.info("Vector upsert finished: collection=%s, vectors=%s", self.collection_name, len(vectors))
            else:
                await self._store.aadd_texts(texts=texts, metadatas=metadatas, ids=ids)
                logger.info("PGVector add_texts finished: chunks=%s", len(chunks))
        except Exception as exc:
            logger.exception("Vector store add_chunks failed: %s", exc)
            return 0

        return len(chunks)

    async def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        if not self._store or not self.embeddings:
            logger.warning("Vector store is unavailable, skip search")
            return []

        try:
            if self._backend == "chroma":
                q_vec = await self.embeddings.aembed_query(query)
                if not q_vec:
                    return []
                results = self._store.query(
                    query_embeddings=[q_vec],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )
                out: list[tuple[dict, float]] = []
                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                dists = results.get("distances", [[]])[0]
                for doc, meta, dist in zip(docs, metas, dists):
                    score = 1.0 - dist
                    out.append(({"content": doc, "source": meta.get("source", ""), "metadata": meta}, score))
                return out

            results = await self._store.asimilarity_search_with_score(query, k=top_k)
            return [
                ({"content": doc.page_content, "source": doc.metadata.get("source", ""), "metadata": doc.metadata}, score)
                for doc, score in results
            ]
        except Exception as exc:
            logger.exception("Vector store search failed: %s", exc)
            return []

    async def delete_by_doc_id(self, doc_id: str) -> int:
        if not self._store:
            return 0
        if self._backend == "chroma":
            existing = self._store.get(where={"doc_id": doc_id}, include=[])
            ids = existing.get("ids", [])
            if ids:
                self._store.delete(ids=ids)
            return len(ids)
        return 0

    async def get_stats(self) -> dict:
        if not self._store:
            return {"backend": self._backend, "available": False, "collection": self.collection_name}
        if self._backend == "chroma":
            count = self._store.count()
            return {"backend": "chroma", "available": True, "total_vectors": count, "collection": self.collection_name}
        return {"backend": "pgvector", "available": True, "collection": self.collection_name}

