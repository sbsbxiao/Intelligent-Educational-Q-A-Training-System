from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnableParallel
from pydantic import ConfigDict, Field

from services.knowledge_graph import KnowledgeGraphService
from services.local_text_generation import LocalTextGenerationModel
from services.rerank import SharedRerankService
from services.structured_output import CypherGenerationOutput, StructuredOutputAdapter
from services.vector_store import VectorStoreService


CYPHER_GENERATION_PROMPT = """\\
浣犳槸涓€涓?Neo4j Cypher 鏌ヨ鐢熸垚涓撳銆傛牴鎹敤鎴烽棶棰樺拰鎻愬彇鐨勫疄浣擄紝鐢熸垚 Cypher 鏌ヨ銆?
鐭ヨ瘑鍥捐氨 Schema:
- 鑺傜偣鏍囩: Person, Organization, Technology, Product, Concept, Location
- 鍏崇郴绫诲瀷: belongs_to, works_at, located_in, developed_by, related_to, part_of, uses, depends_on
- 鑺傜偣灞炴€? name, type, description, created_at, version

鐢熸垚 1-2 鏉?Cypher 鏌ヨ锛岃繑鍥?JSON: {"queries": ["MATCH ...", "MATCH ..."]}
鍙繑鍥?JSON锛屼笉瑕佸叾浠栨枃瀛椼€?"""

cypher_generation_parser = StructuredOutputAdapter(CypherGenerationOutput)


class VectorKnowledgeRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_store: VectorStoreService = Field(...)
    top_k: int = Field(default=5)
    rerank_service: SharedRerankService = Field(default_factory=SharedRerankService)

    def _get_relevant_documents(self, query: str | dict[str, Any], *, run_manager: Any = None) -> list[Document]:
        raise NotImplementedError("Use async retrieval.")

    async def _aget_relevant_documents(self, query: str | dict[str, Any], *, run_manager: Any = None) -> list[Document]:
        payload = self._normalize_payload(query)
        candidates = self._build_query_candidates(payload)
        if not candidates:
            return []

        merged: dict[str, Document] = {}
        hits_count: dict[str, int] = {}

        for candidate_query, query_kind in candidates:
            results = await self.vector_store.search(candidate_query, top_k=self.top_k)
            for doc, score in results:
                metadata = {
                    **doc.get("metadata", {}),
                    "source": doc.get("source", "vector_store"),
                    "score": float(score),
                    "retrieval_type": "vector",
                    "matched_query": candidate_query,
                    "query_kind": query_kind,
                    "matched_queries": [candidate_query],
                }
                key = self._document_key(doc)
                hits_count[key] = hits_count.get(key, 0) + 1
                existing = merged.get(key)
                if existing is None or float(existing.metadata.get("score", 0.0)) < float(score):
                    merged[key] = Document(page_content=doc.get("content", ""), metadata=metadata)
                elif candidate_query not in existing.metadata.get("matched_queries", []):
                    existing.metadata["matched_queries"] = [*existing.metadata.get("matched_queries", []), candidate_query]

        documents: list[Document] = []
        for key, document in merged.items():
            metadata = dict(document.metadata)
            match_bonus = min(0.05 * max(hits_count.get(key, 1) - 1, 0), 0.15)
            metadata["score"] = min(float(metadata.get("score", 0.0)) + match_bonus, 1.0)
            documents.append(Document(page_content=document.page_content, metadata=metadata))

        documents.sort(key=lambda item: float(item.metadata.get("score", 0.0)), reverse=True)
        return self.rerank_service.rerank_documents(payload, documents, top_k=self.top_k)

    @staticmethod
    def _normalize_payload(query: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(query, dict):
            question = str(query.get("question", "")).strip()
            queries = [str(item).strip() for item in query.get("queries", []) if str(item).strip()]
            keywords = [str(item).strip() for item in query.get("keywords", []) if str(item).strip()]
            return {
                "question": question,
                "queries": queries,
                "keywords": keywords,
            }

        text = str(query).strip()
        return {
            "question": text,
            "queries": [text] if text else [],
            "keywords": [],
        }

    @classmethod
    def _build_query_candidates(cls, payload: dict[str, Any]) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        question = str(payload.get("question", "")).strip()
        queries = [str(item).strip() for item in payload.get("queries", []) if str(item).strip()]
        keywords = [str(item).strip() for item in payload.get("keywords", []) if str(item).strip()]

        if question:
            candidates.append((question, "original"))
        for item in queries:
            candidates.append((item, "rewrite"))
        if keywords:
            candidates.append((" ".join(keywords), "keywords"))

        deduped: list[tuple[str, str]] = []
        seen: set[str] = set()
        for candidate_query, query_kind in candidates:
            key = candidate_query.lower()
            if not candidate_query or key in seen:
                continue
            seen.add(key)
            deduped.append((candidate_query, query_kind))
        return deduped[:4]

    @staticmethod
    def _document_key(doc: dict[str, Any]) -> str:
        metadata = doc.get("metadata", {}) or {}
        return str(metadata.get("doc_id") or metadata.get("source") or doc.get("content", "")[:120])


class GraphKnowledgeRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    knowledge_graph: KnowledgeGraphService = Field(...)
    local_llm: LocalTextGenerationModel = Field(...)
    top_k: int = Field(default=5)
    rerank_service: SharedRerankService = Field(default_factory=SharedRerankService)

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        raise NotImplementedError("Use async retrieval.")

    async def _aget_relevant_documents(self, query: dict[str, Any], *, run_manager: Any = None) -> list[Document]:
        question = str(query.get("question", "")).strip()
        entities = query.get("entities", [])
        if not question:
            return []

        try:
            raw = await self.local_llm.agenerate(
                CYPHER_GENERATION_PROMPT,
                f"闂: {question}\n瀹炰綋: {entities}",
            )
            cypher_data = cypher_generation_parser.parse_or_default(
                raw,
                lambda: CypherGenerationOutput(queries=[]),
            )
        except Exception:
            cypher_data = CypherGenerationOutput(queries=[])

        documents: list[Document] = []
        for cypher in cypher_data.queries[: self.top_k]:
            try:
                records = await self.knowledge_graph.execute_cypher(cypher)
            except Exception:
                continue
            for record in records:
                documents.append(
                    Document(
                        page_content=str(record),
                        metadata={
                            "source": "knowledge_graph",
                            "score": 0.8,
                            "retrieval_type": "graph",
                            "cypher": cypher,
                            "entities": entities,
                        },
                    )
                )
        return self.rerank_service.rerank_documents(query, documents, top_k=self.top_k)


class HybridKnowledgeRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_retriever: VectorKnowledgeRetriever = Field(...)
    graph_retriever: GraphKnowledgeRetriever | None = Field(default=None)
    top_k: int = Field(default=8)
    rerank_service: SharedRerankService = Field(default_factory=SharedRerankService)

    def _get_relevant_documents(self, query: dict[str, Any], *, run_manager: Any = None) -> list[Document]:
        raise NotImplementedError("Use async retrieval.")

    async def _aget_relevant_documents(self, query: str | dict[str, Any], *, run_manager: Any = None) -> list[Document]:
        payload = self._normalize_payload(query)
        branches: dict[str, Any] = {"vector": self.vector_retriever}
        if self.graph_retriever:
            branches["graph"] = RunnableLambda(self._graph_payload) | self.graph_retriever

        parallel = RunnableParallel(**branches)
        results = await parallel.ainvoke(payload)
        merged: list[Document] = []
        for docs in results.values():
            if isinstance(docs, list):
                merged.extend(doc for doc in docs if isinstance(doc, Document))
        coarse_ranked = self._coarse_rank_documents(merged)
        return self.rerank_service.rerank_documents(payload, coarse_ranked, top_k=self.top_k)

    @staticmethod
    def _normalize_payload(query: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(query, dict):
            question = str(query.get("question", "")).strip()
            queries = [str(item).strip() for item in query.get("queries", []) if str(item).strip()]
            entities = [str(item).strip() for item in query.get("entities", []) if str(item).strip()]
            keywords = [str(item).strip() for item in query.get("keywords", []) if str(item).strip()]
            return {
                "question": question,
                "queries": queries or ([question] if question else []),
                "entities": entities,
                "keywords": keywords,
            }
        text = str(query).strip()
        return {"question": text, "queries": [text] if text else [], "entities": [], "keywords": []}

    @staticmethod
    def _graph_payload(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "question": payload.get("question", ""),
            "entities": payload.get("entities", []),
        }

    @staticmethod
    def _coarse_rank_documents(documents: list[Document]) -> list[Document]:
        weight_map = {"vector": 1.0, "graph": 1.2, "hybrid": 1.1}
        unique: list[Document] = []
        seen: set[str] = set()

        for document in documents:
            metadata = dict(document.metadata)
            score = float(metadata.get("score", 0.0))
            retrieval_type = str(metadata.get("retrieval_type", "vector"))
            metadata["score"] = score * weight_map.get(retrieval_type, 1.0)
            key = document.page_content[:100]
            if key in seen:
                continue
            seen.add(key)
            unique.append(Document(page_content=document.page_content, metadata=metadata))

        unique.sort(key=lambda doc: float(doc.metadata.get("score", 0.0)), reverse=True)
        return unique
