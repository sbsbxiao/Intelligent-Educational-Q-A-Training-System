from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableParallel
from pydantic import ConfigDict, Field

from services.knowledge_graph import KnowledgeGraphService
from services.local_text_generation import LocalTextGenerationModel
from services.structured_output import CypherGenerationOutput, StructuredOutputAdapter
from services.vector_store import VectorStoreService


CYPHER_GENERATION_PROMPT = """\
你是一个 Neo4j Cypher 查询生成专家。根据用户问题和提取的实体，生成 Cypher 查询。

知识图谱 Schema:
- 节点标签: Person, Organization, Technology, Product, Concept, Location
- 关系类型: belongs_to, works_at, located_in, developed_by, related_to, part_of, uses, depends_on
- 节点属性: name, type, description, created_at, version

生成 1-2 条 Cypher 查询，返回 JSON: {"queries": ["MATCH ...", "MATCH ..."]}
只返回 JSON，不要其他文字。
"""

cypher_generation_parser = StructuredOutputAdapter(CypherGenerationOutput)


class VectorKnowledgeRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_store: VectorStoreService = Field(...)
    top_k: int = Field(default=5)

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        raise NotImplementedError("Use async retrieval.")

    async def _aget_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        results = await self.vector_store.search(query, top_k=self.top_k)
        return [
            Document(
                page_content=doc.get("content", ""),
                metadata={
                    **doc.get("metadata", {}),
                    "source": doc.get("source", "vector_store"),
                    "score": score,
                    "retrieval_type": "vector",
                },
            )
            for doc, score in results
        ]


class GraphKnowledgeRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    knowledge_graph: KnowledgeGraphService = Field(...)
    local_llm: LocalTextGenerationModel = Field(...)
    top_k: int = Field(default=5)

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
                f"问题: {question}\n实体: {entities}",
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
                        },
                    )
                )
        return documents[: self.top_k]


class HybridKnowledgeRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_retriever: VectorKnowledgeRetriever = Field(...)
    graph_retriever: GraphKnowledgeRetriever | None = Field(default=None)

    def _get_relevant_documents(self, query: dict[str, Any], *, run_manager: Any = None) -> list[Document]:
        raise NotImplementedError("Use async retrieval.")

    async def _aget_relevant_documents(self, query: dict[str, Any], *, run_manager: Any = None) -> list[Document]:
        rewritten_queries = list(query.get("queries", []))
        if not rewritten_queries:
            rewritten_queries = [str(query.get("question", "")).strip()]
        graph_payload = {
            "question": query.get("question", ""),
            "entities": query.get("entities", []),
        }

        branches: dict[str, Any] = {
            f"vector_{index}": self.vector_retriever
            for index, _ in enumerate(rewritten_queries)
        }
        if self.graph_retriever:
            branches["graph"] = self.graph_retriever

        if not branches:
            return []

        parallel = RunnableParallel(**branches)
        parallel_input = {
            f"vector_{index}": rewritten_queries[index]
            for index, _ in enumerate(rewritten_queries)
        }
        if self.graph_retriever:
            parallel_input["graph"] = graph_payload

        results = await parallel.ainvoke(parallel_input)
        merged: list[Document] = []
        for docs in results.values():
            if isinstance(docs, list):
                merged.extend(doc for doc in docs if isinstance(doc, Document))
        return self._rerank_documents(merged)

    @staticmethod
    def _rerank_documents(documents: list[Document]) -> list[Document]:
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
