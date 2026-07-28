from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document


class SharedRerankService:
    """Shared reranker for QA and education retrieval contexts."""

    def rerank_documents(
        self,
        query: str | dict[str, Any],
        documents: list[Document],
        *,
        top_k: int | None = None,
    ) -> list[Document]:
        payload = self._normalize_payload(query)
        reranked: list[Document] = []

        for document in documents:
            metadata = dict(document.metadata)
            coarse_score = float(metadata.get("score", 0.0))
            rerank_score = self._score_document(payload, document, coarse_score)
            metadata["coarse_score"] = coarse_score
            metadata["rerank_score"] = rerank_score
            metadata["score"] = rerank_score
            reranked.append(Document(page_content=document.page_content, metadata=metadata))

        reranked.sort(
            key=lambda doc: (
                float(doc.metadata.get("rerank_score", 0.0)),
                float(doc.metadata.get("coarse_score", 0.0)),
            ),
            reverse=True,
        )
        if top_k is not None:
            return reranked[:top_k]
        return reranked

    @staticmethod
    def _normalize_payload(query: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(query, dict):
            return {
                "question": str(query.get("question", "")).strip(),
                "queries": [str(item).strip() for item in query.get("queries", []) if str(item).strip()],
                "entities": [str(item).strip() for item in query.get("entities", []) if str(item).strip()],
                "keywords": [str(item).strip() for item in query.get("keywords", []) if str(item).strip()],
            }
        text = str(query).strip()
        return {"question": text, "queries": [text] if text else [], "entities": [], "keywords": []}

    def _score_document(self, payload: dict[str, Any], document: Document, coarse_score: float) -> float:
        metadata = dict(document.metadata)
        content_text = str(document.page_content or "")
        content_lower = content_text.lower()
        metadata_text = " ".join(
            str(metadata.get(field, ""))
            for field in ["source", "file_name", "section_title", "parent_section"]
            if metadata.get(field)
        ).lower()

        query_terms = self._collect_terms(payload)
        overlap_score = self._term_overlap_score(query_terms, content_text)
        exact_hit_score = self._exact_hit_score(payload, content_lower, metadata_text)
        structure_score = self._structure_score(payload, metadata_text)
        matched_queries = metadata.get("matched_queries", [])
        multi_query_bonus = min(max(len(matched_queries) - 1, 0) * 0.04, 0.12)

        final_score = (
            coarse_score * 0.55
            + overlap_score * 0.25
            + exact_hit_score * 0.12
            + structure_score * 0.08
            + multi_query_bonus
        )
        return min(final_score, 1.0)

    def _collect_terms(self, payload: dict[str, Any]) -> set[str]:
        text_parts = [
            payload.get("question", ""),
            *payload.get("queries", []),
            *payload.get("entities", []),
            *payload.get("keywords", []),
        ]
        terms: set[str] = set()
        for part in text_parts:
            for token in self._tokenize(str(part)):
                if token:
                    terms.add(token)
        return terms

    def _term_overlap_score(self, query_terms: set[str], content_text: str) -> float:
        if not query_terms:
            return 0.0
        content_terms = set(self._tokenize(content_text))
        if not content_terms:
            return 0.0
        overlap = len(query_terms & content_terms)
        return overlap / max(len(query_terms), 1)

    @staticmethod
    def _exact_hit_score(payload: dict[str, Any], content_lower: str, metadata_text: str) -> float:
        targets = [
            *payload.get("entities", []),
            *payload.get("keywords", []),
        ]
        if not targets:
            return 0.0
        hits = 0
        total = 0
        for item in targets:
            normalized = str(item).strip().lower()
            if not normalized:
                continue
            total += 1
            if normalized in content_lower or normalized in metadata_text:
                hits += 1
        if total == 0:
            return 0.0
        return hits / total

    @staticmethod
    def _structure_score(payload: dict[str, Any], metadata_text: str) -> float:
        if not metadata_text:
            return 0.0
        targets = [
            payload.get("question", ""),
            *payload.get("entities", []),
            *payload.get("keywords", []),
        ]
        for item in targets:
            normalized = str(item).strip().lower()
            if normalized and normalized in metadata_text:
                return 1.0
        return 0.0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token.lower() for token in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", text)]
