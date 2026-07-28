"""
GraphRAG 娣峰悎妫€绱㈢閬?鈥?鍚戦噺妫€绱?+ 鍥捐氨閬嶅巻 + 閲嶆帓搴?
杩欐槸鏈」鐩殑鏍稿績鎶€鏈寒鐐逛箣涓€锛?  浼犵粺 RAG 鍙仛鍚戦噺妫€绱紝涓㈠け瀹炰綋闂寸殑缁撴瀯鍖栧叧绯?  GraphRAG 灏嗙煡璇嗗浘璋卞拰鍚戦噺妫€绱㈣瀺鍚堬紝瀹炵幇澶氳烦鎺ㄧ悊

宸ヤ綔娴?
  Query 鈫?[鍚戦噺妫€绱㈠垎鏀痌 鈹€鈹€鈹€鈹€鈫?鍚堝苟 鈫?浜ゅ弶閲嶆帓搴?鈫?Top-K
         [鍥捐氨妫€绱㈠垎鏀痌 鈹€鈹€鈹€鈹€鈫?
鍥捐氨妫€绱㈢瓥鐣?
  1. 瀹炰綋閾炬帴: 浠?query 涓瘑鍒疄浣?鈫?鍦ㄥ浘璋变腑瀹氫綅
  2. 瀛愬浘鍙洖: 浠庡畾浣嶅疄浣撳嚭鍙?N 璺抽亶鍘?  3. 璺緞鎺ㄧ悊: 鎵惧埌瀹炰綋闂寸殑鏈€鐭矾寰勶紝鎻愪緵鎺ㄧ悊閾?"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings
from services.knowledge_graph import KnowledgeGraphService
from services.token_usage import token_usage_service
from services.vector_store import VectorStoreService


@dataclass
class GraphRAGContext:
    content: str
    source_type: str  # "vector" | "subgraph" | "path" | "community"
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


ENTITY_LINKING_PROMPT = """\
浠庝互涓嬮棶棰樹腑鎻愬彇鎵€鏈夊彲鑳界殑瀹炰綋鍚嶇О锛堜汉鍚嶃€佺粍缁囥€佹妧鏈€佷骇鍝併€佹蹇电瓑锛夈€?杩斿洖 JSON: {"entities": ["瀹炰綋1", "瀹炰綋2"]}
鍙繑鍥?JSON銆?"""

COMMUNITY_SUMMARY_PROMPT = """\
浣犳槸涓€涓煡璇嗗浘璋卞垎鏋愪笓瀹躲€傛牴鎹互涓嬪瓙鍥句俊鎭紝鐢熸垚涓€娈电粨鏋勫寲鎽樿銆?瑕佹眰锛?1. 姒傝堪瀛愬浘涓殑鏍稿績瀹炰綋鍜屽叧绯?2. 绐佸嚭瀹炰綋闂寸殑鍏抽敭鑱旂郴
3. 鎸囧嚭浠讳綍鏈変环鍊肩殑鎺ㄧ悊閾?"""


class GraphRAGPipeline:
    """
    GraphRAG 娣峰悎妫€绱㈢閬?
    铻嶅悎涓夌妫€绱㈢瓥鐣?
      1. 鍚戦噺璇箟妫€绱?鈥?鎹曡幏璇箟鐩镐技鍐呭
      2. 鍥捐氨瀛愬浘妫€绱?鈥?閫氳繃瀹炰綋鍏崇郴杩涜缁撴瀯鍖栨帹鐞?      3. 绀惧尯鎽樿妫€绱?鈥?瀵瑰瓙鍥捐繘琛屾憳瑕侊紝鎻愪緵楂樺眰姒傝
    """

    def __init__(
        self,
        vector_store: VectorStoreService,
        knowledge_graph: KnowledgeGraphService,
    ) -> None:
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
        )

    async def retrieve(self, query: str, top_k: int = 10) -> list[GraphRAGContext]:
        """
        娣峰悎妫€绱㈠叆鍙?        骞惰鎵ц鍚戦噺妫€绱㈠拰鍥捐氨妫€绱紝鐒跺悗浜ゅ弶閲嶆帓搴?        """
        vector_results = await self._vector_search(query, top_k=top_k)
        entities = await self._entity_linking(query)
        subgraph_results = await self._subgraph_search(entities)
        path_results = await self._path_search(entities)

        all_results = vector_results + subgraph_results + path_results

        if subgraph_results:
            community_ctx = await self._community_summary(subgraph_results)
            all_results.append(community_ctx)

        reranked = self._cross_rerank(all_results, query)
        return reranked[:top_k]

    # 鈹€鈹€ Step 1: 鍚戦噺妫€绱?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    async def _vector_search(self, query: str, top_k: int = 5) -> list[GraphRAGContext]:
        results = await self.vector_store.search(query, top_k=top_k)
        return [
            GraphRAGContext(
                content=doc["content"],
                source_type="vector",
                score=score,
                metadata=doc.get("metadata", {}),
            )
            for doc, score in results
        ]

    # 鈹€鈹€ Step 2: 瀹炰綋閾炬帴 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    async def _entity_linking(self, query: str) -> list[str]:
        messages = [
            SystemMessage(content=ENTITY_LINKING_PROMPT),
            HumanMessage(content=query),
        ]
        resp = await self.llm.ainvoke(messages)
        try:
            cleaned = resp.content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(cleaned)
            return data.get("entities", [])
        except (json.JSONDecodeError, IndexError):
            return []

    # 鈹€鈹€ Step 3: 瀛愬浘妫€绱?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    async def _subgraph_search(self, entities: list[str], hops: int = 2) -> list[GraphRAGContext]:
        contexts: list[GraphRAGContext] = []
        for entity_name in entities:
            neighbors = await self.knowledge_graph.get_neighbors(entity_name, hops=hops)
            for record in neighbors:
                content = (
                    f"{record.get('source', '')} "
                    f"--[{', '.join(record.get('relations', []))}]--> "
                    f"{record.get('target', '')} "
                    f"({record.get('target_type', '')}): "
                    f"{record.get('target_desc', '')}"
                )
                contexts.append(GraphRAGContext(
                    content=content,
                    source_type="subgraph",
                    score=0.75,
                    metadata={"entity": entity_name, "hops": hops},
                ))
        return contexts

    # 鈹€鈹€ Step 4: 璺緞妫€绱?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    async def _path_search(self, entities: list[str]) -> list[GraphRAGContext]:
        """鏌ユ壘瀹炰綋瀵逛箣闂寸殑鏈€鐭矾寰勶紝鎻愪緵鎺ㄧ悊閾?""
        if len(entities) < 2:
            return []

        contexts: list[GraphRAGContext] = []
        for i in range(len(entities)):
            for j in range(i + 1, min(i + 3, len(entities))):
                cypher = """
                MATCH path = shortestPath(
                    (a:Entity {name: $name_a})-[*..5]-(b:Entity {name: $name_b})
                )
                RETURN
                    [n IN nodes(path) | n.name] AS node_names,
                    [r IN relationships(path) | type(r)] AS rel_types
                LIMIT 3
                """
                try:
                    records = await self.knowledge_graph.execute_cypher(
                        cypher, {"name_a": entities[i], "name_b": entities[j]}
                    )
                    for rec in records:
                        nodes = rec.get("node_names", [])
                        rels = rec.get("rel_types", [])
                        path_str = ""
                        for k, node in enumerate(nodes):
                            path_str += node
                            if k < len(rels):
                                path_str += f" --[{rels[k]}]--> "
                        contexts.append(GraphRAGContext(
                            content=f"鎺ㄧ悊璺緞: {path_str}",
                            source_type="path",
                            score=0.85,
                            metadata={"from": entities[i], "to": entities[j]},
                        ))
                except Exception:
                    continue
        return contexts

    # 鈹€鈹€ Step 5: 绀惧尯鎽樿 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    async def _community_summary(self, subgraph_results: list[GraphRAGContext]) -> GraphRAGContext:
        """瀵规绱㈠埌鐨勫瓙鍥句俊鎭繘琛屾憳瑕?""
        subgraph_text = "\n".join(r.content for r in subgraph_results[:20])
        messages = [
            SystemMessage(content=COMMUNITY_SUMMARY_PROMPT),
            HumanMessage(content=f"瀛愬浘淇℃伅:\n{subgraph_text}"),
        ]
        resp = await self.llm.ainvoke(messages)
        return GraphRAGContext(
            content=resp.content,
            source_type="community",
            score=0.9,
            metadata={"type": "community_summary"},
        )

    # 鈹€鈹€ Step 6: 浜ゅ弶閲嶆帓搴?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    @staticmethod
    def _cross_rerank(contexts: list[GraphRAGContext], query: str) -> list[GraphRAGContext]:
        """
        浜ゅ弶閲嶆帓搴忕瓥鐣?
          - 鍚戦噺妫€绱? 鍩虹鍒?脳 1.0
          - 瀛愬浘妫€绱? 鍩虹鍒?脳 1.15 (缁撴瀯鍖栦俊鎭洿绮惧噯)
          - 璺緞妫€绱? 鍩虹鍒?脳 1.25 (鎺ㄧ悊閾炬渶鏈変环鍊?
          - 绀惧尯鎽樿: 鍩虹鍒?脳 1.1  (楂樺眰姒傝)
        """
        weight_map = {"vector": 1.0, "subgraph": 1.15, "path": 1.25, "community": 1.1}
        for ctx in contexts:
            ctx.score *= weight_map.get(ctx.source_type, 1.0)

        seen: set[str] = set()
        unique: list[GraphRAGContext] = []
        for ctx in contexts:
            key = ctx.content[:80]
            if key not in seen:
                seen.add(key)
                unique.append(ctx)

        unique.sort(key=lambda c: c.score, reverse=True)
        return unique

