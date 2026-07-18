import { useState } from "react";

import { apiClient } from "../api";
import type { EducationAskResponse } from "../api";
import { useSessionState } from "../hooks/useSessionState";

const demoNodes = ["课程", "章节", "知识点", "题目", "错因"];
const demoEdges = ["课程 -> 章节", "章节 -> 知识点", "知识点 -> 题目", "题目 -> 错因"];

export function GraphPage() {
  const [entity, setEntity] = useSessionState("agenthub:graph:entity", "");
  const [result, setResult] = useSessionState<EducationAskResponse | null>("agenthub:graph:result", null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useSessionState("agenthub:graph:error", "");

  async function handleSearch() {
    const text = entity.trim();
    if (!text || loading) {
      return;
    }

    setLoading(true);
    setError("");
    try {
      const response = await apiClient.askEducation({ question: `请查询并总结与实体「${text}」相关的课程、知识点、题目和关系。` });
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "图谱查询失败");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="page">
      <div className="page-header">
        <h2>图谱管理</h2>
        <p>查看课程、章节、知识点、题目之间的关系。</p>
      </div>

      <div className="graph-search">
        <input
          className="wide-input"
          value={entity}
          onChange={(event) => setEntity(event.target.value)}
          placeholder="搜索实体名称，例如：函数、Python 基础、错题分析..."
        />
        <button disabled={!entity.trim() || loading} onClick={() => void handleSearch()}>
          {loading ? "查询中" : "查询关系"}
        </button>
      </div>

      {error && <div className="chat-error">{error}</div>}

      <div className="graph-layout">
        <div className="graph-placeholder graph-demo">
          <div className="node-row">
            {demoNodes.map((node) => (
              <span className="demo-node" key={node}>{node}</span>
            ))}
          </div>
          <div className="edge-row">
            {demoEdges.map((edge) => (
              <span className="demo-edge" key={edge}>{edge}</span>
            ))}
          </div>
        </div>

        <div className="graph-side">
          <div className="graph-detail-card">
            <h3>节点详情</h3>
            <p>{entity.trim() || "请选择或搜索实体。"}</p>
          </div>
          <div className="graph-detail-card">
            <h3>关系详情</h3>
            <p>当前为静态示例关系，后续可接入图谱查询 API。</p>
          </div>
        </div>
      </div>

      <div className="result-panel graph-result">
        {!result ? (
          "实体关系查询结果占位"
        ) : (
          <div>
            <h3>查询结果</h3>
            <p>{result.answer}</p>
            <div className="meta-row">
              <strong>Skill</strong>
              <span>{result.skill}</span>
            </div>
            <div className="meta-row">
              <strong>Tools</strong>
              <span>{result.tools_used.join(", ") || "无"}</span>
            </div>
            <div className="meta-row">
              <strong>Sources</strong>
              <span>{result.sources.length} 条</span>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}



