import { useEffect, useState } from "react";

import { apiClient } from "../api";
import type { EducationAskResponse, StatsResponse } from "../api";
import { EmptyState, ErrorNotice, LoadingState } from "../components/CommonUI";
import { useSessionState } from "../hooks/useSessionState";

type StatBlockProps = {
  title: string;
  data: Record<string, unknown> | null;
};

function StatBlock({ title, data }: StatBlockProps) {
  return (
    <div className="stat-card knowledge-stat">
      <strong>{title}</strong>
      {!data ? (
        <span>暂无数据</span>
      ) : (
        Object.entries(data).map(([key, value]) => (
          <span key={key}>{key}: {String(value)}</span>
        ))
      )}
    </div>
  );
}

export function KnowledgePage() {
  const [stats, setStats] = useSessionState<StatsResponse | null>("agenthub:knowledge:stats", null);
  const [statsLoading, setStatsLoading] = useState(false);
  const [statsError, setStatsError] = useSessionState("agenthub:knowledge:statsError", "");
  const [query, setQuery] = useSessionState("agenthub:knowledge:query", "");
  const [result, setResult] = useSessionState<EducationAskResponse | null>("agenthub:knowledge:result", null);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useSessionState("agenthub:knowledge:searchError", "");

  useEffect(() => {
    if (!stats) {
      void loadStats();
    }
  }, [stats]);

  async function loadStats() {
    setStatsLoading(true);
    setStatsError("");
    try {
      const response = await apiClient.getStats();
      setStats(response);
    } catch (err) {
      setStatsError(err instanceof Error ? err.message : "统计信息加载失败");
    } finally {
      setStatsLoading(false);
    }
  }

  async function handleSearch() {
    const text = query.trim();
    if (!text || searching) {
      return;
    }

    setSearching(true);
    setSearchError("");
    try {
      const response = await apiClient.askEducation({ question: `请检索并总结以下资料相关内容：${text}` });
      setResult(response);
    } catch (err) {
      setSearchError(err instanceof Error ? err.message : "资料检索失败");
    } finally {
      setSearching(false);
    }
  }

  return (
    <section className="page">
      <div className="page-header">
        <h2>知识管理</h2>
        <p>查看知识库统计，并检索资料片段。</p>
      </div>

      <div className="knowledge-actions">
        <button onClick={() => void loadStats()} disabled={statsLoading}>
          {statsLoading ? "刷新中" : "刷新统计"}
        </button>
      </div>

      <ErrorNotice message={statsError} />

      <div className="stat-grid">
        <StatBlock title="向量库信息" data={stats?.vector_store ?? null} />
        <StatBlock title="知识图谱信息" data={stats?.knowledge_graph ?? null} />
      </div>

      <div className="knowledge-search">
        <input
          className="wide-input"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="搜索课程资料、知识点或文档内容..."
        />
        <button disabled={!query.trim() || searching} onClick={() => void handleSearch()}>
          {searching ? "检索中" : "检索资料"}
        </button>
      </div>

      <ErrorNotice message={searchError} />

      <div className="result-panel">
        {searching ? (
          <LoadingState text="资料检索中..." />
        ) : !result ? (
          <EmptyState title="暂无检索结果" description="输入课程资料、知识点或文档内容后开始检索。" />
        ) : (
          <div className="knowledge-result">
            <h3>回答</h3>
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



