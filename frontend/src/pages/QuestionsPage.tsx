import { useState } from "react";

import { apiClient } from "../api";
import type { EducationAskResponse } from "../api";
import { EmptyState, ErrorNotice, LoadingState, SourceCard } from "../components/CommonUI";
import { useSessionState } from "../hooks/useSessionState";

export function QuestionsPage() {
  const [query, setQuery] = useSessionState("agenthub:questions:query", "");
  const [result, setResult] = useSessionState<EducationAskResponse | null>("agenthub:questions:result", null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useSessionState("agenthub:questions:error", "");

  async function handleAnalyze() {
    const text = query.trim();
    if (!text || loading) {
      return;
    }

    setLoading(true);
    setError("");
    try {
      const response = await apiClient.askEducation({ question: `请解析这道题，并说明答案、考点和常见错因：${text}` });
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "题目解析失败");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="page">
      <div className="page-header">
        <h2>题库管理</h2>
        <p>检索题目、解析、相似题和关联知识点。</p>
      </div>

      <div className="question-search">
        <input
          className="wide-input"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="输入题目、选项、答案或考点..."
        />
        <button disabled={!query.trim() || loading} onClick={() => void handleAnalyze()}>
          {loading ? "解析中" : "解析题目"}
        </button>
      </div>

      <ErrorNotice message={error} />

      <div className="question-layout">
        <div className="result-panel question-result">
          {loading ? (
            <LoadingState text="题目解析中..." />
          ) : !result ? (
            <EmptyState title="暂无解析结果" description="输入题目后开始解析。" />
          ) : (
            <div>
              <h3>解析结果</h3>
              <p>{result.answer}</p>
              <div className="meta-row">
                <strong>Skill</strong>
                <span>{result.skill}</span>
              </div>
              <div className="meta-row">
                <strong>Tools</strong>
                <span>{result.tools_used.join(", ") || "无"}</span>
              </div>
            </div>
          )}
        </div>

        <div className="question-side-panel">
          <h3>关联知识点</h3>
          <p>第一阶段通过题目解析结果间接展示，后续可接入结构化题库和图谱数据。</p>
        </div>
      </div>

      <div className="sources-panel">
        <h3>来源信息</h3>
        {!result || result.sources.length === 0 ? (
          <p>暂无来源信息。</p>
        ) : (
          result.sources.map((source, index) => <SourceCard source={source} index={index} key={index} />)
        )}
      </div>
    </section>
  );
}


