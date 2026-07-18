import { useState } from "react";

import { apiClient } from "../api";
import type { EducationAskResponse } from "../api";
import { EmptyState, ErrorNotice, LoadingState } from "../components/CommonUI";
import { useSessionState } from "../hooks/useSessionState";

const planTypes = [
  "学习路径设计",
  "课程大纲设计",
  "备考复习计划",
  "知识点讲解方案",
  "题目解析方案"
];

export function PlanPage() {
  const [planType, setPlanType] = useSessionState("agenthub:plan:type", planTypes[0]);
  const [target, setTarget] = useSessionState("agenthub:plan:target", "");
  const [period, setPeriod] = useSessionState("agenthub:plan:period", "30 天");
  const [studentLevel, setStudentLevel] = useSessionState("agenthub:plan:studentLevel", "");
  const [result, setResult] = useSessionState<EducationAskResponse | null>("agenthub:plan:result", null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useSessionState("agenthub:plan:error", "");

  async function handleGenerate() {
    if (!target.trim() || loading) {
      return;
    }

    setLoading(true);
    setError("");
    try {
      const response = await apiClient.askEducation({ question: buildQuestion() });
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "方案生成失败");
    } finally {
      setLoading(false);
    }
  }

  function buildQuestion() {
    return [
      `请生成${planType}。`,
      `目标：${target.trim()}。`,
      `学习周期：${period}。`,
      studentLevel.trim() ? `学员基础：${studentLevel.trim()}。` : "学员基础：未提供。"
    ].join("");
  }

  return (
    <section className="page">
      <div className="page-header">
        <h2>方案设计</h2>
        <p>生成学习路径、课程大纲、备考计划和讲解方案。</p>
      </div>

      <div className="plan-form">
        <label>
          <span>方案类型</span>
          <select value={planType} onChange={(event) => setPlanType(event.target.value)}>
            {planTypes.map((item) => (
              <option key={item}>{item}</option>
            ))}
          </select>
        </label>
        <label>
          <span>目标输入</span>
          <input value={target} onChange={(event) => setTarget(event.target.value)} placeholder="例如：零基础学习 Python" />
        </label>
        <label>
          <span>学习周期</span>
          <select value={period} onChange={(event) => setPeriod(event.target.value)}>
            <option>7 天</option>
            <option>30 天</option>
            <option>90 天</option>
            <option>自定义周期</option>
          </select>
        </label>
        <label>
          <span>学员基础</span>
          <input value={studentLevel} onChange={(event) => setStudentLevel(event.target.value)} placeholder="例如：零基础 / 有编程经验" />
        </label>
        <button disabled={!target.trim() || loading} onClick={() => void handleGenerate()}>
          {loading ? "生成中" : "生成方案"}
        </button>
      </div>

      <ErrorNotice message={error} />

      <div className="result-panel">
        {loading ? (
          <LoadingState text="方案生成中..." />
        ) : !result ? (
          <EmptyState title="暂无方案结果" description="填写目标后点击生成方案。" />
        ) : (
          <div className="plan-result">
            <h3>生成内容</h3>
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


