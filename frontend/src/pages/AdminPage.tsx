import { FormEvent, useEffect, useState } from "react";

import { apiClient } from "../api";
import type { HealthResponse, StatsResponse, UpdateResponse } from "../api";
import { EmptyState, ErrorNotice, LoadingState } from "../components/CommonUI";
import { useSessionState } from "../hooks/useSessionState";

type ChangeType = "created" | "modified" | "deleted";

function formatValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "-";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

export function AdminPage() {
  const [health, setHealth] = useSessionState<HealthResponse | null>("agenthub:admin:health", null);
  const [stats, setStats] = useSessionState<StatsResponse | null>("agenthub:admin:stats", null);
  const [updateResult, setUpdateResult] = useSessionState<UpdateResponse | null>("agenthub:admin:updateResult", null);
  const [filePath, setFilePath] = useSessionState("agenthub:admin:filePath", "");
  const [changeType, setChangeType] = useSessionState<ChangeType>("agenthub:admin:changeType", "modified");
  const [loading, setLoading] = useState(false);
  const [updating, setUpdating] = useState(false);
  const [error, setError] = useSessionState("agenthub:admin:error", "");
  const [updateError, setUpdateError] = useSessionState("agenthub:admin:updateError", "");

  async function loadAdminData() {
    setLoading(true);
    setError("");
    try {
      const [healthData, statsData] = await Promise.all([apiClient.health(), apiClient.getStats()]);
      setHealth(healthData);
      setStats(statsData);
    } catch (err) {
      setError(err instanceof Error ? err.message : "系统状态加载失败");
    } finally {
      setLoading(false);
    }
  }

  async function handleUpdate(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!filePath.trim()) {
      setUpdateError("请输入 file_path");
      return;
    }

    setUpdating(true);
    setUpdateError("");
    setUpdateResult(null);
    try {
      const result = await apiClient.updateKnowledge({
        file_path: filePath.trim(),
        change_type: changeType
      });
      setUpdateResult(result);
    } catch (err) {
      setUpdateError(err instanceof Error ? err.message : "手动更新失败");
    } finally {
      setUpdating(false);
    }
  }

  useEffect(() => {
    if (!health && !stats) {
      void loadAdminData();
    }
  }, [health, stats]);

  return (
    <section className="page">
      <div className="page-header">
        <h2>系统管理</h2>
        <p>查看服务健康状态、知识库统计，并触发指定文件的手动更新。</p>
      </div>

      <div className="admin-actions">
        <button type="button" onClick={loadAdminData} disabled={loading}>
          {loading ? "刷新中..." : "刷新状态"}
        </button>
      </div>

      <ErrorNotice message={error} />

      <div className="stat-grid">
        <div className="stat-card admin-stat-card">
          <span>服务健康状态</span>
          <strong>{health?.status || "-"}</strong>
          <small>{health?.service || "-"}</small>
        </div>
        <div className="stat-card admin-stat-card">
          <span>向量库统计</span>
          <strong>{stats ? Object.keys(stats.vector_store || {}).length : 0} 项</strong>
          <small>vector_store</small>
        </div>
        <div className="stat-card admin-stat-card">
          <span>知识图谱统计</span>
          <strong>{stats ? Object.keys(stats.knowledge_graph || {}).length : 0} 项</strong>
          <small>knowledge_graph</small>
        </div>
      </div>

      <div className="admin-detail-grid">
        <div className="result-panel admin-detail-panel">
          <h3>向量库信息</h3>
          {loading && !stats ? (
            <LoadingState text="向量库统计加载中..." />
          ) : stats ? (
            Object.entries(stats.vector_store || {}).map(([key, value]) => (
              <div className="meta-row" key={key}>
                <strong>{key}</strong>
                <span>{formatValue(value)}</span>
              </div>
            ))
          ) : (
            <EmptyState title="暂无统计数据" />
          )}
        </div>

        <div className="result-panel admin-detail-panel">
          <h3>知识图谱信息</h3>
          {loading && !stats ? (
            <LoadingState text="知识图谱统计加载中..." />
          ) : stats ? (
            Object.entries(stats.knowledge_graph || {}).map(([key, value]) => (
              <div className="meta-row" key={key}>
                <strong>{key}</strong>
                <span>{formatValue(value)}</span>
              </div>
            ))
          ) : (
            <EmptyState title="暂无统计数据" />
          )}
        </div>
      </div>

      <form className="admin-update-form" onSubmit={handleUpdate}>
        <h3>手动更新</h3>
        <div className="form-grid">
          <input
            value={filePath}
            onChange={(event) => setFilePath(event.target.value)}
            placeholder="file_path"
          />
          <select value={changeType} onChange={(event) => setChangeType(event.target.value as ChangeType)}>
            <option value="created">created</option>
            <option value="modified">modified</option>
            <option value="deleted">deleted</option>
          </select>
          <button type="submit" disabled={updating}>
            {updating ? "更新中..." : "触发更新"}
          </button>
        </div>
        <ErrorNotice message={updateError} />
      </form>

      <div className="result-panel admin-update-result">
        <h3>更新结果</h3>
        {updating ? (
          <LoadingState text="手动更新处理中..." />
        ) : updateResult ? (
          <>
            <div className="meta-row">
              <strong>vectors_added</strong>
              <span>{updateResult.vectors_added}</span>
            </div>
            <div className="meta-row">
              <strong>vectors_deleted</strong>
              <span>{updateResult.vectors_deleted}</span>
            </div>
            <div className="meta-row">
              <strong>entities_added</strong>
              <span>{updateResult.entities_added}</span>
            </div>
            <div className="meta-row">
              <strong>relations_added</strong>
              <span>{updateResult.relations_added}</span>
            </div>
            <div className="meta-row">
              <strong>success</strong>
              <span>{String(updateResult.success)}</span>
            </div>
            <div className="meta-row">
              <strong>processing_time_ms</strong>
              <span>{updateResult.processing_time_ms}</span>
            </div>
          </>
        ) : (
          <EmptyState title="暂无更新结果" description="输入文件路径和变更类型后触发更新。" />
        )}
      </div>
    </section>
  );
}



