import { useRef, useState } from "react";
import type { DragEvent } from "react";

import { apiClient } from "../api";
import type { IngestResponse } from "../api";
import { ErrorNotice } from "../components/CommonUI";
import { useSessionState } from "../hooks/useSessionState";

export function UploadPage() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [selectedFileNames, setSelectedFileNames] = useSessionState<string[]>("agenthub:upload:fileNames", []);
  const [results, setResults] = useSessionState<IngestResponse[]>("agenthub:upload:results", []);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useSessionState("agenthub:upload:error", "");
  const [uploadStatus, setUploadStatus] = useSessionState("agenthub:upload:status", "等待上传");

  function handleFiles(files: FileList | null) {
    if (!files) {
      return;
    }
    const nextFiles = Array.from(files);
    setSelectedFiles(nextFiles);
    setSelectedFileNames(nextFiles.map((file) => file.name));
    setUploadStatus("已选择文件");
    setError("");
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    handleFiles(event.dataTransfer.files);
  }

  async function uploadSingle() {
    if (!selectedFiles[0] || uploading) {
      return;
    }

    setUploading(true);
    setUploadStatus("单文件上传处理中");
    setError("");
    try {
      console.info("[upload] single upload started", selectedFiles[0].name);
      const result = await apiClient.uploadDocument(selectedFiles[0]);
      console.info("[upload] single upload finished", result);
      setResults([result]);
      setUploadStatus("单文件上传完成");
    } catch (err) {
      console.error("[upload] single upload failed", err);
      setError(err instanceof Error ? err.message : "上传失败");
      setUploadStatus("单文件上传失败");
    } finally {
      setUploading(false);
    }
  }

  async function uploadBatch() {
    if (!selectedFiles.length || uploading) {
      return;
    }

    setUploading(true);
    setUploadStatus("批量上传处理中");
    setError("");
    try {
      console.info("[upload] batch upload started", selectedFiles.map((file) => file.name));
      const result = await apiClient.uploadBatch(selectedFiles);
      console.info("[upload] batch upload finished", result);
      setResults(result);
      setUploadStatus("批量上传完成");
    } catch (err) {
      console.error("[upload] batch upload failed", err);
      setError(err instanceof Error ? err.message : "批量上传失败");
      setUploadStatus("批量上传失败");
    } finally {
      setUploading(false);
    }
  }

  return (
    <section className="page">
      <div className="page-header">
        <h2>文件上传</h2>
        <p>上传教材、讲义、题库、服务规则等资料。</p>
      </div>

      <div
        className="upload-box"
        onDragOver={(event) => event.preventDefault()}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          className="hidden-input"
          onChange={(event) => handleFiles(event.target.files)}
        />
        <div>
          <strong>拖拽文件到这里，或点击选择文件</strong>
          <p>已选择 {selectedFiles.length || selectedFileNames.length} 个文件</p>
          {selectedFileNames.length > 0 && <p>上次文件：{selectedFileNames.join(", ")}</p>}
        </div>
      </div>

      <div className="upload-actions">
        <button disabled={!selectedFiles.length || uploading} onClick={() => void uploadSingle()}>
          {uploading ? "上传中" : "单文件上传"}
        </button>
        <button disabled={!selectedFiles.length || uploading} onClick={() => void uploadBatch()}>
          {uploading ? "上传中" : "批量上传"}
        </button>
        <span>{uploading ? "正在处理文件..." : uploadStatus}</span>
      </div>

      <ErrorNotice message={error} />

      <div className="upload-result-table">
        <div className="table-row table-head">
          <span>文件名</span>
          <span>Chunks</span>
          <span>实体数</span>
          <span>关系数</span>
          <span>状态</span>
        </div>
        {results.length === 0 ? (
          <div className="empty-row">暂无上传结果</div>
        ) : (
          results.map((item) => (
            <div className="table-row" key={item.file_name}>
              <span>{item.file_name}</span>
              <span>{item.chunks_count}</span>
              <span>{item.entities_count}</span>
              <span>{item.relations_count}</span>
              <span>{item.status}</span>
            </div>
          ))
        )}
      </div>
    </section>
  );
}
