import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode } from "react";

import type { SourceItem } from "./InsightContext";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost";
  loading?: boolean;
};

export function Button({ children, className = "", disabled, loading, variant = "primary", ...props }: ButtonProps) {
  return (
    <button className={`ui-button ui-button-${variant} ${className}`.trim()} disabled={disabled || loading} {...props}>
      {loading ? "处理中..." : children}
    </button>
  );
}

type TextInputProps = InputHTMLAttributes<HTMLInputElement> & {
  label?: string;
  error?: string;
};

export function TextInput({ className = "", error, label, ...props }: TextInputProps) {
  return (
    <label className="ui-field">
      {label ? <span>{label}</span> : null}
      <input className={`ui-input ${className}`.trim()} {...props} />
      {error ? <small>{error}</small> : null}
    </label>
  );
}

export function EmptyState({ title = "暂无数据", description }: { title?: string; description?: string }) {
  return (
    <div className="ui-empty-state">
      <strong>{title}</strong>
      {description ? <p>{description}</p> : null}
    </div>
  );
}

export function LoadingState({ text = "加载中..." }: { text?: string }) {
  return (
    <div className="ui-loading-state">
      <span />
      <strong>{text}</strong>
    </div>
  );
}

export function ErrorNotice({ message }: { message: string }) {
  if (!message) {
    return null;
  }
  return <div className="ui-error-notice">{message}</div>;
}

export function ResultCard({ title, children }: { title?: string; children: ReactNode }) {
  return (
    <section className="ui-result-card">
      {title ? <h3>{title}</h3> : null}
      {children}
    </section>
  );
}

export function ToolTag({ name }: { name: string }) {
  return <span className="tool-tag ui-tool-tag">{name}</span>;
}

export function SkillTag({ name }: { name?: string }) {
  return <span className="ui-skill-tag">{name || "待触发"}</span>;
}

function getSourceTitle(source: SourceItem, index: number): string {
  return String(source.source || source.file_name || source.title || `来源 ${index + 1}`);
}

function getSourceMeta(source: SourceItem): string {
  if (source.score !== undefined) {
    return `分数：${String(source.score)}`;
  }
  if (source.type !== undefined) {
    return `类型：${String(source.type)}`;
  }
  return "来源文档";
}

export function SourceCard({ source, index }: { source: SourceItem; index: number }) {
  return (
    <div className="source-card ui-source-card">
      <strong>{getSourceTitle(source, index)}</strong>
      <span>{getSourceMeta(source)}</span>
    </div>
  );
}
