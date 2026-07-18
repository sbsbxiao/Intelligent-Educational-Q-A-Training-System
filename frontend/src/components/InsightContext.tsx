import { createContext, useContext } from "react";

export type SourceItem = Record<string, unknown>;

export type InsightState = {
  skill?: string;
  toolsUsed: string[];
  sources: SourceItem[];
  status?: string;
};

type InsightContextValue = {
  insight: InsightState;
  setInsight: (insight: InsightState) => void;
};

export const emptyInsight: InsightState = {
  toolsUsed: [],
  sources: [],
  status: "等待提问"
};

export const InsightContext = createContext<InsightContextValue | null>(null);

export function useInsight() {
  const context = useContext(InsightContext);
  if (!context) {
    throw new Error("useInsight must be used inside InsightContext.Provider");
  }
  return context;
}

export function SourceSummary({ sources }: { sources: SourceItem[] }) {
  if (!sources.length) {
    return <p>暂无来源信息。</p>;
  }

  return (
    <div className="source-stack">
      {sources.slice(0, 5).map((source, index) => (
        <div className="source-card" key={index}>
          <strong>{String(source.source || `来源 ${index + 1}`)}</strong>
          {source.score !== undefined && <span>分数：{String(source.score)}</span>}
        </div>
      ))}
    </div>
  );
}

export function ToolTags({ tools }: { tools: string[] }) {
  if (!tools.length) {
    return <strong>暂无调用</strong>;
  }

  return (
    <div className="tag-list">
      {tools.map((tool) => (
        <span className="tool-tag" key={tool}>{tool}</span>
      ))}
    </div>
  );
}

export function SkillValue({ skill }: { skill?: string }) {
  return <strong>{skill || "待触发"}</strong>;
}
