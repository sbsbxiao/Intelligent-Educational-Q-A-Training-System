import { useState } from "react";

import { SkillTag, SourceCard, ToolTag } from "./CommonUI";
import type { SourceItem } from "./InsightContext";

export type RightInfoPanelProps = {
  skill?: string;
  tools_used?: string[];
  toolsUsed?: string[];
  sources?: SourceItem[];
  status?: string;
  systemTip?: string;
};

export function RightInfoPanel({
  skill,
  tools_used,
  toolsUsed,
  sources = [],
  status = "等待操作",
  systemTip = "当前页面会在调用后显示 Skill、Tool 和来源信息。"
}: RightInfoPanelProps) {
  const [expanded, setExpanded] = useState(false);
  const tools = tools_used ?? toolsUsed ?? [];

  return (
    <aside className={`insight-panel right-info-panel ${expanded ? "expanded" : ""}`}>
      <div className="right-info-header">
        <div>
          <h2>调用信息</h2>
          <span>{status}</span>
        </div>
        <button className="right-info-toggle" type="button" onClick={() => setExpanded((current) => !current)}>
          {expanded ? "收起" : "展开"}
        </button>
      </div>

      <div className="right-info-body">
        <section className="right-info-block">
          <span>当前 Skill</span>
          <SkillTag name={skill} />
        </section>

        <section className="right-info-block">
          <span>Tool 列表</span>
          {tools.length ? (
            <div className="tag-list">
              {tools.map((tool) => (
                <ToolTag name={tool} key={tool} />
              ))}
            </div>
          ) : (
            <strong>暂无调用</strong>
          )}
        </section>

        <section className="right-info-block source-list">
          <span>来源文档</span>
          {sources.length ? (
            <div className="source-stack">
              {sources.slice(0, 5).map((source, index) => (
                <SourceCard source={source} index={index} key={index} />
              ))}
            </div>
          ) : (
            <p>暂无来源信息。</p>
          )}
        </section>

        <section className="right-info-block">
          <span>系统提示</span>
          <p>{systemTip}</p>
        </section>
      </div>
    </aside>
  );
}
