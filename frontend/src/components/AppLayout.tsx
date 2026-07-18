import { NavLink, useLocation } from "react-router-dom";
import type { ReactNode } from "react";

import { emptyInsight, InsightContext, type InsightState } from "./InsightContext";
import { RightInfoPanel } from "./RightInfoPanel";
import { useSessionState } from "../hooks/useSessionState";

const navItems = [
  { path: "/chat", label: "智能问答" },
  { path: "/upload", label: "文件上传" },
  { path: "/plan", label: "方案设计" },
  { path: "/knowledge", label: "知识管理" },
  { path: "/questions", label: "题库管理" },
  { path: "/graph", label: "图谱管理" },
  { path: "/tools-skills", label: "工具与 Skill" },
  { path: "/admin", label: "系统管理" }
];

const pageTitles: Record<string, string> = {
  "/chat": "智能问答",
  "/upload": "文件上传",
  "/plan": "方案设计",
  "/knowledge": "知识管理",
  "/questions": "题库管理",
  "/graph": "图谱管理",
  "/tools-skills": "工具与 Skill",
  "/admin": "系统管理"
};

type AppLayoutProps = {
  children: ReactNode;
};

export function AppLayout({ children }: AppLayoutProps) {
  const location = useLocation();
  const title = pageTitles[location.pathname] ?? "教育知识助手";
  const [insight, setInsight] = useSessionState<InsightState>("agenthub:insight", emptyInsight);

  return (
    <InsightContext.Provider value={{ insight, setInsight }}>
      <div className="app-shell">
        <aside className="sidebar">
          <div className="brand">
            <div className="brand-mark">教</div>
            <div>
              <strong>教育知识助手</strong>
              <span>Knowledge Agent</span>
            </div>
          </div>
          <nav className="nav-list">
            {navItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) => `nav-item${isActive ? " active" : ""}`}
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </aside>

        <div className="workspace">
          <header className="topbar">
            <div>
              <h1>{title}</h1>
              <p>蓝白主题教育知识工作台</p>
            </div>
            <div className="status-pill">{insight.status || "后端未连接"}</div>
          </header>

          <div className="content-grid">
            <main className="main-panel">{children}</main>
            <RightInfoPanel
              skill={insight.skill}
              tools_used={insight.toolsUsed}
              sources={insight.sources}
              status={insight.status}
            />
          </div>
        </div>
      </div>
    </InsightContext.Provider>
  );
}





