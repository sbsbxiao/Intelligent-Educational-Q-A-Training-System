import { useState } from "react";
import type { KeyboardEvent } from "react";

import { apiClient } from "../api";
import type { EducationAskResponse, QaAskResponse } from "../api";
import { ErrorNotice } from "../components/CommonUI";
import { useInsight } from "../components/InsightContext";
import { useSessionState } from "../hooks/useSessionState";

type ChatMode = "education" | "qa";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export function ChatPage() {
  const { setInsight } = useInsight();
  const [mode, setMode] = useSessionState<ChatMode>("agenthub:chat:mode", "education");
  const [input, setInput] = useSessionState("agenthub:chat:input", "");
  const [messages, setMessages] = useSessionState<ChatMessage[]>("agenthub:chat:messages", [
    { role: "assistant", content: "这里将作为类 ChatGPT 的教育问答主界面。" }
  ]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useSessionState("agenthub:chat:error", "");

  async function handleSend() {
    const question = input.trim();
    if (!question || loading) {
      return;
    }

    setInput("");
    setError("");
    setLoading(true);
    setMessages((current) => [...current, { role: "user", content: question }]);
    setInsight({ toolsUsed: [], sources: [], status: "问答中" });

    try {
      if (mode === "education") {
        const result = await apiClient.askEducation({ question });
        appendEducationResult(result);
      } else {
        const result = await apiClient.askQa({ question });
        appendQaResult(result);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "请求失败";
      setError(message);
      setInsight({ toolsUsed: [], sources: [], status: "请求失败" });
    } finally {
      setLoading(false);
    }
  }

  function appendEducationResult(result: EducationAskResponse) {
    setMessages((current) => [...current, { role: "assistant", content: result.answer }]);
    setInsight({
      skill: result.skill,
      toolsUsed: result.tools_used,
      sources: result.sources,
      status: "教育问答完成"
    });
  }

  function appendQaResult(result: QaAskResponse) {
    setMessages((current) => [...current, { role: "assistant", content: result.answer }]);
    setInsight({
      skill: result.intent,
      toolsUsed: [],
      sources: result.sources,
      status: "普通问答完成"
    });
  }

  function handleClear() {
    setMessages([]);
    setError("");
    setInsight({ toolsUsed: [], sources: [], status: "等待提问" });
  }

  async function copyLastAnswer() {
    const lastAnswer = [...messages].reverse().find((message) => message.role === "assistant");
    if (lastAnswer) {
      await navigator.clipboard.writeText(lastAnswer.content);
    }
  }

  function handleKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Enter") {
      void handleSend();
    }
  }

  return (
    <section className="page chat-page">

      <div className="chat-stream">
        {messages.map((message, index) => (
          <div className={`message ${message.role}`} key={index}>
            <span>{message.role === "user" ? "你" : "教育知识助手"}</span>
            <p>{message.content}</p>
          </div>
        ))}
        {loading && (
          <div className="message assistant">
            <span>教育知识助手</span>
            <p>正在生成回答...</p>
          </div>
        )}
        <ErrorNotice message={error} />
      </div>

      <div className="chat-compose">
        <div className="chat-input">
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="输入课程、题目、学习路径或服务规则问题..."
          />
          <button disabled={loading} onClick={() => void handleSend()}>
            {loading ? "发送中" : "发送"}
          </button>
        </div>
        <div className="chat-bottom-actions">
          <div className="mode-switch">
            <button className={mode === "education" ? "active" : ""} onClick={() => setMode("education")}>
              教育问答
            </button>
            <button className={mode === "qa" ? "active" : ""} onClick={() => setMode("qa")}>
              普通问答
            </button>
          </div>
          <div className="chat-actions">
            <button onClick={() => void copyLastAnswer()}>复制回答</button>
            <button onClick={handleClear}>清空对话</button>
          </div>
        </div>
      </div>
    </section>
  );
}



