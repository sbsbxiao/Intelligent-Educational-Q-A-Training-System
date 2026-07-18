import { Navigate, Route, Routes } from "react-router-dom";

import { AppLayout } from "./components/AppLayout";
import { AdminPage } from "./pages/AdminPage";
import { ChatPage } from "./pages/ChatPage";
import { GraphPage } from "./pages/GraphPage";
import { KnowledgePage } from "./pages/KnowledgePage";
import { PlanPage } from "./pages/PlanPage";
import { QuestionsPage } from "./pages/QuestionsPage";
import { ToolsSkillsPage } from "./pages/ToolsSkillsPage";
import { UploadPage } from "./pages/UploadPage";

export default function App() {
  return (
    <AppLayout>
      <Routes>
        <Route path="/" element={<Navigate to="/chat" replace />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/plan" element={<PlanPage />} />
        <Route path="/knowledge" element={<KnowledgePage />} />
        <Route path="/questions" element={<QuestionsPage />} />
        <Route path="/graph" element={<GraphPage />} />
        <Route path="/tools-skills" element={<ToolsSkillsPage />} />
        <Route path="/admin" element={<AdminPage />} />
      </Routes>
    </AppLayout>
  );
}
