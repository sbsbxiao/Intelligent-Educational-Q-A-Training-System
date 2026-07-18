const skills = [
  {
    name: "course_explanation",
    description: "课程讲解、章节总结、知识点解释。",
    tools: ["course_material_search", "knowledge_graph_query"]
  },
  {
    name: "question_analysis",
    description: "题目解析、答案解释、错题分析、考点分析。",
    tools: ["question_bank_search", "course_material_search", "knowledge_graph_query"]
  },
  {
    name: "study_plan",
    description: "学习路径、复习计划、备考顺序、零基础学习建议。",
    tools: ["course_material_search", "knowledge_graph_query"]
  },
  {
    name: "service_qa",
    description: "请假、作业、报名、证书、课时等学员服务问答。",
    tools: ["student_service_policy_search"]
  }
];

const tools = [
  {
    name: "course_material_search",
    description: "检索课程资料、讲义、课程大纲和教学文档。"
  },
  {
    name: "question_bank_search",
    description: "检索题库、相似题、答案解析和错题说明。"
  },
  {
    name: "student_service_policy_search",
    description: "检索学员服务规则、作业要求、报名规则和证书政策。"
  },
  {
    name: "knowledge_graph_query",
    description: "查询知识图谱中的实体邻居、知识点关系和结构化路径。"
  }
];

export function ToolsSkillsPage() {
  return (
    <section className="page">
      <div className="page-header">
        <h2>工具与 Skill</h2>
        <p>展示教育 Agent 的能力结构和调用关系。</p>
      </div>

      <div className="tools-skills-grid">
        <div className="capability-section">
          <h3>Skills</h3>
          <div className="capability-list">
            {skills.map((skill) => (
              <div className="capability-card" key={skill.name}>
                <strong>{skill.name}</strong>
                <p>{skill.description}</p>
                <div className="relation-tools">
                  {skill.tools.map((tool) => (
                    <span key={tool}>{tool}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="capability-section">
          <h3>Tools</h3>
          <div className="capability-list">
            {tools.map((tool) => (
              <div className="capability-card" key={tool.name}>
                <strong>{tool.name}</strong>
                <p>{tool.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="relation-section">
        <h3>Skill 与 Tool 调用关系</h3>
        {skills.map((skill) => (
          <div className="relation-row" key={skill.name}>
            <strong>{skill.name}</strong>
            <span>调用</span>
            <div className="relation-tools">
              {skill.tools.map((tool) => (
                <span key={tool}>{tool}</span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
