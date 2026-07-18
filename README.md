# 智能教育问答&培训系统

智能教育问答&培训系统是一个面向教学资料、课程知识库和培训场景的 Agent 应用。项目后端基于 FastAPI、LangGraph、RAG、知识图谱和向量检索构建，支持文档上传、知识抽取、教育问答、学习方案设计、题库管理、工具/Skill 管理等能力；前端目前使用 React + Vite + TypeScript 实现交互界面。

## 前端界面

![前端界面](image/前端界面.png)

上图展示了系统前端的主要工作台入口。左侧是功能导航，覆盖智能问答、文件上传、方案设计、知识管理、题库管理、图谱管理、工具与 Skill、系统管理等模块；中间区域承载当前任务的主要操作；右侧用于展示上下文信息、检索来源、Skill 调用结果和后续优化提示。

## 项目结构

```text
.
├── agents/              # 文档解析、知识抽取、问答、教育问答、增量更新 Agent
├── api/                 # FastAPI 后端接口入口
├── config/              # 环境变量配置，真实值从 .env 读取
├── frontend/            # React + Vite + TypeScript 前端项目
├── image/               # README 与项目展示图片
├── orchestrator/        # LangGraph 工作流编排
├── services/            # 向量库、知识图谱、Graph RAG、Embedding、多模态、会话记忆等服务
├── skills/              # 面向教育场景的 Skill 定义和注册
├── tools/               # Tool 基类、注册表和业务工具
├── tests/               # 自动化测试
├── Dockerfile           # 后端 API 镜像构建
├── docker-compose.yml   # 后端依赖服务与 API 编排
├── requirements.txt     # Python 依赖
├── .env.example         # 环境变量模板，不包含真实密钥
└── DO_NOT_UPLOAD.md     # 不应上传 GitHub 的文件/目录说明
```

`model_texttotext/` 和 `model_embedding/` 用于存放本地大模型或 Embedding 模型权重，体积较大且可能受模型许可证约束，因此不会提交到 GitHub。需要本地模型时，请自行下载后在 `.env` 中配置 `LOCAL_TEXT_GENERATION_PATH` 和 `LOCAL_EMBEDDING_PATH`。

## 关键技术栈

后端：Python 3.12、FastAPI、Uvicorn、Pydantic Settings、LangGraph、OpenAI 兼容接口、本地模型推理、RAG、Graph RAG。

数据与检索：ChromaDB、Neo4j、Kafka、可选 PGVector、本地 Embedding 模型。

文档处理：文件上传、文档解析、知识块切分、实体关系抽取、向量入库、知识图谱写入。

前端：React、Vite、TypeScript、React Router。

工程化：Docker、Docker Compose、pytest、`.env` 环境变量管理、`.gitignore` 敏感文件保护。

## 环境变量

先复制模板：

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`，填入你本地真实配置。至少需要关注：

```env
OPENAI_API_KEY=
OPENAI_BASE_URL=
OPENAI_MODEL=
NEO4J_PASSWORD=
LOCAL_EMBEDDING_PATH=
LOCAL_TEXT_GENERATION_PATH=
```

注意：`.env` 不会上传到 GitHub；`.env.example` 只保留占位字段。

## 启动方式一：Docker 启动后端，前端单独启动

适合快速拉起 Neo4j、ChromaDB、Kafka 和后端 API。

```powershell
Copy-Item .env.example .env
docker compose up --build
```

后端服务启动后：

```text
API: http://localhost:8080
Swagger: http://localhost:8080/docs
Neo4j Browser: http://localhost:7474
ChromaDB: http://localhost:8000
```

前端单独启动：

```powershell
cd frontend
npm install
npm run dev
```

默认访问地址：

```text
http://localhost:5173
```

如果后端地址不是 `http://localhost:8080`，可在 `frontend/.env` 中配置：

```env
VITE_API_BASE_URL=http://localhost:8080
```

## 启动方式二：不使用 Docker，全程本地启动

后端依赖 Python 环境，并要求你自己准备 Neo4j、ChromaDB、Kafka 或对应替代服务。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `.env` 后启动后端：

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

再启动前端：

```powershell
cd frontend
npm install
npm run dev
```

## 常用接口

健康检查：

```powershell
curl http://localhost:8080/api/health
```

智能问答：

```powershell
curl -X POST http://localhost:8080/api/qa/ask `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"系统里有哪些课程知识？\"}"
```

教育问答：

```powershell
curl -X POST http://localhost:8080/api/education/ask `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"请帮我讲解 Python 函数的基础知识\"}"
```

## 当前能力

- 文档上传与批量导入
- 文档解析、知识切分和知识抽取
- 向量检索与知识图谱联合问答
- 教育场景问答 Agent
- 学习计划、课程讲解、题目分析等 Skill
- 工具注册与业务工具调用
- 知识图谱统计、增量更新和基础管理接口
- React 前端多页面工作台

## 后续开发方向

方案设计模块还在规划中，后续会基于 React 继续完善课程方案、培训路径、教学目标和阶段性评估的可视化编辑体验。

工具和 Skill 模块会继续扩展：包括更细粒度的工具权限、Skill 编排、工具调用日志、失败重试、可视化调试和可插拔注册机制。

优化方向包括：更稳定的 RAG 检索策略、Graph RAG 查询优化、问答质量评估、文档解析准确率提升、异步任务队列、前后端鉴权、部署配置拆分、CI 自动化测试和更完整的项目文档。

## GitHub 提交注意事项

提交前请确认以下内容不会进入仓库：

- `.env`
- `model_texttotext/`
- `model_embedding/`
- `frontend/node_modules/`
- `uploads/`
- `runtime/`
- `data/`
- `logs/`

详细说明见 [DO_NOT_UPLOAD.md](DO_NOT_UPLOAD.md)。
