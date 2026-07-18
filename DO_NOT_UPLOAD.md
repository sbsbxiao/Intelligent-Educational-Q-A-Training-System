# 不上传到 GitHub 的内容

以下文件或目录包含本地环境、密钥、运行数据、依赖缓存或大模型权重，不应提交到 GitHub。

| 路径 | 原因 | 替代说明 |
| --- | --- | --- |
| `.env` | 包含真实 API Key、数据库密码、私有服务地址等敏感配置 | 提交 `.env.example`，真实值只保留在本地 |
| `model_texttotext/` | 本地文本生成模型权重，体积大且可能受模型许可证约束 | 在 README 中说明下载后放置路径 |
| `model_embedding/` | 本地 Embedding 模型权重，体积大且可能受模型许可证约束 | 在 `.env` 中配置 `LOCAL_EMBEDDING_PATH` |
| `frontend/node_modules/` | 前端依赖缓存，可通过 `npm install` 重新生成 | 提交 `frontend/package.json` 和 `frontend/package-lock.json` |
| `frontend/dist/` | 前端构建产物，可通过 `npm run build` 重新生成 | 需要部署时由 CI 或本地构建生成 |
| `uploads/` | 用户上传的文档，可能包含隐私数据 | 本地运行时自动创建 |
| `runtime/` | 会话历史、运行态缓存等本地数据 | 本地运行时自动创建 |
| `data/`、`logs/` | 数据库文件、日志、调试输出 | 本地运行或容器卷保存 |
| `.agents/`、`.codex/` | 本地 Agent/Codex 工作区元数据 | 与项目源码无关 |

提交前建议执行：

```powershell
git status --short
git check-ignore -v .env model_texttotext model_embedding frontend/node_modules uploads runtime
```
