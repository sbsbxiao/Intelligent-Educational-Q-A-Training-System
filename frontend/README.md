# 前端使用说明

本目录是教育知识助手的前端项目，基于 React + Vite + TypeScript。

## 1. 安装依赖

进入前端目录：

```powershell
cd frontend
npm install
```

## 2. 配置后端 API 地址

默认后端地址为：

```text
http://localhost:8080
```

如需修改，在 `frontend` 目录下创建 `.env` 文件：

```env
VITE_API_BASE_URL=http://localhost:8080
```

如果后端端口不是 `8080`，把地址改成实际后端地址即可。

## 3. 启动开发服务

```powershell
npm run dev
```

启动后访问终端提示的地址，通常是：

```text
http://localhost:5173
```

## 4. 页面访问

前端包含以下页面：

| 路径 | 页面 |
| --- | --- |
| `/chat` | 智能问答 |
| `/upload` | 文件上传 |
| `/plan` | 方案设计 |
| `/knowledge` | 知识管理 |
| `/questions` | 题库管理 |
| `/graph` | 图谱管理 |
| `/tools-skills` | 工具与 Skill |
| `/admin` | 系统管理 |

根路径 `/` 会自动跳转到 `/chat`。

## 5. 测试教育问答

确保后端已启动后，打开：

```text
http://localhost:5173/chat
```

选择“教育问答”，输入示例问题：

```text
请帮我讲解 Python 函数的基础知识
```

正常情况下页面会展示：

- 用户问题
- AI 回答
- 当前调用的 Skill
- 使用的 Tool 列表
- 来源文档信息

## 6. 测试文件上传

确保后端已启动后，打开：

```text
http://localhost:5173/upload
```

操作步骤：

1. 拖拽文件到上传区域，或点击选择文件。
2. 点击“单文件上传”或“批量上传”。
3. 查看上传结果列表。

上传结果会展示：

- `file_name`
- `chunks_count`
- `entities_count`
- `relations_count`
- `status`

## 7. 常用后端启动命令

在项目根目录启动后端：

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

后端接口文档：

```text
http://localhost:8080/docs
```
