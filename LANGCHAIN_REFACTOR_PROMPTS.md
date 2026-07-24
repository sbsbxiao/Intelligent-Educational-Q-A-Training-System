# LangChain 重构提示词清单

本文档用于指导把当前项目中“智能问答”相关后端实现，从“LangChain 风格但大量自定义实现”的状态，逐步收敛到更标准的 LangChain / LangGraph 体系内。

日期：2026-07-24

## 重构优先级

下面的顺序按“收益高、依赖合理、风险相对可控”排序，建议按顺序推进。

### 1. 结构化输出解析

优先级最高。

原因：

- 这是多个模块的共性基础设施。
- 当前项目里有不少手写 `json.loads`、Markdown 代码块清洗、JSON 截取逻辑。
- 先统一解析规范，后续改“查询改写”“Cypher 生成”“记忆摘要”时会更顺。

### 2. 查询意图识别与查询改写

原因：

- 这部分逻辑边界清晰，容易从 `QAAgent` 中抽出来。
- 与 LangChain 的 `PromptTemplate`、结构化输出、Runnable 很契合。
- 对业务侵入较小，适合作为第一批“链式化”模块。

### 3. 检索器封装

原因：

- 这是从“自定义服务调用”迈向“LangChain retriever 体系”的关键一步。
- 一旦 retriever 标准化，后面的问答链和 LangGraph 节点都会更清晰。
- 但它依赖前面输出结构较稳定，所以排在查询改写之后更合适。

### 4. 答案生成链

原因：

- 它天然适合迁移为 `ChatPromptTemplate + Runnable`。
- 需要依赖前面的输入结构更稳定，尤其是 retriever 输出和 history 注入方式。
- 因此排在检索器之后能减少返工。

### 5. 上下文记忆与历史管理

原因：

- 这是收益很高的一块，但改动也更深。
- 当前系统已经有可用的“最近历史 + 短时记忆 + 长时记忆”机制，设计思路并不差。
- 更适合在主问答链初步标准化之后，再把它适配到 `RunnableWithMessageHistory` 一类机制上。

### 6. QA 主流程编排

原因：

- 这是最终收口步骤。
- 当前项目虽然用了 LangGraph，但 QA 图仍然只有一个黑盒节点。
- 等前面的链、retriever、memory 都更清晰后，再拆成显式节点最稳妥，也最不容易互相打架。

## 推荐执行顺序

建议按这个顺序把提示词逐段发给模型：

1. 统一约束提示词
2. 模块 6：结构化输出解析
3. 模块 1：查询意图识别与查询改写
4. 模块 2：检索器封装
5. 模块 3：答案生成链
6. 模块 4：上下文记忆与历史管理
7. 模块 5：QA 主流程编排

如果你希望更保守一点，也可以在第 4 步之后暂停，先做一次人工 review，再继续记忆和图编排迁移。

## 统一约束提示词

这段建议作为所有模块提示词的前置公共约束，避免不同模块改出来的风格打架。

```text
你正在重构一个 Python 后端项目中的智能问答模块，目标是尽量收敛到 LangChain / LangGraph 体系内部，而不是继续使用项目外部自定义流程。

统一要求如下：

1. 保持现有业务能力不丢失：
   - 仍然支持意图识别
   - 仍然支持查询改写
   - 仍然支持向量检索与图谱检索
   - 仍然支持混合 RAG 问答
   - 仍然支持短时/长时上下文记忆
   - 仍然保留现有 API 输入输出语义，尽量减少上层调用方变更

2. 优先使用 LangChain / LangGraph 原生抽象：
   - PromptTemplate / ChatPromptTemplate
   - Runnable / RunnableSequence / RunnableParallel / RunnableLambda
   - BaseRetriever / 自定义 Retriever
   - RunnableWithMessageHistory
   - LangGraph StateGraph
   - OutputParser / PydanticOutputParser / Structured Output

3. 不要为了迁移而破坏现有 services 层能力。允许复用已有 vector_store、knowledge_graph、local_text_generation 等 service，但要把它们包进 LangChain 标准接口。

4. 代码风格要求：
   - 尽量减少手写消息拼接
   - 尽量减少裸 json.loads + markdown fence 清洗
   - 尽量减少把流程逻辑硬编码在 agent 方法体内
   - 尽量让每一步成为可组合、可测试、可替换的 Runnable 或 Retriever

5. 不直接修改 API 契约，不随意删除已有字段。
6. 优先小步重构，保留兼容层。
```

## 模块 1：查询意图识别与查询改写

当前问题：这部分写在 `QAAgent` 内部，属于手工 prompt + 手工解析，没进入 LangChain chain/runnable 体系。

```text
请重构“查询意图识别 + 查询改写”模块，使其从 QAAgent 内部手写流程迁移到 LangChain 标准链式结构中。

目标：
1. 将“意图识别”和“查询改写”拆成两个独立的 Runnable 子链。
2. 使用 ChatPromptTemplate 定义提示词，而不是在业务类中直接硬编码字符串消息。
3. 使用结构化输出方式返回结果，优先使用 Pydantic 模型或 LangChain 的结构化输出能力，而不是手写 json.loads 和 markdown 清洗。
4. 让这两个子链都可以单独测试、单独调用，也可以被 QA 主链组合调用。
5. 保持现有语义：
   - 意图类别仍然包括 factoid / analytical / comparative / procedural / exploratory
   - 改写结果仍然至少包含 queries、entities、keywords

实现要求：
- 输出应包含：
  1. 适合新增的 schema / Pydantic 数据结构
  2. 适合新增的 chain/runnable 定义
  3. 如何从 QAAgent 中替换原有 _classify_intent 和 _rewrite_query 调用
- 尽量保留当前 LLM 配置来源
- 不要直接改 API 层返回结构

请输出重构后的代码草案，并说明与原始写法相比，哪些地方更符合 LangChain 体系。
```

## 模块 2：检索器封装

当前问题：向量检索、图谱检索、混合重排都写成普通方法，没有进入 `Retriever` / `RunnableParallel` 体系。

```text
请重构智能问答中的检索层，把当前项目中“向量检索 + 图谱检索 + 混合重排”从自定义普通方法迁移到 LangChain 风格的检索器体系中。

目标：
1. 将现有 vector_store.search 封装为一个 LangChain 风格的 Retriever。
2. 将现有 knowledge_graph 查询逻辑封装为一个 LangChain 风格的 Retriever，或至少封装为可组合的 Runnable 检索节点。
3. 提供一个“HybridRetriever”或等价组合层，用 RunnableParallel 或其他合适方式并行执行多路检索，然后统一去重、加权、排序。
4. 保留当前的混合策略：
   - 向量检索结果保留 score
   - 图谱检索结果保留 score
   - 仍有去重和权重加成逻辑
5. 尽量使最终输出兼容 LangChain 下游文档问答链，推荐统一转成 Document 或等价标准结构。

实现要求：
- 不要推翻现有 services 层
- 可以新增适配器，把 services.vector_store 和 services.knowledge_graph 包装起来
- 输出应包括：
  1. 检索器接口设计
  2. 关键适配代码草案
  3. 混合检索组合方式
  4. 如何在 QA 主链中替换原先 _vector_retrieve / _graph_retrieve / _hybrid_rerank

请重点让代码更像 LangChain 生态中的 retriever 设计，而不是继续保留“agent 里塞一堆检索细节”。
```

## 模块 3：答案生成链

当前问题：答案生成还是在 agent 里手工拼接 context 和 history，然后直接调 `ChatOpenAI`。

```text
请重构智能问答中的“答案生成”部分，把当前写在 QAAgent 内部的手工上下文拼接逻辑迁移到 LangChain 的标准问答链结构中。

目标：
1. 使用 ChatPromptTemplate 统一组织 system prompt、history、retrieved context、question。
2. 将答案生成封装为独立 Runnable 或 chain，而不是 QAAgent 的私有方法。
3. 若存在 sources、reasoning_steps、confidence 等附加输出，请设计兼容方式：
   - 回答文本由 LLM 生成
   - sources、confidence、reasoning_steps 可以由链外组合，但不要破坏整体链式结构
4. 避免把大段字符串拼接硬编码在业务方法中。
5. 保持“上下文不足时明确说明”的当前语义。

实现要求：
- 输出应包含：
  1. prompt 设计方式
  2. 输入变量设计
  3. chain/runnable 组合方式
  4. 如何对接上游 retriever 输出
  5. 如何在保留现有返回结构的前提下完成迁移

请优先产出符合 LangChain 风格的实现，而不是仅仅把原函数换个位置。
```

## 模块 4：上下文记忆与历史管理

当前问题：这部分最明显是“项目外自定义 memory”，虽然能用，但没有纳入 LangChain message history / memory 体系。

```text
请重构当前项目中的对话记忆模块，使其尽量收敛到 LangChain 标准的消息历史与记忆体系中，同时保留现有“短时记忆 + 长时记忆 + 最近历史”的业务能力。

背景：
当前项目使用本地 JSON 文件保存：
- 最近对话历史
- 结构化问答记录
- 短时记忆摘要
- 长时记忆摘要

目标：
1. 将“最近历史消息”迁移到 LangChain 的 message history 机制中，优先考虑 RunnableWithMessageHistory 或等价方案。
2. 保留 JSON 文件存储作为底层持久化介质，但把外层接口改造成更符合 LangChain 的 history provider。
3. 短时记忆与长时记忆不要求完全删除自定义逻辑，但要把它们设计为：
   - 可作为 history augmentation 的独立步骤
   - 可在进入主问答链前统一注入
4. 明确区分：
   - 原始消息历史
   - 摘要型短时记忆
   - 摘要型长时记忆
   避免混成一个随意拼接的大字符串
5. 保持 session_id 机制不变。

实现要求：
- 不要简单删除现有 memory 文件读写逻辑
- 应优先保留兼容层
- 输出内容应包括：
  1. 如何定义 message history adapter
  2. 如何把 short memory / long memory 设计成链前增强步骤
  3. 如何把 QA 主链接入 RunnableWithMessageHistory
  4. 如何减少当前 format_history_with_short_memory 这种纯字符串拼接耦合

请输出一种“尽量 LangChain 化，但不激进推翻当前存储设计”的重构方案。
```

## 模块 5：QA 主流程编排

当前问题：虽然已经用了 LangGraph，但 QA 图实际上只有一个 `answer` 节点，真正流程还是埋在 `QAAgent.answer()` 里，没有把步骤显式图化。

```text
请重构当前智能问答主流程，使其更符合 LangGraph 的显式状态流设计，而不是把大部分流程隐藏在 QAAgent.answer() 一个黑盒方法中。

目标：
1. 将 QA 流程拆成多个 LangGraph 节点，至少包括：
   - 加载历史/记忆
   - 意图识别
   - 查询改写
   - 并行检索
   - 检索结果融合/重排
   - 答案生成
   - 保存消息与记忆
2. 保持 graph.py 作为主要编排入口。
3. QAAgent 可以保留，但应弱化为节点能力提供者，而不是整个黑盒 orchestrator。
4. 每个节点尽量调用 LangChain Runnable、Retriever、MessageHistory 等标准组件。
5. 不改变 API 层对 /api/qa/ask 的调用方式和返回结构。

实现要求：
- 给出推荐的 QAState 结构
- 给出节点划分建议
- 给出哪些逻辑应迁入 graph，哪些逻辑仍可保留在 agent/service
- 给出兼容旧实现的小步迁移方案

请重点输出“如何把黑盒 answer() 拆成 LangGraph 显式节点”的方案，并尽量保证与其他模块重构方案兼容。
```

## 模块 6：结构化输出解析

当前问题：很多地方都在手写 JSON fence 清洗和 `json.loads`，这也是典型“在 LangChain 外面做”的部分。

```text
请重构当前项目中所有 LLM 结构化输出解析逻辑，目标是减少手写 JSON 清洗、markdown fence 处理和裸 json.loads，尽量收敛到 LangChain 的结构化输出机制。

适用范围：
- 意图识别输出
- 查询改写输出
- Cypher 生成输出
- 短时记忆摘要输出
- 长时记忆摘要输出
- 其他任何当前依赖“模型输出 JSON 文本再手动解析”的地方

目标：
1. 优先使用 Pydantic schema + LangChain 结构化输出能力。
2. 对于不能完全结构化的场景，也应提供统一解析器，而不是在多个文件重复写清洗逻辑。
3. 保证失败时有合理 fallback，但 fallback 不应继续散落在各个业务方法里。
4. 让 QA、memory、graph retrieval 等模块共用一致的输出解析规范。

实现要求：
- 输出应包含：
  1. 推荐的 schema 组织方式
  2. 推荐的 parser/adapter 设计
  3. 如何替换当前 scattered 的 _clean_json_text 与 json.loads 模式
  4. 如何保持向后兼容

请产出一个可被多个模块复用的统一结构化输出重构方案。
```

## 使用建议

后续你可以这样操作：

1. 先把“统一约束提示词”贴给模型。
2. 再单独贴某一个模块提示词。
3. 让模型输出代码草案后，人工 review 一次。
4. 确认不冲突后，再推进下一个模块。

比较稳的节奏是：

1. 先完成结构化输出解析
2. 再完成 query rewrite / intent
3. 再完成 retriever
4. 再完成 answer chain
5. 最后推进 memory 和 graph orchestration

这样能最大程度减少不同模块改造之间的来回返工。
