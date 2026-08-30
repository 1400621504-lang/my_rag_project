# 本地 RAG 知识库问答系统

基于 LangChain 1.x + LangGraph + Ollama 的本地全栈 RAG：向量召回 + BM25 关键词召回 + RRF 融合 + bge-reranker 精排 + 父子文档展开，另有 Agentic RAG（模型自主决定检索几次、怎么改写检索词）和一层 MCP server。零 API 费用，数据不出本机。

**检索链路**

```
                    ┌─ 向量召回（bge-m3 + ChromaDB，similarity/MMR）──┐
用户问题 ─┤                                                        ├─ RRF 融合 ─ bge-reranker 精排 ─ 子块→父块展开 ─ 拼上下文 ─ LLM 流式生成
                    └─ 关键词召回（jieba 分词 + BM25Okapi）──────────┘
                                                        每一段都可独立开关，并有消融数据（见下文）
```

## 实测指标

检索层消融评测，28 条人工标注 gold 的问题，统一召回 20 → 取 top 5：

| 配置 | Hit@1 | Hit@3 | Hit@5 | MRR@5 | nDCG@5 | 平均延迟 |
|---|---|---|---|---|---|---|
| 仅向量 | 0.821 | 0.929 | 0.929 | 0.869 | 0.825 | 254 ms |
| 向量 + 精排 | 0.786 | 1.000 | 1.000 | 0.881 | 0.885 | 1645 ms |
| 双路 + RRF | 0.821 | 0.964 | 1.000 | **0.890** | 0.884 | **201 ms** |
| 双路 + RRF + 精排 | 0.786 | 1.000 | 1.000 | 0.881 | 0.885 | 1111 ms |

### 端到端：固定管线 vs Agentic RAG

12 题（4 单跳对照 + 5 多跳 + 3 改写/跨语种），逐题独立，人工 `must_contain` 关键词全覆盖判分：

| 引擎 | 全对率 | 关键词覆盖 | 平均延迟 | single | multi | rewrite |
|---|---|---|---|---|---|---|
| 固定管线 | 0.417 | 0.575 | 8.1 s | 0.75 | 0.20 | 0.333 |
| Agent | **0.667** | **0.778** | 33.0 s | 0.75 | **0.40** | **1.000** |

Agent 的收益全部集中在"改写"和"多跳"两类，单跳持平却多付 4 倍延迟 —— 所以它不是默认引擎，
前端和 API 都保留两条路径，按问题类型选。另一个必须记的教训：早期允许"有对话历史时不强制查库"，
Agent 全对率当场掉到 0.167（同期固定管线 0.667），83% 的题一次检索都没发生，
改成"本轮没有检索证据就回退固定管线"才修回来。

三个不太舒服但有价值的发现，细节见 [backend/eval/RESULTS.md](backend/eval/RESULTS.md)：

- **精排不是无条件增益**：字面重合高的查询 Hit@1 从 0.80 → 0.90，但需要语义理解的查询反而 0.75 → 0.67。它真正买到的是"前五必有一条对的"，代价是首条命中率和 1.4 秒延迟。
- **MMR 在当前语料规模下完全无效**（`fetch_k=20 ≥ 语料 14`，没有可 diversify 的余量），加与不加各项指标一模一样。
- **语料太小导致召回饱和**，所以"双路 + 精排"目前测不出比"向量 + 精排"更多的东西 —— 混合检索最核心的补召回价值还没被验证，这是下一步要扩语料的原因，而不是继续调参的理由。

## 功能清单

| 模块 | 能力 | 版本 |
|---|---|---|
| 解析 | PDF（PyMuPDF 文字层）/ DOCX / TXT / Markdown / 图片，扫描页自动回退 OCR | v1.0 + |
| 切分 | recursive 递归切分；parent_child 父子切分（子块检索、父块喂模型） | v1.0 / v1.2 |
| 召回 | ChromaDB 向量检索，similarity / MMR 可切换 | v1.0 |
| 双路 | jieba 分词 BM25 关键词召回，与向量路做 RRF 融合 | v1.1 |
| 精排 | bge-reranker-base cross-encoder，召回 20 → 精排取 5 | v1.1 |
| 生成 | Ollama / OpenAI 兼容接口，流式输出，多轮对话记忆 | v1.0 |
| 服务 | FastAPI 异步 12 个接口，Streamlit 深色前端，参数实时可调 | v1.0 |
| 评测 | 检索层 Hit@K / MRR / nDCG 消融（28 题）+ 端到端问答对比（12 题） | v1.2+ |
| Agent | LangGraph 工具循环：`search_knowledge_base` / `list_knowledge_base`，多次查询改写，未查库自动回退固定管线 | v1.3 |
| MCP | 7 个工具 + 1 个 resource，把知识库接到 Claude Desktop / Codex 等 MCP 客户端 | v1.3 |

## 项目结构

```
my_rag_project/
├── backend/
│   ├── api/
│   │   ├── rag_api.py                  332 行  # FastAPI 12 个接口：/chat /agent/chat /upload /documents /memory /health /stats
│   │   └── mcp_server.py               211 行  # MCP server：检索/问答/多跳问答/入库/状态，stdio 起
│   ├── chains/
│   │   ├── rag_chain.py                360 行  # 固定管线：召回→融合→精排→父块展开→生成 + 检索缓存
│   │   └── agent_chain.py              377 行  # Agentic RAG：工具循环、证据编号、轨迹、兜底回退
│   ├── models/embedding_factory.py     102 行  # Embedding 工厂（ollama / local_hf / api）
│   ├── retriever/fusion.py              63 行  # RRF 融合
│   ├── services/
│   │   ├── vectorstore_service.py      388 行  # ChromaDB 封装：增删查、MMR、统计
│   │   ├── document_processor.py       311 行  # 解析 + 两套切分策略 + 扫描页 OCR 回退
│   │   ├── rerank_service.py           138 行  # bge-reranker 懒加载单例
│   │   ├── bm25_service.py             121 行  # BM25 内存索引，语料变化自动重建
│   │   └── ocr_service.py              131 行  # RapidOCR（ONNX 离线），不可用时自动降级
│   └── eval/                                   # 两级评测，全部结果见 RESULTS.md
│       ├── qa_dev.yaml                 152 行  # 28 条检索 gold 标注
│       ├── agent_dev.yaml               74 行  # 12 条端到端问答（单跳对照 / 多跳 / 改写）
│       ├── run_eval.py                 229 行  # 检索层 5 配置消融
│       ├── run_agent_eval.py           174 行  # 固定管线 vs Agent 对比
│       └── metrics.py                   44 行  # Hit@K / MRR / nDCG
├── config/config.yaml                            # 模型 / 切分 / 检索 / Agent / 对话 / 缓存 / OCR
├── frontend/app.py                     714 行  # Streamlit 界面，可切 RAG / Agent 引擎并展示检索轨迹
└── requirements.txt                            # 46 行，fresh clone 可直接 pip install
```

## 快速开始

```bash
# 1. 准备模型
ollama pull qwen2.5:7b     # 生成
ollama pull bge-m3         # 向量化（1024 维，多语言）

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载精排模型（可选，不装则把 config.yaml 里 rerank.enabled 设为 false）
#    放到 ./models/rerank/bge-reranker-base/

# 4. 启动
streamlit run frontend/app.py                        # 界面，http://localhost:8501
uvicorn backend.api.rag_api:app --reload --port 8000 # 或走 API
```

接入 MCP 客户端（Claude Desktop / Codex 等的配置片段）：

```json
{
  "mcpServers": {
    "local-rag-kb": {
      "command": "/opt/anaconda3/bin/python",
      "args": ["-m", "backend.api.mcp_server"],
      "cwd": "/Users/apple/Desktop/AI_Learning/my_rag_project"
    }
  }
}
```

客户端里会看到 7 个工具：`search_knowledge_base`（只检索不生成）、`ask`（固定管线问答）、
`agent_ask`（多跳）、`list_documents`、`add_file`、`add_text`、`kb_stats`。
模型实例是懒加载的，握手阶段不会为了载精排模型而超时。

## 配置说明

`config/config.yaml` 全量可调：

| 段落 | 关键项 | 默认 |
|---|---|---|
| `embedding` | `type`（ollama / local_hf / api）、模型名 | ollama + bge-m3 |
| `rerank` | `enabled`、`candidate_k`（精排前召回数） | true / 20 |
| `llm` | `type`（local / api）、模型、温度、`num_ctx`（上下文窗口） | local + qwen2.5:7b / 8192 |
| `chunking` | `strategy`（recursive / parent_child）、父子块大小与重叠 | recursive，800/200，子块 200/50 |
| `retrieval` | `search_type`（mmr / similarity）、`k`、`fetch_k`、`lambda_mult` | mmr / 5 / 20 / 0.7 |
| `retrieval.max_context_chars` | 进入提示词的资料字符预算，按名次从后往前丢 | 6000（0=不裁剪） |
| `retrieval.hybrid` | `enabled`、`bm25_top_k`、`rrf_k` | **true** / 20 / 60（改为默认开，依据见消融表） |
| `agent` | `max_iterations`（LangGraph 步数上限）、`max_search_calls`、`temperature` | 8 / 4 / 0.1 |
| `cache` | 检索结果缓存 `enabled`、`ttl`（秒）、`max_size` | true / 60 / 200 |
| `ocr` | `enabled`、`pdf_dpi`、`min_page_chars`（页面可提取文字低于此值判为扫描件） | true / 200 / 20 |
| `conversation` | `max_history` 对话轮数 | 10 |
| `async` | 只有配置项，未接线（当前并发能力来自 FastAPI + asyncio 本身） | — |

评测脚本读的是同一套配置，`--recall` / `--top-k` / `--rrf-k` / `--lambda-mult` 可临时覆盖，不用改文件。

## 设计取舍

几个决策的理由，比结论本身更重要：

- **RRF 用排名倒数而不是分数加权**：向量相似度是 0~1 的余弦，BM25 分数无上界，两者量纲不可比，直接加权会由分数范围主导结果。名次是可比的，所以融合只用名次。
- **去重用内容哈希而不是 Chroma id**：向量路返回的 Document 不带 store id，BM25 路带，同一块在两条路边标识不同，用 id 去重会让融合直接失效。内容哈希是唯一在两路都成立的键。
- **BM25 必须先 jieba 分词**：`rank_bm25` 按空格切词，中文没有空格，不分会退化成按字符切，`idf` 完全失真。
- **精排在子块上做、再展开父块**：子块语义聚焦，cross-encoder 的 512 token 窗口吃得下且不被无关内容稀释；父块只在最后一步替换，保证喂给 LLM 的是完整上下文。展开时按 `parent_doc_id` 去重，避免多个子块命中同一父块把它重复塞进上下文。
- **BM25 索引全量常驻内存**：万级以下分块没问题，十万级需要换成倒排索引或 Elasticsearch。属于已知的规模上限，不是疏忽。
- **Agent 没查库就一定回退固定管线**：直觉上"允许模型复用上一轮已检索的资料"更聪明，实测相反 ——
  qwen2.5:7b 对话累积几轮后干脆不再调用工具，直接凭上下文作答，端到端全对率从固定管线的 0.667 掉到 0.167，
  其中 83% 的题一次检索都没发生。所以这里不让步：没有检索证据的答案一律视为不可信，退回固定管线重答。
- **Agent 用独立低温度实例**：工具调用要求输出结构稳定，生成用的 temperature 0.7 会让 7B 模型吐出畸形 tool_call，
  所以 `agent.temperature` 单独设 0.1，而不是全局调低牺牲生成质量。
- **MCP 层只做协议适配，不重写检索**：`mcp_server.py` 里每个工具都是转发到 `RAGChain.retrieve` / `AgentRAGChain.ask`，
  避免出现第二套检索实现和主链路漂移。模型实例也做成懒加载，否则客户端握手会等精排模型载完而超时。
- **上下文自己按预算裁剪，不让推理引擎静默截断**：不显式设 `num_ctx` 时 Ollama 用自己的默认窗口，
  提示词超出后从头部丢弃，而资料就拼在头部 —— 实测 Top K 从 5 调到 18，命中率更高、回答却三条全丢，
  还编出了语料里不存在的部署地址。现在窗口写死 8192，另加 6000 字预算按精排名次从后往前丢，
  并且链路和前端来源共用同一个裁剪入口，不会出现"给用户看了 14 条、模型只看到 9 条"。
- **检索缓存的键必须带配置签名**：`/chat` 会先取来源再走生成，同一问题实际检索两遍，CPU 精排下这份重复很贵。
  但前端能实时改 `k` / 双路 / 精排开关，只按问题文本缓存会让"改了参数没生效"，
  所以键里编进 `k`、`search_type`、`hybrid`、`rerank`，文档增删时再主动清一次。

## 已知限制

诚实清单，不是"未来展望"：

- 评测语料只有 14 个分块、单一文档，召回 20 ≥ 语料总数导致候选集饱和，混合检索的补召回价值尚未验证。
  扩到 100+ 分块之前，检索层的相对结论只在"当前这份库"上成立
- 端到端问答集只有 12 题，且生成温度 0.7、单轮跑一次，**±2 题以内的差异属于噪声**；
  换模型或换语料后必须重跑，别把绝对数字当结论
- Agent 在 7B 本地模型上不稳定：会跳过工具、会伪造 `[1]` 引用、会被隐私理由挡住不作答。
  现在的兜底是"没查库就退回固定管线"，本质是用固定管线上限托底，而不是让 Agent 变强
- Agent 延迟高一个量级（固定管线 2~24s，Agent 15~40s/题），因为一次问答里串了多轮 7B 生成 + 多次精排
- 父子文档（v1.2）仍需在评测里补一组对照：当前库是 recursive 策略入库，不含 `parent_content`，展开逻辑实际是 no-op
- `multi_query` 与 `condense_question`（多轮问题改写）只有配置项，没有实现
- MMR 在当前规模下无效果，需要更大语料重新评估
- 检索缓存是进程内的，多副本之间不共享；TTL 60 秒，只写了"文档增删时主动清"，
  别的进程直接改库（比如另一个终端跑评测）不会被感知
- 无并发压测、无鉴权、无检索日志与可观测性，不具备多用户生产能力
- 向量库为单机 ChromaDB，规模上限在十万分块量级

## 评测复现

```bash
python -m backend.eval.run_eval --check-gold     # 只校验标注与语料是否对得上
python -m backend.eval.run_eval                   # 五种配置全量对比
python -m backend.eval.run_eval --json out.json   # 另存明细

# 端到端：固定管线 vs Agentic RAG（--fresh 逐题清历史，否则 Agent 会被上一轮带偏）
python -m backend.eval.run_agent_eval --engine both --fresh
python -m backend.eval.run_agent_eval --engine agent --fresh --json /tmp/agent_eval.json
```

判分不用 LLM 当裁判（7B 模型判卷本身不可靠还引入随机性），改用人工给定的 `must_contain` 关键词全覆盖判定，
事实级、可复现，还能直接指出漏了哪个事实。

改检索评测动 `backend/eval/qa_dev.yaml`：gold 用"必然出现在正确分块里的特征串"标注，重建索引后不会失效。
改端到端评测动 `backend/eval/agent_dev.yaml`，每题给 `type`（single / multi / rewrite）和 `must_contain`。
注意别把评测集放进 `data/`，那个目录整个在 `.gitignore` 里。

## 许可证

MIT License
