# 本地 RAG 知识库问答系统

基于 LangChain 1.x + Ollama 的本地全栈 RAG：向量召回 + BM25 关键词召回 + RRF 融合 + bge-reranker 精排 + 父子文档展开。零 API 费用，数据不出本机。

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
| 服务 | FastAPI 异步 10 个接口，Streamlit 深色前端，参数实时可调 | v1.0 |
| 评测 | Hit@K / MRR / nDCG 消融脚本 + 28 条标注问题集 | v1.2+ |

## 项目结构

```
my_rag_project/
├── backend/
│   ├── api/rag_api.py                  251 行  # FastAPI：/chat /chat/stream /upload /documents /memory /health /stats
│   ├── chains/rag_chain.py             292 行  # 主链：召回→融合→精排→父块展开→生成，多轮记忆
│   ├── models/embedding_factory.py              # Embedding 工厂（ollama / local_hf / api 三选一）
│   ├── retriever/fusion.py              63 行  # RRF 融合
│   ├── services/
│   │   ├── vectorstore_service.py      388 行  # ChromaDB 封装：增删查、MMR、统计、chunk 可视化
│   │   ├── document_processor.py       311 行  # 解析 + 两套切分策略 + 扫描页 OCR 回退
│   │   ├── bm25_service.py             121 行  # BM25 内存索引，语料变化自动重建
│   │   ├── rerank_service.py           138 行  # bge-reranker 懒加载单例
│   │   └── ocr_service.py              131 行  # RapidOCR（ONNX，离线），不可用时自动降级
│   └── eval/                           277 行  # 检索层消融评测（指标 + 脚本 + 问题集 + 结果分析）
├── config/config.yaml                            # 全量配置：模型 / 切分 / 检索 / 对话 / OCR
├── frontend/app.py                     653 行  # Streamlit 深色界面，检索与模型参数侧边栏实时调
└── requirements.txt                              # 42 行，fresh clone 可直接 pip install
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

## 配置说明

`config/config.yaml` 全量可调：

| 段落 | 关键项 | 默认 |
|---|---|---|
| `embedding` | `type`（ollama / local_hf / api）、模型名 | ollama + bge-m3 |
| `rerank` | `enabled`、`candidate_k`（精排前召回数） | true / 20 |
| `llm` | `type`（local / api）、模型、温度 | local + qwen2.5:7b |
| `chunking` | `strategy`（recursive / parent_child）、父子块大小与重叠 | recursive，800/200，子块 200/50 |
| `retrieval` | `search_type`（mmr / similarity）、`k`、`fetch_k`、`lambda_mult` | mmr / 5 / 20 / 0.7 |
| `retrieval.hybrid` | `enabled`、`bm25_top_k`、`rrf_k` | false / 20 / 60 |
| `ocr` | `enabled`、`pdf_dpi`、`min_page_chars`（页面可提取文字低于此值判为扫描件） | true / 200 / 20 |
| `conversation` | `max_history` 对话轮数 | 10 |

评测脚本读的是同一套配置，`--recall` / `--top-k` / `--rrf-k` / `--lambda-mult` 可临时覆盖，不用改文件。

## 设计取舍

几个决策的理由，比结论本身更重要：

- **RRF 用排名倒数而不是分数加权**：向量相似度是 0~1 的余弦，BM25 分数无上界，两者量纲不可比，直接加权会由分数范围主导结果。名次是可比的，所以融合只用名次。
- **去重用内容哈希而不是 Chroma id**：向量路返回的 Document 不带 store id，BM25 路带，同一块在两条路边标识不同，用 id 去重会让融合直接失效。内容哈希是唯一在两路都成立的键。
- **BM25 必须先 jieba 分词**：`rank_bm25` 按空格切词，中文没有空格，不分会退化成按字符切，`idf` 完全失真。
- **精排在子块上做、再展开父块**：子块语义聚焦，cross-encoder 的 512 token 窗口吃得下且不被无关内容稀释；父块只在最后一步替换，保证喂给 LLM 的是完整上下文。展开时按 `parent_doc_id` 去重，避免多个子块命中同一父块把它重复塞进上下文。
- **BM25 索引全量常驻内存**：万级以下分块没问题，十万级需要换成倒排索引或 Elasticsearch。属于已知的规模上限，不是疏忽。

## 已知限制

诚实清单，不是"未来展望"：

- 评测语料只有 14 个分块、单一文档，召回阶段饱和，混合检索的补召回价值尚未验证
- 父子文档（v1.2）需要在评测里补一组对照：当前库是 recursive 策略入库，不含 `parent_content`，展开逻辑实际是 no-op
- `multi_query` 与 `condense_question`（多轮问题改写）只有配置项，没有实现
- MMR 在当前规模下无效果，需要更大语料重新评估
- 无并发压测、无鉴权、无检索日志与可观测性，不具备多用户生产能力
- 生成层质量没有量化指标，目前只评了检索
- 向量库为单机 ChromaDB，规模上限在十万分块量级

## 评测复现

```bash
python -m backend.eval.run_eval --check-gold     # 只校验标注与语料是否对得上
python -m backend.eval.run_eval                   # 五种配置全量对比
python -m backend.eval.run_eval --json out.json   # 另存明细
```

新增查询直接改 `backend/eval/qa_dev.yaml`：gold 用"必然出现在正确分块里的特征串"标注，
重建索引后不会失效。注意别把评测集放进 `data/`，那个目录整个在 `.gitignore` 里。

## 许可证

MIT License
