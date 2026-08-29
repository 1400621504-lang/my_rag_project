# RAG 知识库问答系统

基于 LangChain + Ollama 的本地 RAG 知识库问答系统，零 API 费用，数据不出本地。

## 核心功能

- **文档上传与解析** — 支持 PDF / TXT / Markdown / DOCX
- **文档智能切分** — recursive（递归字符切分）/ parent_child（父子文档切分），chunk 大小和重叠量可调
- **向量检索** — ChromaDB 向量库，支持 MMR 多样性检索和相似度检索
- **检索参数可调** — Top K、检索策略、多样性 λ 实时调整
- **多轮对话** — 对话历史记忆，上下文连贯
- **流式输出** — 边生成边显示
- **深色主题 UI** — DeepSeek 风格深色界面，参数面板在侧边栏

## 项目架构

```
my_rag_project/
├── backend/
│   ├── api/
│   │   └── rag_api.py              # FastAPI 异步接口（流式输出）
│   ├── chains/
│   │   └── rag_chain.py            # RAG 主链（检索+生成）
│   ├── models/
│   │   └── embedding_factory.py    # Embedding 工厂（Ollama/HF/API）
│   └── services/
│       ├── document_processor.py   # 文档解析+切分
│       └── vectorstore_service.py  # ChromaDB 向量库管理
├── config/
│   └── config.yaml                 # 全局配置
├── frontend/
│   ── app.py                      # Streamlit 深色主题界面
├── .gitignore
├── README.md
└── requirements.txt
```

## 快速开始

### 1. 前置条件
- 安装 [Ollama](https://ollama.com/)
- 拉取模型：`ollama pull qwen2.5:7b` 和 `ollama pull bge-m3`

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 启动
```bash
streamlit run frontend/app.py
```

### 4. 访问界面
打开浏览器：http://localhost:8501

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM | Ollama + qwen2.5:7b |
| Embedding | Ollama + bge-m3（1024维） |
| 向量数据库 | ChromaDB |
| RAG 框架 | LangChain |
| 前端 | Streamlit |
| API | FastAPI（异步） |

## 配置说明

编辑 `config/config.yaml` 可调整：
- **切分策略**：recursive / parent_child
- **Chunk Size**：默认 800 字符
- **Overlap**：默认 200 字符
- **检索策略**：mmr / similarity
- **Top K**：默认 5
- **多样性 λ**：默认 0.7

## 许可证

MIT License
