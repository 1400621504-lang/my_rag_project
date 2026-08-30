"""MCP server —— 把本地知识库暴露成模型上下文协议工具

为什么要这一层：RAG 系统如果只有一层 HTTP 接口，就只能被自己的前端调用。
包成 MCP server 之后，任何支持 MCP 的客户端（Claude Desktop、Codex、
Cursor 等）都能把这个本地知识库当成自己的长期记忆来用，不用重复造接入层。

设计取舍：
- 复用 backend.chains 里已有的 RAGChain / AgentRAGChain，MCP 层只做协议适配，
  不重写检索逻辑，避免两套实现漂移。
- 启动时不加载模型（embedding 走 HTTP、rerank 是几百 MB 权重），
  首次调用才初始化，否则 MCP 客户端握手容易超时。
- 除了问答检索，额外提供 add_file / add_text 两个写入工具：
  MCP 客户端只能传文本，没法走 HTTP 上传文件，给它一个入库的口子才闭环。

启动：
    python -m backend.api.mcp_server                 # stdio（默认）
然后在 MCP 客户端配置里加：
    command: "/opt/anaconda3/bin/python"
    args: ["-m", "backend.api.mcp_server"]
    cwd: "/Users/apple/Desktop/AI_Learning/my_rag_project"
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer

server = MCPServer(
    name="local-rag-kb",
    version="1.0",
    instructions=(
        "这是一个本地运行的 RAG 知识库（向量检索 + BM25 双路召回 + RRF 融合 + cross-encoder 精排）。"
        "回答与用户笔记、项目、学业相关的问题时，先用 search_knowledge_base 或 agent_ask 查证，"
        "不要凭常识直接回答。库里查不到就如实说查不到。"
    ),
)

# 延迟初始化的全局实例
_rag = None
_agent = None


def _get_rag():
    """首次调用才建 RAGChain（要连 Ollama、载 rerank 模型）"""
    global _rag
    if _rag is None:
        from backend.chains.rag_chain import RAGChain
        _rag = RAGChain()
    return _rag


def _get_agent():
    """Agent 复用同一个 RAGChain，避免重复加载 embedding 与精排模型"""
    global _agent
    if _agent is None:
        from backend.chains.agent_chain import AgentRAGChain
        _agent = AgentRAGChain(rag=_get_rag())
    return _agent


def _docs_to_items(docs, limit: int = 800) -> List[Dict[str, Any]]:
    """把 LangChain Document 转成 MCP 友好的字典"""
    items = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        rerank_score = meta.get("rerank_score")
        items.append(
            {
                "index": i,
                "source": meta.get("source", "未知"),
                "chunk_type": meta.get("chunk_type", "unknown"),
                "rerank_score": round(rerank_score, 3) if isinstance(rerank_score, float) else None,
                "content": doc.page_content[:limit],
            }
        )
    return items


@server.tool(title="检索知识库")
def search_knowledge_base(query: str, top_k: int = 5) -> Dict[str, Any]:
    """只检索不生成：返回命中的资料片段、来源和精排分数。

    适合你自己已经有推理能力、只缺本地资料的时候用；
    想要一段完整的中文回答请用 agent_ask。
    """
    docs = _get_rag().retrieve(query)[: max(1, min(int(top_k), 20))]
    return {"query": query, "count": len(docs), "results": _docs_to_items(docs)}


@server.tool(title="知识库问答")
def ask(question: str) -> Dict[str, Any]:
    """固定管线问答：一次检索 + 一次生成，延迟低，适合单跳事实问题。"""
    rag = _get_rag()
    return {
        "question": question,
        "answer": rag.invoke(question),
        "sources": rag.get_sources(question),
        "mode": "single_shot",
    }


@server.tool(title="知识库多跳问答")
def agent_ask(question: str) -> Dict[str, Any]:
    """Agentic 问答：模型自己决定查几次、怎么改写检索词，适合跨多条资料的问题。

    返回里 steps 是工具调用轨迹，answer_source 标出这次到底是 Agent 答的还是
    兜底回退到固定管线答的（没查库就作答会被判为不可信并回退）。
    """
    result = _get_agent().ask(question)
    return {
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"],
        "steps": result["steps"],
        "search_calls": result["search_calls"],
        "answer_source": result["answer_source"],
        "latency_ms": result["latency_ms"],
        "mode": "agent",
    }


@server.tool(title="列出知识库文档")
def list_documents() -> Dict[str, Any]:
    """列出知识库里已有的文档及各自分块数量。"""
    from backend.services.vectorstore_service import VectorstoreService

    docs = VectorstoreService().list_documents()
    return {"count": len(docs), "documents": docs}


@server.tool(title="入库本地文件")
def add_file(path: str) -> Dict[str, Any]:
    """把本机上的一个文件（pdf/docx/md/txt/图片）解析、切分、向量化后写入知识库。

    给 MCP 客户端补上"写"的能力：客户端只能传文本，传不了文件上传请求。
    """
    from pathlib import Path as _Path

    from backend.services.document_processor import DocumentProcessor
    from backend.services.vectorstore_service import VectorstoreService

    file_path = _Path(path).expanduser()
    if not file_path.exists():
        raise ValueError(f"文件不存在：{file_path}")

    processor = DocumentProcessor()
    documents = processor.process_file(str(file_path))
    if not documents:
        raise ValueError(f"文件解析结果为空：{file_path.name}")

    ids = VectorstoreService().add_documents(documents)
    return {
        "filename": file_path.name,
        "chunk_count": len(documents),
        "vector_ids": len(ids),
        "message": f"{file_path.name} 已入库，新增 {len(documents)} 个分块",
    }


@server.tool(title="入库一段文本")
def add_text(text: str, source: str = "mcp_input.md") -> Dict[str, Any]:
    """把一段纯文本切分向量化后写入知识库，source 作为文档名方便后续删除。"""
    if not text.strip():
        raise ValueError("text 为空")

    from backend.services.document_processor import DocumentProcessor
    from backend.services.vectorstore_service import VectorstoreService

    processor = DocumentProcessor()
    # chunk_documents 就是纯文本入口，不需要临时文件绕一圈
    documents = processor.chunk_documents(text, source=source)

    if not documents:
        raise ValueError("文本切分后没有有效内容")

    ids = VectorstoreService().add_documents(documents)
    return {
        "source": source,
        "chunk_count": len(documents),
        "vector_ids": len(ids),
        "message": f"已入库 {len(documents)} 个分块",
    }


@server.tool(title="知识库状态")
def kb_stats() -> Dict[str, Any]:
    """查看知识库规模与检索配置，用来判断当前是不是"语料太少所以指标失真"。"""
    from backend.services.vectorstore_service import VectorstoreService

    rag = _get_rag()
    retrieval = rag.retrieval_config
    return {
        "vectorstore": VectorstoreService().get_stats(),
        "search_type": retrieval.get("search_type"),
        "k": retrieval.get("search_kwargs", {}).get("k"),
        "hybrid_enabled": rag.hybrid_enabled,
        "rerank_enabled": rag.rerank_enabled,
        "chunking_strategy": rag.config.get("chunking", {}).get("strategy"),
    }


@server.resource("kb://documents")
def documents_resource() -> Dict[str, Any]:
    """以 MCP resource 形式暴露文档清单，客户端可直接挂载成上下文来源。"""
    from backend.services.vectorstore_service import VectorstoreService

    return {"documents": VectorstoreService().list_documents()}


if __name__ == "__main__":
    server.run()
