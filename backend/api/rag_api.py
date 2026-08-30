"""FastAPI 异步接口 - 高并发 RAG 服务

功能：
1. 异步处理（高并发）
2. 流式输出（SSE）
3. 对话管理
4. 文档上传、解析、向量化
5. 文档管理（列表、删除）
6. 健康检查 + 统计信息
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import json

app = FastAPI(title="RAG API", version="1.0.0")

# 跨域配置（允许 Streamlit 前端访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局实例
rag_chain = None
document_processor = None
agent_chain = None  # Agentic RAG，复用 rag_chain 的检索与模型，懒加载


def _invalidate_retrieval_cache():
    """文档增删后清掉检索缓存

    缓存本身有 TTL 兜底，但用户在界面上删完文档马上问，
    60 秒窗口里还会拿到旧结果，这里主动失效一次。
    """
    if rag_chain is not None:
        rag_chain.clear_retrieval_cache()


class ChatRequest(BaseModel):
    """聊天请求"""
    question: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    """聊天响应"""
    answer: str
    sources: List[dict] = []
    conversation_id: str


class AgentResponse(BaseModel):
    """Agent 响应

    比普通 RAG 多了 steps（工具调用轨迹）和 search_calls（检索次数），
    前端可以据此展示"模型查了什么"。
    """
    answer: str
    sources: List[dict] = []
    steps: List[dict] = []
    search_calls: int = 0
    answer_source: str = "agent"
    latency_ms: int = 0


class UploadResponse(BaseModel):
    """上传响应"""
    message: str
    filename: str
    chunk_count: int
    total_chunks: int


@app.on_event("startup")
async def startup_event():
    """启动时初始化 RAG 链和文档处理器"""
    global rag_chain, document_processor

    from backend.chains.rag_chain import RAGChain
    from backend.services.document_processor import DocumentProcessor

    rag_chain = RAGChain()
    document_processor = DocumentProcessor()
    print("✅ RAG 链和文档处理器初始化完成")


# ==================== 聊天接口 ====================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """聊天接口（异步）

    接收用户问题，检索知识库，调用 LLM 生成回答。
    """
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="RAG 链未初始化")

    try:
        # 先获取检索来源
        sources = rag_chain.get_sources(request.question)

        # 异步调用 RAG 链
        answer = await rag_chain.ainvoke(request.question)

        return ChatResponse(
            answer=answer,
            sources=sources,
            conversation_id=request.conversation_id or "default"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败：{str(e)}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """聊天接口（流式输出 SSE）

    边生成边返回，适合长回答场景。
    """
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="RAG 链未初始化")

    def generate():
        try:
            for chunk in rag_chain.stream(request.question):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ==================== Agentic RAG 接口 ====================

def _get_agent():
    """懒加载 Agent（首轮请求才付一次建图代价，且复用已建好的 rag_chain）"""
    global agent_chain
    if agent_chain is None:
        from backend.chains.agent_chain import AgentRAGChain
        agent_chain = AgentRAGChain(rag=rag_chain)
    return agent_chain


@app.post("/agent/chat", response_model=AgentResponse)
async def agent_chat(request: ChatRequest):
    """Agent 问答：由模型自主决定是否检索、检索几次、怎么改写检索词

    与 /chat 的差别：/chat 是固定管线（一次检索一次生成），
    这里允许多跳检索，代价是延迟更高，且没查库时会自动回退到固定管线。
    """
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="RAG 链未初始化")
    try:
        agent = _get_agent()
        # 检索+生成是同步阻塞的，放线程池跑，避免卡死事件循环
        result = await asyncio.to_thread(agent.ask, request.question)
        return AgentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 处理失败：{str(e)}")


@app.post("/agent/chat/stream")
async def agent_chat_stream(request: ChatRequest):
    """Agent 问答（流式 SSE）

    每个事件是一行 JSON：{"type": "step|token|reset|done", ...}
    reset 表示模型这轮其实在发起工具调用，前面吐出的文字要丢弃。
    """
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="RAG 链未初始化")

    def generate():
        try:
            agent = _get_agent()
            for event in agent.stream_ask(request.question):
                yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    # 同步生成器交给 starlette 放到线程池里迭代，不会阻塞事件循环
    return StreamingResponse(generate(), media_type="text/event-stream")


# ==================== 文档管理接口 ====================

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """上传文档并入库

    流程：上传 → 解析 → 切分 → 向量化 → 存入 ChromaDB
    支持格式：PDF、TXT、Markdown、DOCX
    """
    if document_processor is None:
        raise HTTPException(status_code=500, detail="文档处理器未初始化")

    # 检查文件类型
    allowed_types = {'.pdf', '.txt', '.md', '.docx',
                     '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    filename = file.filename or "unknown"
    suffix = filename.rfind('.')
    if suffix == -1 or filename[suffix:].lower() not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式，仅支持：{', '.join(allowed_types)}"
        )

    try:
        # 读取文件内容
        content = await file.read()

        # 解析 + 切分
        documents = document_processor.process_bytes(content, filename)

        if not documents:
            raise HTTPException(status_code=400, detail="文件内容为空或无法解析")

        # 向量化并存入 ChromaDB
        from backend.services.vectorstore_service import VectorstoreService
        vs_service = VectorstoreService()
        ids = vs_service.add_documents(documents)

        # 获取当前总块数
        stats = vs_service.get_stats()

        _invalidate_retrieval_cache()
        return UploadResponse(
            message=f"文档 {filename} 上传成功",
            filename=filename,
            chunk_count=len(documents),
            total_chunks=stats['total_chunks']
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档处理失败：{str(e)}")


@app.get("/documents")
async def list_documents():
    """列出知识库中所有文档"""
    from backend.services.vectorstore_service import VectorstoreService
    vs_service = VectorstoreService()
    docs = vs_service.list_documents()
    return {"documents": docs, "count": len(docs)}


@app.delete("/documents/{source}")
async def delete_document(source: str):
    """删除指定文档

    Args:
        source: 文档来源名称（文件名）
    """
    from backend.services.vectorstore_service import VectorstoreService
    vs_service = VectorstoreService()
    success = vs_service.delete_by_source(source)
    if success:
        _invalidate_retrieval_cache()
        return {"message": f"文档 {source} 已删除"}
    else:
        raise HTTPException(status_code=404, detail=f"文档 {source} 不存在")


@app.delete("/documents")
async def clear_documents():
    """清空知识库所有文档"""
    from backend.services.vectorstore_service import VectorstoreService
    vs_service = VectorstoreService()
    success = vs_service.clear_all()
    if success:
        _invalidate_retrieval_cache()
        return {"message": "知识库已清空"}
    else:
        raise HTTPException(status_code=500, detail="清空失败")


# ==================== 对话管理接口 ====================

@app.post("/memory/clear")
async def clear_memory():
    """清空对话历史"""
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="RAG 链未初始化")
    rag_chain.clear_memory()
    return {"message": "对话历史已清空"}


@app.get("/memory/stats")
async def memory_stats():
    """获取对话历史统计"""
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="RAG 链未初始化")
    return rag_chain.get_memory_stats()


# ==================== 系统接口 ====================

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok"}


@app.get("/stats")
async def system_stats():
    """系统统计信息"""
    from backend.services.vectorstore_service import VectorstoreService
    vs_service = VectorstoreService()
    vs_stats = vs_service.get_stats()

    memory_stats = {}
    if rag_chain:
        memory_stats = rag_chain.get_memory_stats()

    return {
        "vectorstore": vs_stats,
        "conversation": memory_stats,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
