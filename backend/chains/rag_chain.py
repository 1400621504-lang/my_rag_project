"""RAG 主链 - 包含所有高级功能

功能：
1. 从真实向量库检索（ChromaDB）
2. MMR 多样性检索
3. 流式输出（提高体验）
4. 对话历史（多轮对话）
5. 异步高并发（提高性能）
"""
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict
import time
import yaml
from pathlib import Path


def format_docs(docs: List) -> str:
    """把检索到的文档拼成给 LLM 看的上下文文本

    带 [序号] 和来源，方便模型在回答里用 [1] [2] 标注引用。
    固定管线和 Agentic RAG 共用这一份，保证两条路径下模型看到的格式一致。
    """
    if not docs:
        return "（未检索到相关资料）"
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', '未知')
        formatted.append(f"[{i}] 来源：{source}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


class RAGChain:
    """RAG 主链：召回 → RRF 融合 → 精排 → 父块展开 → 生成

    固定管线（single-shot）：一次检索、一次生成，延迟可控，
    适合单跳事实问答。多跳/信息不足的问题交给 agent_chain.AgentRAGChain。
    """

    def __init__(self, config_path: str = None):
        """初始化 RAG 链"""
        # 加载配置
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 初始化配置
        self.retrieval_config = self.config.get('retrieval', {})
        conversation_config = self.config.get('conversation', {})
        self.max_history = conversation_config.get('max_history', 10)

        # Rerank 服务（单例，懒加载模型）
        from backend.services.rerank_service import RerankService
        self.reranker = RerankService()
        self.rerank_enabled = self.reranker.enabled

        # 双路检索（向量 + BM25），融合配置
        self.hybrid_config = self.retrieval_config.get('hybrid', {})
        self.hybrid_enabled = self.hybrid_config.get('enabled', False)
        from backend.services.bm25_service import BM25Service
        self.bm25 = BM25Service()

        # 检索结果缓存（同一次问答会被检索两遍时用得上）
        cache_config = self.config.get('cache', {})
        self.cache_enabled = cache_config.get('enabled', True)
        self.cache_ttl = float(cache_config.get('ttl', 60))
        self.cache_max_size = int(cache_config.get('max_size', 200))
        self._retrieval_cache: "OrderedDict[str, Tuple[float, List]]" = OrderedDict()

        # 对话历史（自己管理，不依赖 langchain.memory）
        self.chat_messages: List = []  # 存储 HumanMessage 和 AIMessage

        # 初始化组件
        self.embeddings = self._init_embeddings()
        self.llm = self._init_llm()
        self.retriever = self._init_retriever()

        # 创建链（依赖 self.retriever / self.reranker，须在其后）
        self.chain = self._create_chain()

    def _init_embeddings(self):
        """初始化 Embedding"""
        from backend.models.embedding_factory import EmbeddingFactory
        return EmbeddingFactory.create()

    def _init_llm(self):
        """初始化 LLM"""
        llm_config = self.config['llm']
        if llm_config['type'] == 'api':
            api_config = llm_config['api']
            return ChatOpenAI(
                model=api_config['model_name'],
                api_key=api_config['api_key'],
                base_url=api_config['base_url'],
                temperature=api_config.get('temperature', 0.7),
                streaming=api_config.get('stream', True)
            )
        else:
            local_config = llm_config['local']
            return ChatOllama(
                model=local_config['model_name'],
                base_url=local_config['base_url'],
                temperature=local_config.get('temperature', 0.7),
            )

    def _init_retriever(self):
        """初始化检索器 - 使用真实向量库

        若开启 Rerank，检索器负责"召回"更多候选（recall_k），
        真正的"最终数量"由后续 rerank 精排决定；否则直接召回最终数量。
        """
        from backend.services.vectorstore_service import VectorstoreService

        vs_service = VectorstoreService()

        search_type = self.retrieval_config.get('search_type', 'mmr')
        search_kwargs = self.retrieval_config.get('search_kwargs', {})

        final_k = search_kwargs.get('k', 5)
        lambda_mult = search_kwargs.get('lambda_mult', 0.7)

        if self.rerank_enabled:
            # 召回阶段放大：取候选数与最终数的较大者，保证精排有足够料
            recall_k = max(self.reranker.candidate_k, final_k)
        else:
            recall_k = final_k

        fetch_k = search_kwargs.get('fetch_k', 20)
        # MMR 的 fetch_k 必须 >= 召回数
        fetch_k = max(fetch_k, recall_k)

        return vs_service.get_retriever(
            search_type=search_type,
            k=recall_k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )

    def _retrieve_rerank(self, question: str) -> List:
        """统一的检索入口：召回 →（可选）融合 →（可选）精排 → 最终文档

        完整 RAG 检索管线（对标 QAnything）：
          双路检索开启时：向量召回 + BM25 召回 → RRF 融合
          否则：仅向量召回
          精排开启时：对候选集做 cross-encoder rerank 取 top_k
        链里的 context 组装和前端展示来源都走这个方法，
        保证 LLM 看到的和前端列出的来源是同一批。
        """
        final_k = self.retrieval_config.get('search_kwargs', {}).get('k', 5)

        cache_key = self._cache_key(final_k)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # 第一阶段：召回
        if self.hybrid_enabled:
            from backend.retriever.fusion import reciprocal_rank_fusion
            vector_docs = self.retriever.invoke(question)
            bm25_docs = self.bm25.search(
                question, k=self.hybrid_config.get('bm25_top_k', 20)
            )
            # 融合后的候选数：精排开则多留点给 rerank 挑，否则直接到最终数
            fused_n = max(self.reranker.candidate_k, final_k) if self.rerank_enabled else final_k
            docs = reciprocal_rank_fusion(
                [vector_docs, bm25_docs],
                k=self.hybrid_config.get('rrf_k', 60),
                top_n=fused_n,
            )
        else:
            docs = self.retriever.invoke(question)

        # 第二阶段：精排（在子块上做，语义聚焦更准）
        if self.rerank_enabled and docs:
            docs = self.reranker.rerank(question, docs, top_n=final_k)

        # 第三阶段：子块 → 父块展开（small-to-big）
        docs = self._expand_to_parents(docs)

        self._cache_put(cache_key, docs)
        return docs

    def _cache_key(self, final_k: int) -> str:
        """缓存键必须带上检索配置

        前端能实时改 k / 双路 / 精排 / 检索策略，只按问题缓存会让调完参数
        还拿到旧结果，看起来就像参数没生效。
        """
        return "|".join([
            f"k={final_k}",
            f"type={self.retrieval_config.get('search_type', 'mmr')}",
            f"hybrid={int(bool(self.hybrid_enabled))}",
            f"rerank={int(bool(self.rerank_enabled))}",
        ]) + "||"

    def _cache_get(self, key: str) -> Optional[List]:
        """命中就顺带做 LRU 提权，过期就删掉"""
        if not self.cache_enabled:
            return None
        entry = self._retrieval_cache.get(key)
        if entry is None:
            return None
        created_at, docs = entry
        if time.perf_counter() - created_at > self.cache_ttl:
            del self._retrieval_cache[key]
            return None
        self._retrieval_cache.move_to_end(key)
        return docs

    def _cache_put(self, key: str, docs: List):
        if not self.cache_enabled:
            return
        self._retrieval_cache[key] = (time.perf_counter(), docs)
        self._retrieval_cache.move_to_end(key)
        while len(self._retrieval_cache) > self.cache_max_size:
            self._retrieval_cache.popitem(last=False)

    def clear_retrieval_cache(self):
        """文档增删后手动清一下，避免 TTL 窗口内拿到旧结果"""
        self._retrieval_cache.clear()

    def retrieve(self, question: str) -> List:
        """对外公开的检索入口（Agentic RAG 的工具、MCP server 都复用它）

        与内部 _retrieve_rerank 等价，只是给外部调用一个稳定契约，
        避免 Agent 侧依赖下划线开头的私有方法。
        """
        return self._retrieve_rerank(question)

    def _expand_to_parents(self, docs: List) -> List:
        """把命中的子块替换为其父块（父子检索的最后一步）

        - 文档若不带 parent_content（recursive 切分），原样返回。
        - 多个子块命中同一父块时按 parent_doc_id 去重，
          保留首次出现的那条（= 精排分最高），避免父块重复占满上下文。
        - 展开后的父块继承子块已有的 rerank_score / rrf_score 等元数据。
        """
        if not docs:
            return docs

        seen_parents = set()
        expanded = []
        for doc in docs:
            parent_content = doc.metadata.get('parent_content')
            if not parent_content:
                expanded.append(doc)  # 非父子模式
                continue

            pid = doc.metadata.get('parent_doc_id') or doc.metadata.get('parent_id')
            if pid in seen_parents:
                continue
            seen_parents.add(pid)

            meta = {k: v for k, v in doc.metadata.items() if k != 'parent_content'}
            meta['chunk_type'] = 'parent'
            meta['matched_child'] = doc.page_content[:60]  # 调试：记录命中的子块
            expanded.append(Document(page_content=parent_content, metadata=meta))

        return expanded

    def _create_chain(self):
        """创建 RAG 链

        流程：问题 → 检索 → 拼接上下文 → 提示词 → LLM → 回答
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的 AI 助手。根据以下检索到的资料回答用户的问题。

规则：
1. 优先使用检索到的资料回答
2. 如果资料中没有相关信息，请明确说明"根据现有资料无法回答"
3. 回答要简洁准确，不要编造信息
4. 如果涉及多个资料来源，请综合回答

检索到的资料：
{context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        chain = (
            {
                "context": RunnableLambda(self._retrieve_rerank) | format_docs,
                "question": RunnablePassthrough(),
                "chat_history": lambda x: self.chat_messages[-self.max_history * 2:]
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def _save_to_memory(self, question: str, answer: str):
        """保存对话到历史"""
        self.chat_messages.append(HumanMessage(content=question))
        self.chat_messages.append(AIMessage(content=answer))

    def invoke(self, question: str) -> str:
        """同步调用 RAG 链"""
        result = self.chain.invoke(question)
        self._save_to_memory(question, result)
        return result

    async def ainvoke(self, question: str) -> str:
        """异步调用 RAG 链"""
        result = await self.chain.ainvoke(question)
        self._save_to_memory(question, result)
        return result

    def stream(self, question: str):
        """流式输出"""
        full_answer = []
        for chunk in self.chain.stream(question):
            full_answer.append(chunk)
            yield chunk

        # 流式结束后保存到对话历史
        self._save_to_memory(question, "".join(full_answer))

    def get_sources(self, question: str) -> List[Dict[str, Any]]:
        """获取检索来源信息（不经过 LLM），与 LLM 看到的是同一批文档"""
        docs = self._retrieve_rerank(question)
        sources = []
        for doc in docs:
            src = {
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "source": doc.metadata.get('source', '未知'),
                "chunk_type": doc.metadata.get('chunk_type', 'unknown'),
            }
            if 'rerank_score' in doc.metadata:
                src['rerank_score'] = round(doc.metadata['rerank_score'], 3)
            sources.append(src)
        return sources

    def clear_memory(self):
        """清空对话历史"""
        self.chat_messages.clear()

    def get_memory_stats(self) -> dict:
        """获取对话历史统计"""
        return {
            "total_messages": len(self.chat_messages),
            "total_turns": len(self.chat_messages) // 2,
        }


# 测试
if __name__ == "__main__":
    rag = RAGChain()
    result = rag.invoke("什么是机器学习？")
    print(result)
