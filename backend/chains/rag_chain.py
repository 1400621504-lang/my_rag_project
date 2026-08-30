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
import hashlib
import time
import yaml
from pathlib import Path


def fit_context_budget(docs: List, max_chars: Optional[int]) -> List:
    """按字符预算裁剪最终喂给模型的资料

    存在的理由：Ollama 的上下文窗口是有限的，超出的部分会被从头部静默丢弃，
    结果是"检索明明命中了，模型却说不知道"。与其让它截断，不如我们自己
    按名次从后往前丢，保证留下的永远是精排分最高的那几条。
    至少保留一条，哪怕它自己就超预算（截断单条比丢掉整条信息损失小）。
    """
    if not max_chars or max_chars <= 0 or not docs:
        return docs
    kept: List = []
    used = 0
    for doc in docs:
        # 40 是序号与来源标签的额外开销
        cost = len(doc.page_content) + 40
        if kept and used + cost > max_chars:
            break
        kept.append(doc)
        used += cost
    return kept


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
        # 上下文字符预算：从 retrieval 段读，不跟着前端重建的 retrieval_config 走，
        # 否则前端每次改参数都会把这个值丢掉
        self.max_context_chars = int(
            self.config.get('retrieval', {}).get('max_context_chars', 6000)
        )
        self._retrieval_cache: "OrderedDict[str, Tuple[float, List]]" = OrderedDict()
        # 本轮检索/上下文块数，仅供前端状态行展示
        self.last_retrieved_count = 0
        self.last_context_count = 0

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
                # 不显式给 num_ctx 的话，Ollama 会用它自己的默认窗口，
                # 检索资料一多就会被静默截断（实测见 RESULTS.md 的 F13）
                num_ctx=int(local_config.get('num_ctx', 8192)),
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

        cache_key = self._cache_key(question, final_k)
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

    def _cache_key(self, question: str, final_k: int) -> str:
        """缓存键 = 问题 + 全部影响检索结果的参数

        两个坑都踩过：
        - 不带问题（旧实现）→ TTL 内所有问题共用同一份资料，换个问题
          拿到的是上一个问题的上下文，答非所问（实测见 RESULTS.md 的 F14）。
        - 不带参数 → 前端调完 k / 双路 / 精排还拿旧结果，看起来像参数没生效。
        """
        sk = self.retrieval_config.get('search_kwargs', {}) or {}
        return "|".join([
            f"q={hashlib.sha1(question.strip().encode('utf-8')).hexdigest()[:16]}",
            f"k={final_k}",
            f"type={self.retrieval_config.get('search_type', 'mmr')}",
            f"fetch={sk.get('fetch_k', 20)}",
            f"lambda={sk.get('lambda_mult', 0.7)}",
            f"hybrid={int(bool(self.hybrid_enabled))}",
            f"bm25={self.hybrid_config.get('bm25_top_k', 20)}",
            f"rerank={int(bool(self.rerank_enabled))}",
            f"recall={self.reranker.candidate_k}",
        ])

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

    def _context_docs(self, question: str) -> List:
        """最终进入提示词的那批资料：检索结果再按字符预算裁剪

        链路的 {context} 和前端的来源列表都走这里，
        保证"列给用户看的"和"模型实际看到的"是同一批。
        公开的 retrieve() 不裁剪，评测要量的是检索本身。

        顺带把裁剪前后的块数记在实例上，前端状态行要用它区分
        "Top K 没生效"和"检索到了但超上下文预算被裁掉"这两种完全不同的情况。
        """
        docs = self._retrieve_rerank(question)
        final = fit_context_budget(docs, self.max_context_chars)
        self.last_retrieved_count = len(docs)
        self.last_context_count = len(final)
        return final

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
                "context": RunnableLambda(self._context_docs) | format_docs,
                "question": RunnablePassthrough(),
                "chat_history": lambda x: self._history_for(x)
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def _history_for(self, question: str) -> List:
        """本轮真正带进提示词的历史

        同一个问题连着问时，上一轮那条"同题问答"若还留在历史里，
        模型会直接照抄自己上次的答案，检索参数改了也看不出来变化。
        所以把末尾与当前问题相同的问答对先剥掉。
        """
        history = self.chat_messages[-self.max_history * 2:]
        target = (question or "").strip()
        while len(history) >= 2 and str(history[-2].content).strip() == target:
            history = history[:-2]
        return history

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
        docs = self._context_docs(question)
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

    def apply_llm_params(self, temperature=None, max_tokens=None):
        """把前端调的 Temperature / Max Tokens 真正写进模型对象

        滑块只读不写的话就是个摆设。ChatOllama 的采样上限字段叫
        num_predict，ChatOpenAI 叫 max_tokens，这里统一屏蔽掉差异。
        """
        is_api = self.config['llm']['type'] == 'api'
        if temperature is not None:
            self.llm.temperature = float(temperature)
        if max_tokens is not None:
            if is_api:
                self.llm.max_tokens = int(max_tokens)
            else:
                self.llm.num_predict = int(max_tokens)

    def clear_memory(self):
        """清空对话历史（同时清检索缓存，避免清完还拿到旧资料）"""
        self.chat_messages.clear()
        self.clear_retrieval_cache()

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
