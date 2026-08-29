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
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict, Any, Optional
import yaml
from pathlib import Path


class RAGChain:
    """RAG 主链

    包含所有高级功能的生产级 RAG 系统
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

        # 对话历史（自己管理，不依赖 langchain.memory）
        self.chat_messages: List = []  # 存储 HumanMessage 和 AIMessage

        # 初始化组件
        self.embeddings = self._init_embeddings()
        self.llm = self._init_llm()
        self.retriever = self._init_retriever()

        # 创建链
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
        """初始化检索器 - 使用真实向量库"""
        from backend.services.vectorstore_service import VectorstoreService

        vs_service = VectorstoreService()

        search_type = self.retrieval_config.get('search_type', 'mmr')
        search_kwargs = self.retrieval_config.get('search_kwargs', {})

        k = search_kwargs.get('k', 5)
        fetch_k = search_kwargs.get('fetch_k', 20)
        lambda_mult = search_kwargs.get('lambda_mult', 0.7)

        return vs_service.get_retriever(
            search_type=search_type,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )

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

        def format_docs(docs):
            """将检索到的文档格式化为文本"""
            if not docs:
                return "（未检索到相关资料）"
            formatted = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', '未知')
                formatted.append(f"[{i}] 来源：{source}\n{doc.page_content}")
            return "\n\n---\n\n".join(formatted)

        chain = (
            {
                "context": self.retriever | format_docs,
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
        """获取检索来源信息（不经过 LLM）"""
        docs = self.retriever.invoke(question)
        sources = []
        for doc in docs:
            sources.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "source": doc.metadata.get('source', '未知'),
                "chunk_type": doc.metadata.get('chunk_type', 'unknown'),
            })
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
