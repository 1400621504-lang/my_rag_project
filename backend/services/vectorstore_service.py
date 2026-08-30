"""向量库服务 - 管理文档的向量化存储与检索

功能：
1. 将文档向量化并存入 ChromaDB
2. 加载已有的向量库
3. 提供检索接口（相似度检索、MMR 检索）
4. 文档管理（添加、删除、列表）
"""
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
import yaml
import shutil


class VectorstoreService:
    """向量库服务 - 单例模式管理 ChromaDB"""

    _instance = None
    _vectorstore = None
    _config = None

    def __new__(cls, config_path: str = None):
        """单例模式 - 确保全局只有一个向量库实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: str = None):
        """初始化向量库服务

        Args:
            config_path: 配置文件路径
        """
        # 避免重复初始化（单例模式下 __init__ 可能多次调用）
        if hasattr(self, '_initialized') and self._initialized:
            return

        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

        self._vectorstore = self._init_vectorstore()
        self._initialized = True

    def _init_vectorstore(self) -> Chroma:
        """初始化 ChromaDB 向量库

        Returns:
            Chroma 向量库实例
        """
        from backend.models.embedding_factory import EmbeddingFactory

        vs_config = self._config.get('vectorstore', {})
        chroma_config = vs_config.get('chromadb', {})

        # 向量库存储路径
        db_path = chroma_config.get('path', './data/chroma_db')
        db_path = Path(__file__).parent.parent.parent / db_path
        db_path.mkdir(parents=True, exist_ok=True)

        collection_name = chroma_config.get('collection_name', 'rag_collection')

        # 创建 Embedding 实例
        embeddings = EmbeddingFactory.create()

        # 加载或创建 ChromaDB
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(db_path)
        )

        return vectorstore

    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档到向量库

        Args:
            documents: 要添加的文档列表

        Returns:
            添加的文档 ID 列表
        """
        if not documents:
            return []

        # 批量添加，避免一次性处理太多
        batch_size = 100
        all_ids = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            ids = self._vectorstore.add_documents(batch)
            all_ids.extend(ids)

        # ChromaDB 自动持久化，无需手动调用

        return all_ids

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """相似度检索

        Args:
            query: 查询文本
            k: 返回的文档数量

        Returns:
            最相似的文档列表
        """
        return self._vectorstore.similarity_search(query, k=k)

    def mmr_search(self, query: str, k: int = 5, fetch_k: int = 20,
                   lambda_mult: float = 0.7) -> List[Document]:
        """MMR 多样性检索

        平衡相关性和多样性，避免返回重复的内容。

        Args:
            query: 查询文本
            k: 最终返回的文档数量
            fetch_k: 候选文档数量（越大多样性越好）
            lambda_mult: 多样性权重（0=多样性优先，1=相关性优先）

        Returns:
            检索到的文档列表
        """
        return self._vectorstore.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )

    def search(self, query: str, search_type: str = "similarity",
               k: int = 5, **kwargs) -> List[Document]:
        """统一检索接口

        Args:
            query: 查询文本
            search_type: 检索类型（similarity / mmr）
            k: 返回数量
            **kwargs: 其他参数（fetch_k, lambda_mult 等）

        Returns:
            检索到的文档列表
        """
        if search_type == "mmr":
            fetch_k = kwargs.get('fetch_k', 20)
            lambda_mult = kwargs.get('lambda_mult', 0.7)
            return self.mmr_search(query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
        else:
            return self.similarity_search(query, k=k)

    def get_retriever(self, search_type: str = "mmr",
                      k: int = 5, **kwargs):
        """获取 LangChain Retriever 接口

        Args:
            search_type: 检索类型
            k: 返回数量
            **kwargs: 其他参数

        Returns:
            LangChain Retriever 对象
        """
        search_kwargs = {"k": k}
        if search_type == "mmr":
            search_kwargs["fetch_k"] = kwargs.get('fetch_k', 20)
            search_kwargs["lambda_mult"] = kwargs.get('lambda_mult', 0.7)

        return self._vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

    def delete_by_source(self, source: str) -> bool:
        """删除指定来源的所有文档

        Args:
            source: 来源文件名

        Returns:
            是否删除成功
        """
        try:
            # 查询该来源的所有文档
            results = self._vectorstore.get(where={"source": source})
            if results and results['ids']:
                self._vectorstore.delete(ids=results['ids'])
                return True
            return False
        except Exception as e:
            print(f"删除文档失败：{e}")
            return False

    def list_documents(self) -> List[dict]:
        """列出向量库中所有文档的信息

        Returns:
            文档信息列表 [{source, chunk_count, ...}, ...]
        """
        try:
            results = self._vectorstore.get(include=['metadatas'])
            if not results or not results['ids']:
                return []

            # 按来源分组统计
            source_stats = {}
            for metadata in results['metadatas']:
                source = metadata.get('source', 'unknown')
                if source not in source_stats:
                    source_stats[source] = {
                        'source': source,
                        'chunk_count': 0,
                        'chunk_types': set()
                    }
                source_stats[source]['chunk_count'] += 1
                chunk_type = metadata.get('chunk_type')
                if chunk_type:
                    source_stats[source]['chunk_types'].add(chunk_type)

            # 转换为列表格式
            doc_list = []
            for info in source_stats.values():
                doc_list.append({
                    'source': info['source'],
                    'chunk_count': info['chunk_count'],
                    'chunk_types': list(info['chunk_types'])
                })

            return doc_list
        except Exception as e:
            print(f"列出文档失败：{e}")
            return []

    def get_chunks_by_source(self, source: str) -> List[dict]:
        """获取指定来源的所有文档块内容

        Args:
            source: 来源文件名

        Returns:
            文档块列表 [{chunk_id, content, chunk_type, ...}, ...]
        """
        try:
            results = self._vectorstore.get(
                where={"source": source},
                include=['documents', 'metadatas']
            )
            if not results or not results['ids']:
                return []

            chunks = []
            for i, (doc_id, content, metadata) in enumerate(
                zip(results['ids'], results['documents'], results['metadatas'])
            ):
                chunks.append({
                    'id': doc_id,
                    'chunk_id': i + 1,
                    'content': content,
                    'chunk_type': metadata.get('chunk_type', 'unknown'),
                    'parent_id': metadata.get('parent_id'),
                })
            return chunks
        except Exception as e:
            print(f"获取文档块失败：{e}")
            return []

    def get_all_chunks(self) -> List[dict]:
        """获取向量库中所有文档块

        Returns:
            所有文档块列表
        """
        try:
            results = self._vectorstore.get(
                include=['documents', 'metadatas']
            )
            if not results or not results['ids']:
                return []

            chunks = []
            for i, (doc_id, content, metadata) in enumerate(
                zip(results['ids'], results['documents'], results['metadatas'])
            ):
                chunks.append({
                    'id': doc_id,
                    'chunk_id': i + 1,
                    'content': content,
                    'source': metadata.get('source', '未知'),
                    'chunk_type': metadata.get('chunk_type', 'unknown'),
                })
            return chunks
        except Exception as e:
            print(f"获取文档块失败：{e}")
            return []

    def get_all_documents(self) -> List[Document]:
        """获取向量库中所有文档块（LangChain Document 对象）

        供 BM25 等需要全量语料的检索方式建索引使用。

        Returns:
            所有文档块组成的 Document 列表
        """
        try:
            results = self._vectorstore.get(
                include=['documents', 'metadatas']
            )
            if not results or not results['ids']:
                return []

            docs = []
            for doc_id, content, metadata in zip(
                results['ids'], results['documents'], results['metadatas']
            ):
                meta = dict(metadata or {})
                meta['id'] = doc_id
                docs.append(Document(page_content=content, metadata=meta))
            return docs
        except Exception as e:
            print(f"获取全部文档失败：{e}")
            return []

    def count(self) -> int:
        """返回向量库中的文档块总数（用于判断语料是否变化）"""
        try:
            results = self._vectorstore.get()
            return len(results['ids']) if results and results['ids'] else 0
        except Exception:
            return 0

    def get_stats(self) -> dict:
        """获取向量库统计信息

        Returns:
            统计信息 {total_chunks, total_documents, ...}
        """
        try:
            results = self._vectorstore.get(include=['metadatas'])
            if not results or not results['ids']:
                return {
                    'total_chunks': 0,
                    'total_documents': 0,
                    'sources': []
                }

            sources = set()
            for metadata in results['metadatas']:
                sources.add(metadata.get('source', 'unknown'))

            return {
                'total_chunks': len(results['ids']),
                'total_documents': len(sources),
                'sources': list(sources)
            }
        except Exception as e:
            return {
                'total_chunks': 0,
                'total_documents': 0,
                'sources': [],
                'error': str(e)
            }

    def clear_all(self) -> bool:
        """清空向量库中所有文档

        Returns:
            是否清空成功
        """
        try:
            # 获取所有 ID 并删除
            results = self._vectorstore.get()
            if results and results['ids']:
                self._vectorstore.delete(ids=results['ids'])
            return True
        except Exception as e:
            print(f"清空向量库失败：{e}")
            return False

    @classmethod
    def reset_instance(cls):
        """重置单例（用于测试或重新加载配置）"""
        cls._instance = None
        cls._vectorstore = None
        cls._config = None
