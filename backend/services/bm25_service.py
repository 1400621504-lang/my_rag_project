"""BM25 关键词检索服务 - 与向量检索互补的第二路召回

为什么需要它：
向量检索（bi-encoder）擅长"语义相似"，但对精确关键词不敏感——
比如查错误码 "ORA-00942"、型号、专有名词，向量可能召不回来。
BM25 是经典的词频统计检索，天生擅长精确关键词匹配。

中文注意：
rank_bm25 默认按空格切词，中文没有空格，所以这里先用 jieba 分词，
再喂给 BM25。

语料来源：
直接从 ChromaDB 向量库读取全部 chunk（单一数据源，避免两份存储不同步）。
检测到库内块数变化时自动重建索引。
"""
from pathlib import Path
from typing import List
import threading

import yaml


class BM25Service:
    """BM25 检索服务 - 单例 + 语料懒构建

    首次检索时从向量库拉全量语料建索引；之后每次检索前比对块数，
    变了就重建，保证与向量库一致。
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config_path: str = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: str = None):
        if getattr(self, '_initialized', False):
            return

        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

        # 分词缓存（同一句不重复分词）
        self._bm25 = None
        self._docs: List = []      # 与语料对齐的 Document 列表
        self._corpus_size = -1     # 建索引时的块数，用于判断是否需重建
        self._initialized = True

    def _tokenize(self, text: str) -> List[str]:
        """jieba 分词，过滤空白 token"""
        import jieba
        return [w for w in jieba.lcut(text) if w.strip()]

    def _ensure_index(self):
        """确保 BM25 索引与向量库同步（块数变化则重建）"""
        from backend.services.vectorstore_service import VectorstoreService

        vs = VectorstoreService()
        current = vs.count()

        if self._bm25 is not None and current == self._corpus_size:
            return  # 语料没变，复用索引

        docs = vs.get_all_documents()
        self._docs = docs
        self._corpus_size = current

        if not docs:
            self._bm25 = None
            return

        from rank_bm25 import BM25Okapi
        corpus_tokens = [self._tokenize(d.page_content) for d in docs]
        self._bm25 = BM25Okapi(corpus_tokens)

    def search(self, query: str, k: int = 10) -> List:
        """BM25 关键词检索

        Args:
            query: 查询文本
            k: 返回数量

        Returns:
            相关性最高的 Document 列表（metadata 写入 bm25_score）
        """
        self._ensure_index()
        if self._bm25 is None:
            return []

        query_tokens = self._tokenize(query)
        # 全量打分，取 top_k
        scores = self._bm25.get_scores(query_tokens)

        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed[:k]:
            if score <= 0:
                continue  # 零分说明无任何词命中，丢弃
            doc = self._docs[idx]
            doc.metadata['bm25_score'] = float(score)
            results.append(doc)
        return results


if __name__ == "__main__":
    # 自测：需要向量库里已有数据
    svc = BM25Service()
    for q in ["Python 为什么慢", "ORA-00942"]:
        docs = svc.search(q, k=3)
        print(f"\n查询：{q}")
        for d in docs:
            print(f"  bm25={d.metadata['bm25_score']:.2f} {d.page_content[:36]}")
