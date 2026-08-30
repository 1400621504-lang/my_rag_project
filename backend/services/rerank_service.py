"""Rerank 重排序服务 - 对检索候选做精排

原理：
向量检索按"语义相似度"召回，但相似不等于相关。Rerank 用一个专门的
交叉编码模型（cross-encoder）对 query 和每个候选文档逐对打分，重新排序，
选出真正最相关的 top_n 交给 LLM。

用法：
    service = RerankService()
    ranked = service.rerank(query, docs, top_n=5)

模型（bge-reranker-base）需预先下载到 model_path 指向的本地目录。
Ollama 不支持 reranker，因此这里用 FlagEmbedding 直接加载 HuggingFace 模型。
"""
from pathlib import Path
from typing import List
import threading

import yaml


class RerankService:
    """Rerank 服务 - 单例 + 模型懒加载

    懒加载：只有在第一次真正需要 rerank 时才加载模型，
    这样未开启 Rerank 时不占用内存、不触发模型加载。
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

        rerank_cfg = self._config.get('rerank', {})
        self.enabled = rerank_cfg.get('enabled', False)
        self._type = rerank_cfg.get('type', 'local')
        self._local_cfg = rerank_cfg.get('local', {})
        self._candidate_k = self._local_cfg.get('candidate_k', 20)

        # 模型懒加载，首次 rerank 时填充
        self._model = None
        self._initialized = True

    def _resolve_model_path(self) -> str:
        """把配置里的相对路径解析成项目根目录下的绝对路径"""
        raw = self._local_cfg.get('model_path', './models/rerank/bge-reranker-base')
        raw = raw.lstrip('./')
        return str(Path(__file__).parent.parent.parent / raw)

    def _ensure_model(self):
        """首次调用时加载 reranker 模型"""
        if self._model is not None:
            return

        from FlagEmbedding import FlagReranker

        model_path = self._resolve_model_path()
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Rerank 模型不存在：{model_path}\n"
                f"请先下载 bge-reranker-base 到该目录，或在配置中关闭 rerank.enabled。"
            )

        # CPU 推理，fp32（Mac CPU 上 fp16 无加速反而可能报错）
        self._model = FlagReranker(model_path, use_fp16=False)

    @property
    def candidate_k(self) -> int:
        """rerank 前建议的召回数量"""
        return self._candidate_k

    @candidate_k.setter
    def candidate_k(self, value: int):
        """允许前端实时改召回候选数

        原来只有 getter，侧边栏那个滑块一赋值就抛
        AttributeError: property 'candidate_k' has no setter。
        """
        self._candidate_k = max(1, int(value))

    def rerank(self, query: str, docs: List, top_n: int = 5) -> List:
        """对候选文档做精排，返回相关性最高的 top_n 个

        Args:
            query: 用户查询
            docs: 检索召回的 Document 列表
            top_n: 精排后保留的数量

        Returns:
            重排序后的 Document 列表（前 top_n 个）。每个文档的
            metadata['rerank_score'] 会写入相关性分数。
        """
        if not docs:
            return []

        self._ensure_model()

        pairs = [[query, doc.page_content] for doc in docs]
        # 批量打分，比逐条快
        scores = self._model.compute_score(pairs, batch_size=8)

        # compute_score 单条时返回标量，多条返回 list
        if not isinstance(scores, list):
            scores = [scores]

        scored = list(zip(docs, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        result = []
        for doc, score in scored[:top_n]:
            doc.metadata['rerank_score'] = float(score)
            result.append(doc)
        return result


if __name__ == "__main__":
    # 简易自测
    from langchain_core.documents import Document

    svc = RerankService()
    q = "Python 装饰器怎么用"
    docs = [
        Document(page_content="装饰器是接收函数返回函数的函数，用 @ 语法应用",
                 metadata={"source": "a"}),
        Document(page_content="Java 也有装饰器模式，属于结构型设计模式",
                 metadata={"source": "b"}),
        Document(page_content="Python 的 GIL 限制多线程并行", metadata={"source": "c"}),
    ]
    ranked = svc.rerank(q, docs, top_n=3)
    for d in ranked:
        print(f"{d.metadata['rerank_score']:.4f}  {d.metadata['source']}  {d.page_content[:20]}")
