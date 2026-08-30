"""检索评测指标：Hit@K / MRR@K / nDCG@K

只评检索层（retrieval-only），不调用 LLM。原因有三：
  1. 检索指标可复现，不受采样温度和模型版本影响；
  2. 跑得快，一晚上能扫完所有消融组合；
  3. RAG 的质量上限由检索决定，检索没召回，生成层再怎么调都救不回来。

生成层指标（忠实度、答案相关性）需要固定 LLM + 裁判模型，属于后续工作。
"""
import math
from typing import List, Set


def hit_at_k(ranked_ids: List[str], gold_ids: Set[str], k: int) -> float:
    """前 k 条里是否命中任意一个 gold 块：命中记 1，否则记 0"""
    return 1.0 if set(ranked_ids[:k]) & gold_ids else 0.0


def reciprocal_rank(ranked_ids: List[str], gold_ids: Set[str], k: int) -> float:
    """第一个 gold 块出现在第几位 → 1/rank；前 k 位内没出现则记 0

    比 Hit@K 更严格：同样命中，排第 1 和排第 5 的分数差 5 倍。
    """
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in gold_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: List[str], gold_ids: Set[str], k: int) -> float:
    """二元相关性（相关=1 / 不相关=0）下的归一化折损累积增益

    位置越靠后折损越大（log2 衰减），同时除以理想排序的 DCG 做归一化，
    使得不同 gold 数量的查询之间可比。
    """
    dcg = 0.0
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in gold_ids:
            dcg += 1.0 / math.log2(rank + 1)

    # 理想情况：gold 全部排在最前面
    ideal_n = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg > 0 else 0.0
