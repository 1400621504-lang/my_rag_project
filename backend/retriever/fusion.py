"""多路检索结果融合 - RRF（Reciprocal Rank Fusion）

为什么用 RRF：
向量召回的相似度分数和 BM25 的词频分数量纲完全不同，没法直接相加。
RRF 只看每一路里的"排名"，不看原始分数，天然规避了不可比问题。

公式：
    RRF_score(d) = Σ_i  1 / (k + rank_i(d))
    rank_i(d) 是文档 d 在第 i 路结果里的排名（从 1 开始），
    k 是平滑常数（经验值 60），避免排名靠前的项过度主导。

去重：
同一篇文档可能两路都召回到，用 metadata['id']（Chroma 主键）作为唯一标识合并。
"""
from typing import List, Dict


def reciprocal_rank_fusion(
    result_lists: List[List],
    k: int = 60,
    top_n: int = 20,
) -> List:
    """融合多路检索结果

    Args:
        result_lists: 每路检索返回的 Document 列表（已按相关性排序）
        k: RRF 平滑常数，默认 60
        top_n: 融合后保留的数量

    Returns:
        融合排序后的 Document 列表，metadata['rrf_score'] 写入融合分数
    """
    # doc_key -> [doc, 累计分数]
    fused: Dict[str, list] = {}

    for docs in result_lists:
        for rank, doc in enumerate(docs, start=1):
            key = _doc_key(doc)
            contribution = 1.0 / (k + rank)
            if key in fused:
                fused[key][1] += contribution
            else:
                fused[key] = [doc, contribution]

    ranked = sorted(fused.values(), key=lambda x: x[1], reverse=True)

    result = []
    for doc, score in ranked[:top_n]:
        doc.metadata['rrf_score'] = round(float(score), 5)
        result.append(doc)
    return result


def _doc_key(doc) -> str:
    """文档唯一标识：用内容哈希

    不能用 Chroma 的 id——向量检索器返回的 Document 不带 store id，
    而 BM25 路的带，同一块两路边 key 不同会导致去重失效、结果重复。
    内容哈希对同一文本稳定，是唯一在两路都成立的标识。
    """
    import hashlib
    norm = ' '.join(doc.page_content.split())  # 归一化空白，避免换行差异
    return hashlib.md5(norm.encode('utf-8')).hexdigest()
