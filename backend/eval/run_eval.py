"""检索链路消融评测

对同一份问题集，固定召回候选数，比较五种检索配置的质量与延迟差异：

    A  仅向量（similarity）
    B  向量 + 精排
    C  MMR  + 精排
    D  双路（向量 + BM25）+ RRF
    E  双路 + RRF + 精排      ← v1.1 完整链路

为什么要固定召回数：主链路里 Rerank 开与关会改变 retriever 的 recall_k
（见 rag_chain._init_retriever），直接拿 RAGChain 对比会把"召回更多"和
"精排更准"两个收益混在一起。这里统一召回 20 条再做后处理，差值才归因得干净。

跑的是检索层，不调 LLM，所以结果稳定、可复现、几秒钟一轮。

用法：
    python -m backend.eval.run_eval                      # 全量对比
    python -m backend.eval.run_eval --check-gold         # 只校验标注与语料是否对得上
    python -m backend.eval.run_eval --json result.json   # 另存机器可读结果
"""
import argparse
import json
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Set

import yaml

# 允许直接以脚本方式运行（python backend/eval/run_eval.py）
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.eval.metrics import hit_at_k, ndcg_at_k, reciprocal_rank
from backend.retriever.fusion import _doc_key, reciprocal_rank_fusion
from backend.services.bm25_service import BM25Service
from backend.services.rerank_service import RerankService
from backend.services.vectorstore_service import VectorstoreService

# 五种待比较的配置
MODES = [
    ("A 仅向量",       dict(lane="vec",    mmr=False, rerank=False)),
    ("B 向量+精排",     dict(lane="vec",    mmr=False, rerank=True)),
    ("C MMR+精排",     dict(lane="vec",    mmr=True,  rerank=True)),
    ("D 双路+RRF",     dict(lane="hybrid", mmr=False, rerank=False)),
    ("E 双路+RRF+精排", dict(lane="hybrid", mmr=False, rerank=True)),
]

CUTOFFS = (1, 3, 5)


def display_width(text: str) -> int:
    """按东亚字符宽度计算显示列宽，保证中文表头对齐"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def pad(text: str, width: int) -> str:
    """右侧补空格到指定显示宽度"""
    return text + " " * max(0, width - display_width(text))


def load_corpus(vs: VectorstoreService) -> Dict[str, str]:
    """读出向量库全量分块，返回 {内容哈希: 正文}

    用内容哈希做标识（与 RRF 融合同一个键），这样评测结果不受
    Chroma 重建 id 变化的影响。
    """
    corpus = {}
    for doc in vs.get_all_documents():
        corpus[_doc_key(doc)] = doc.page_content
    return corpus


def resolve_gold(corpus: Dict[str, str], markers: List[str]) -> Set[str]:
    """按特征串反查 gold 块的内容哈希，命中任一特征串即算相关"""
    return {key for key, text in corpus.items() if any(m in text for m in markers)}


def build_ranking(question: str, vs: VectorstoreService, bm25: BM25Service,
                  reranker: RerankService, cfg: dict,
                  recall: int, top_k: int, rrf_k: int,
                  lambda_mult: float) -> List[str]:
    """按给定配置产出一个排序结果（返回内容哈希列表）"""
    search_type = "mmr" if cfg["mmr"] else "similarity"
    kwargs = {}
    if cfg["mmr"]:
        kwargs = dict(fetch_k=max(recall, 20), lambda_mult=lambda_mult)

    vector_docs = vs.search(question, search_type=search_type, k=recall, **kwargs)

    if cfg["lane"] == "hybrid":
        bm25_docs = bm25.search(question, k=recall)
        docs = reciprocal_rank_fusion([vector_docs, bm25_docs], k=rrf_k, top_n=recall)
    else:
        docs = vector_docs

    if cfg["rerank"] and docs:
        docs = reranker.rerank(question, docs, top_n=top_k)

    return [_doc_key(d) for d in docs[:top_k]]


def aggregate(rows: List[dict], key: str) -> Dict[str, dict]:
    """按某个维度（overall / category）聚合各指标的均值"""
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)

    summary = {}
    for name, items in groups.items():
        summary[name] = {
            metric: round(mean(it[metric] for it in items), 4)
            for metric in items[0] if isinstance(metric, str) and metric.startswith(("hit", "mrr", "ndcg"))
        }
        summary[name]["latency_ms"] = round(mean(it["latency_ms"] for it in items), 1)
        summary[name]["n"] = len(items)
    return summary


def print_table(title: str, summary: Dict[str, dict], order: List[str]) -> None:
    """打印聚合结果表"""
    cols = [f"hit@{c}" for c in CUTOFFS] + ["mrr@5", "ndcg@5"]
    widths = [max(display_width(title), 12)] + [max(display_width("Hit@5"), 8) for _ in cols] + [10]
    print()
    print(pad(title, widths[0]) + "".join(pad(c, w) for c, w in zip(cols + ["延迟ms"], widths[1:])))
    print("-" * sum(widths))
    for name in order:
        if name not in summary:
            continue
        s = summary[name]
        cells = [f"{s[c]:.3f}" for c in cols] + [f"{s['latency_ms']:.0f}"]
        print(pad(name, widths[0]) + "".join(pad(c, w) for c, w in zip(cells, widths[1:])))


def main() -> int:
    parser = argparse.ArgumentParser(description="检索链路消融评测")
    parser.add_argument("--dataset", default=str(Path(__file__).parent / "qa_dev.yaml"))
    parser.add_argument("--recall", type=int, default=20, help="每种配置的召回候选数")
    parser.add_argument("--top-k", type=int, default=5, help="最终返回条数，与线上配置保持一致")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF 平滑常数")
    parser.add_argument("--lambda-mult", type=float, default=0.7, help="MMR 多样性权重")
    parser.add_argument("--check-gold", action="store_true", help="只校验标注，不跑检索")
    parser.add_argument("--json", dest="json_out", default=None, help="结果另存为 JSON")
    args = parser.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = yaml.safe_load(f)["queries"]

    vs = VectorstoreService()
    corpus = load_corpus(vs)
    print(f"语料：{len(corpus)} 个分块 | 问题集：{len(dataset)} 条 | 召回 {args.recall} → 取 top {args.top_k}")

    # 第一步：确认每条问题都能定位到 gold 块，标注失效时及早报出来
    labeled = []
    unmatched = []
    for item in dataset:
        gold = resolve_gold(corpus, item["gold_markers"])
        if not gold:
            unmatched.append((item["id"], item["question"], item["gold_markers"]))
            continue
        labeled.append({**item, "gold": gold})

    if unmatched:
        print(f"\n[标注失效] {len(unmatched)} 条问题在语料里找不到 gold 块，已跳过：")
        for qid, q, markers in unmatched:
            print(f"  {qid}: {q}  特征串={markers}")

    if not labeled:
        print("没有任何可用问题，请先上传文档再跑评测。")
        return 1

    if args.check_gold:
        print(f"\n标注校验通过 {len(labeled)}/{len(dataset)} 条。")
        return 0

    bm25 = BM25Service()
    reranker = RerankService()

    rows: List[dict] = []
    for label, cfg in MODES:
        t0 = time.perf_counter()
        for item in labeled:
            start = time.perf_counter()
            ranked = build_ranking(item["question"], vs, bm25, reranker, cfg,
                                   args.recall, args.top_k, args.rrf_k, args.lambda_mult)
            elapsed = (time.perf_counter() - start) * 1000

            row = {
                "id": item["id"],
                "category": item["category"],
                "mode": label,
                "latency_ms": elapsed,
            }
            for cut in CUTOFFS:
                row[f"hit@{cut}"] = hit_at_k(ranked, item["gold"], cut)
            row["mrr@5"] = reciprocal_rank(ranked, item["gold"], args.top_k)
            row["ndcg@5"] = ndcg_at_k(ranked, item["gold"], args.top_k)
            rows.append(row)
        print(f"[{label}] 完成，用时 {time.perf_counter() - t0:.1f}s")

    order = [label for label, _ in MODES]
    print_table("配置", aggregate(rows, "mode"), order)

    categories = sorted({r["category"] for r in rows})
    by_cat = aggregate(rows, "category")
    for cat in categories:
        subset = [r for r in rows if r["category"] == cat]
        print_table(f"[{cat}] n={by_cat[cat]['n']}", aggregate(subset, "mode"), order)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"summary": aggregate(rows, "mode"), "detail": rows},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n明细已存：{args.json_out}")

    print("\n说明：语料只有十几个分块时 Hit@5 容易虚高，重点看 MRR 和 nDCG；"
          "想把结论写进简历，语料至少扩到 100+ 分块。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
