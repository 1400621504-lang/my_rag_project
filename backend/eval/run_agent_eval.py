"""Agent 端到端评测：固定管线 vs Agentic RAG

跑法：
    /opt/anaconda3/bin/python -m backend.eval.run_agent_eval            # 两种引擎都跑
    /opt/anaconda3/bin/python -m backend.eval.run_agent_eval --engine agent
    /opt/anaconda3/bin/python -m backend.eval.run_agent_eval --json /tmp/agent_eval.json

判分只看 must_contain 是否全部出现在答案里（事实级判定，不判文采），
另外单独统计 Agent 的检索次数、平均延迟、以及"没查库就作答/回退固定管线"的比例 ——
后者是 Agent 最容易骗人的地方，必须量出来。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEV_SET = Path(__file__).parent / "agent_dev.yaml"


def load_questions() -> list:
    data = yaml.safe_load(DEV_SET.read_text(encoding="utf-8"))
    return data["questions"]


def grade(answer: str, must_contain: list) -> tuple:
    """返回 (是否全中, 命中的关键词数, 总关键词数)"""
    hit = [kw for kw in must_contain if kw in answer]
    return len(hit) == len(must_contain), len(hit), len(must_contain)


def run_engine(engine: str, questions: list, fresh: bool = False) -> dict:
    if engine == "rag":
        from backend.chains.rag_chain import RAGChain
        rag = RAGChain()
        peer = rag
        ask = lambda q: (rag.invoke(q), {})
    else:
        from backend.chains.agent_chain import AgentRAGChain
        agent = AgentRAGChain()
        peer = agent
        ask = lambda q: (agent.ask(q)["answer"], {
            "search_calls": agent.last_result.get("search_calls", 0),
            "answer_source": agent.last_result.get("answer_source", ""),
        })

    records = []
    for q in questions:
        if fresh:
            # 逐题独立：对话历史会污染"要不要查库"这个决策 ——
            # 累积几轮后小模型干脆不再调工具，那测出来的不是能力而是懒惰程度
            peer.clear_memory()
        started = time.perf_counter()
        try:
            answer, extra = ask(q["question"])
            error = ""
        except Exception as exc:  # 单题失败不该终止整轮评测
            answer, extra, error = "", {}, f"{type(exc).__name__}: {exc}"
        latency = time.perf_counter() - started

        ok, hit, total = grade(answer, q["must_contain"])
        missing = [kw for kw in q["must_contain"] if kw not in answer]
        records.append(
            {
                "id": q["id"],
                "type": q["type"],
                "question": q["question"],
                "exact": ok,
                "kw_recall": round(hit / total, 3) if total else 0.0,
                "missing": missing,
                "latency_s": round(latency, 2),
                "answer_head": answer[:160].replace("\n", " "),
                "error": error,
                **extra,
            }
        )
        flag = "PASS" if ok else "fail"
        src = extra.get("answer_source", "")
        print(f"[{engine}] {q['id']} {flag:4s} 关键词 {hit}/{total} {latency:5.1f}s "
              f"查{extra.get('search_calls', '-') if 'search_calls' in extra else '-'}次 {src}"
              + (f"  缺: {missing}" if missing else ""))
    return {"engine": engine, "records": records}


def summarize(result: dict) -> dict:
    recs = result["records"]
    by_type = defaultdict(list)
    for r in recs:
        by_type[r["type"]].append(r)

    summary = {
        "engine": result["engine"],
        "n": len(recs),
        "exact_rate": round(sum(r["exact"] for r in recs) / len(recs), 3),
        "kw_coverage": round(sum(r["kw_recall"] for r in recs) / len(recs), 3),
        "avg_latency_s": round(statistics.mean(r["latency_s"] for r in recs), 2),
        "by_type": {
            t: {
                "n": len(v),
                "exact_rate": round(sum(x["exact"] for x in v) / len(v), 3),
                "kw_coverage": round(sum(x["kw_recall"] for x in v) / len(v), 3),
            }
            for t, v in sorted(by_type.items())
        },
    }
    if result["engine"] == "agent":
        calls = [r.get("search_calls", 0) for r in recs]
        summary["avg_search_calls"] = round(statistics.mean(calls), 2)
        summary["fallback_rate"] = round(
            sum(1 for r in recs if str(r.get("answer_source", "")).startswith("fallback")) / len(recs), 3
        )
        summary["no_search_rate"] = round(
            sum(1 for r in recs if r.get("search_calls", 0) == 0) / len(recs), 3
        )
    return summary


def print_compare(summaries: list):
    if len(summaries) < 2:
        return
    print("\n" + "=" * 72)
    print(f"{'引擎':<8}{'全对率':>8}{'关键词覆盖':>11}{'平均延迟':>10}{'分类别全对率':>26}")
    for s in summaries:
        byt = " ".join(f"{t}:{v['exact_rate']:.2f}" for t, v in s["by_type"].items())
        print(f"{s['engine']:<10}{s['exact_rate']:>7.1%}{s['kw_coverage']:>11.1%}"
              f"{s['avg_latency_s']:>8.1f}s  {byt}")
        if "avg_search_calls" in s:
            print(f"{'':<10}平均检索 {s['avg_search_calls']} 次 | 未查库率 {s['no_search_rate']:.1%} "
                  f"| 兜底回退率 {s['fallback_rate']:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["rag", "agent", "both"], default="both")
    parser.add_argument("--json", help="把逐题结果写到指定路径")
    parser.add_argument("--fresh", action="store_true",
                        help="每题前清空对话历史，测单轮独立表现（Agent 是否查库不受上一轮影响）")
    args = parser.parse_args()

    questions = load_questions()
    engines = ["rag", "agent"] if args.engine == "both" else [args.engine]

    summaries = []
    all_records = {}
    for engine in engines:
        result = run_engine(engine, questions, fresh=args.fresh)
        summaries.append(summarize(result))
        all_records[engine] = result["records"]

    print_compare(summaries)
    for s in summaries:
        print(f"\n--- {s['engine']} 汇总 ---")
        print(json.dumps(s, ensure_ascii=False, indent=2))

    if args.json:
        Path(args.json).write_text(
            json.dumps({"summaries": summaries, "records": all_records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n结果已写入 {args.json}")


if __name__ == "__main__":
    main()
