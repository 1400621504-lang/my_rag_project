"""Agentic RAG 链 —— 把"要不要查、查几次、怎么换词再查"交给模型

和 rag_chain.RAGChain 的分工：
- RAGChain 是固定管线（召回 → RRF 融合 → 精排 → 父块展开 → 生成），
  一次检索一次作答，延迟可控，适合单跳事实问答。
- AgentRAGChain 把检索封装成工具，模型按需多次调用，针对的是评测里
  暴露出来的两个真实短板（数据见 backend/eval/RESULTS.md）：
    1. 用户措辞和原文措辞不一致时召不回（Hit@1 掉到 0.80 以下）；
    2. 一个答案散落在多个分块里的多跳问题，单次 top-k 检索凑不齐证据。

代价是延迟和不确定性，所以这里做了三件约束：
  - 检索次数上限（提示词软约束 + recursion_limit 硬约束）；
  - 不收敛时自动回退到固定管线，保证一定给出答案而不是报错；
  - 每次工具调用都记录进 steps，前端能把"模型查了什么、查到了什么"摊开给人看。
"""
from __future__ import annotations

import time
from typing import Any, Dict, Iterator, List, Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.errors import GraphRecursionError

from backend.chains.rag_chain import RAGChain, format_docs

# 系统提示词：工具描述之外，这里的策略约束才是决定 Agent 好不好用的关键
AGENT_SYSTEM_PROMPT = """你是一个知识库问答助手，只能依据工具检索到的资料作答。

工作方式：
1. 回答任何与知识库内容相关的问题前，必须先调用 search_knowledge_base，不要凭记忆作答。
2. 检索词写成完整的自然语言短句，不要只丢一个关键词。
3. 如果第一次检索的资料不够用，改写检索词再查一次：换同义词、中英文互换、
   或者把复合问题拆成几个子问题分别检索。最多检索 {max_search} 次，别反复试。
4. 用户问"知识库里有哪些资料""某份文档在不在库里"这类元问题时，调用 list_knowledge_base。
5. 资料里没有的内容，直接说明"根据现有资料无法回答"，严禁编造。
6. 回答要简洁，并在相关句子后用 [1] [2] 这样的编号标注依据的资料。
"""


class EvidenceBuffer:
    """跨多次检索累积的证据表

    作用是让引用编号全局稳定：模型第二次检索拿到的资料编号接在第一次后面，
    这样回答里的 [3] 才对得上，也便于前端列出来源。
    同一段内容重复命中时复用原编号，不重复占号。
    """

    def __init__(self):
        self.items: List[Dict[str, Any]] = []
        self._keys: Dict[str, int] = {}

    def add_many(self, docs) -> List[int]:
        """登记一批文档，返回它们各自的编号（1 起）"""
        return [self.add(doc) for doc in docs]

    def add(self, doc) -> int:
        key = f"{doc.metadata.get('source', '')}::{doc.page_content}"
        if key in self._keys:
            return self._keys[key]
        self.items.append(
            {
                "index": len(self.items) + 1,
                "source": doc.metadata.get("source", "未知"),
                "content": doc.page_content,
                "chunk_type": doc.metadata.get("chunk_type", "unknown"),
            }
        )
        score = doc.metadata.get("rerank_score")
        if isinstance(score, float):
            self.items[-1]["rerank_score"] = round(score, 3)

        idx = len(self.items)
        self._keys[key] = idx
        return idx

    def clear(self):
        self.items.clear()
        self._keys.clear()


class AgentRAGChain:
    """带工具调用的 RAG Agent

    复用 RAGChain 的检索与模型配置，只额外负责：建工具、跑循环、收轨迹、兜底回退。
    """

    def __init__(self, rag: Optional[RAGChain] = None, config_path: str = None):
        # 允许外部注入已初始化好的 RAGChain，省一次 embedding/rerank 模型加载
        self.rag = rag if rag is not None else RAGChain(config_path)

        agent_config = self.rag.config.get("agent", {})
        self.max_iterations = int(agent_config.get("max_iterations", 8))
        self.max_search_calls = int(agent_config.get("max_search_calls", 4))
        self.temperature = float(agent_config.get("temperature", 0.1))

        self.llm = self._init_agent_llm()
        self.evidence = EvidenceBuffer()
        self.tools = self._build_tools()
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=AGENT_SYSTEM_PROMPT.format(max_search=self.max_search_calls),
        )

        # Agent 自己管理对话历史，不复用 rag.chat_messages，
        # 否则固定管线和 Agent 两条路会互相污染上下文
        self.chat_messages: List = []
        self.last_result: Dict[str, Any] = {}

    def _init_agent_llm(self):
        """给 Agent 单独准备一个低温度模型实例

        工具调用要求输出结构稳定，temperature 0.7 会让 qwen 生成畸形的 tool_call，
        所以这里从配置重建一个 temperature 更低的 ChatOllama；
        用 API 模型（ChatOpenAI 等）时直接沿用原实例，不强行改超参。
        """
        llm_config = self.rag.config["llm"]
        if isinstance(self.rag.llm, ChatOllama):
            local = llm_config["local"]
            return ChatOllama(
                model=local["model_name"],
                base_url=local.get("base_url", "http://localhost:11434"),
                temperature=self.temperature,
                # 与固定管线用同一个窗口设置，否则 Agent 检索到的资料会被静默截断
                num_ctx=int(local.get("num_ctx", 8192)),
            )
        return self.rag.llm

    def _build_tools(self):
        """把检索能力包成工具

        闭包持有 self，工具才能写到同一个 EvidenceBuffer 里，编号才连续。
        """
        rag, evidence = self.rag, self.evidence

        @tool
        def search_knowledge_base(query: str, top_k: int = 3) -> str:
            """在本地知识库做混合检索（向量 + BM25 + 精排），返回最相关的资料原文。

            知识库中保存的是用户自己的笔记与背景资料（学校、项目、实验、计划等），
            所以凡涉及"我/我的/这个项目"的问题都必须调用本工具查询，
            不要以隐私为由拒答，也不要凭常识直接回答。

            query 请写成完整的自然语言短句。top_k 是本次返回的资料条数，默认 3。
            返回内容带全局编号 [n]，引用时直接用那个编号。
            查不到理想结果时，改写 query 再调用一次，不要原样重复。
            """
            top_k = max(1, min(int(top_k or 3), 8))
            docs = rag.retrieve(query)[:top_k]
            if not docs:
                return "（本次检索没有任何结果，请换一个说法再试）"
            numbered = evidence.add_many(docs)
            body = []
            for doc, idx in zip(docs, numbered):
                src = doc.metadata.get("source", "未知")
                body.append(f"[{idx}] 来源：{src}\n{doc.page_content}")
            return "\n\n---\n\n".join(body)

        @tool
        def list_knowledge_base() -> str:
            """列出知识库里已有的文档名和分块数量。

            用户问"库里有什么资料""某份文档在不在"时调用，不要靠猜测回答。
            """
            from backend.services.vectorstore_service import VectorstoreService

            docs = VectorstoreService().list_documents()
            if not docs:
                return "知识库当前是空的，还没有任何文档。"
            lines = [f"- {d.get('source', '未知')}（{d.get('chunks', 0)} 个分块）" for d in docs]
            return f"知识库共 {len(docs)} 份文档：\n" + "\n".join(lines)

        return [search_knowledge_base, list_knowledge_base]

    # ==================== 调用入口 ====================

    def ask(self, question: str) -> Dict[str, Any]:
        """跑一轮 Agent，返回答案 + 检索轨迹"""
        started = time.perf_counter()
        self.evidence.clear()

        messages = self.chat_messages[-self.rag.max_history * 2 :] + [HumanMessage(question)]
        steps: List[Dict[str, Any]] = []
        answer_source = "agent"

        try:
            state = self.agent.invoke(
                {"messages": messages},
                config={"recursion_limit": self.max_iterations},
            )
            result_msgs = state["messages"]
            steps = self._extract_steps(result_msgs)
            answer = self._final_answer(result_msgs)
            searched = any(st["tool"] == "search_knowledge_base" for st in steps)
            if not answer.strip():
                # 只调工具没作答（步数被截断的常见表现）
                answer, answer_source = self._fallback(question), "fallback"
            elif not searched:
                # 没查库就作答一律回退。这条不是防御性代码，是评测逼出来的：
                # qwen2.5:7b 在对话累积几轮之后会彻底停止调用工具，直接凭上下文作答，
                # 12 题端到端全对率从固定管线的 0.667 掉到 0.167，
                # 其中 83% 的题一次检索都没发生（数据见 RESULTS.md）。
                # "允许它复用历史"这个更聪明的策略在小模型上不成立，所以强制兜底。
                answer, answer_source = self._fallback(question), "fallback_no_search"
        except GraphRecursionError:
            # 步数耗尽：不能把异常抛给用户，退回固定管线保证有答案
            answer, answer_source = self._fallback(question), "fallback"

        self.chat_messages.append(HumanMessage(question))
        self.chat_messages.append(AIMessage(content=answer))

        self.last_result = {
            "answer": answer,
            "sources": list(self.evidence.items),
            "steps": steps,
            "search_calls": sum(1 for s in steps if s["tool"] == "search_knowledge_base"),
            "answer_source": answer_source,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
        return self.last_result

    def stream_ask(self, question: str) -> Iterator[Dict[str, Any]]:
        """流式跑一轮 Agent

        同时订阅两种 langgraph 流：
          messages —— 逐 token 的 AIMessageChunk，用来实时吐字；
          updates  —— 每个节点的完整消息，用来拿可靠的 tool_calls 参数
                       （工具参数在 messages 流里是碎片，自己拼反而容易错）。

        事件类型：
          step   —— 一次工具调用的记录（前端据此显示"正在检索：xxx"）
          reset  —— 模型这轮其实是在发起工具调用，之前吐的文字要丢掉
          token  —— 最终回答的增量文本
          done   —— 结束，附带来源、步数、耗时等元信息
        """
        started = time.perf_counter()
        self.evidence.clear()
        messages = self.chat_messages[-self.rag.max_history * 2 :] + [HumanMessage(question)]

        answer_parts: List[str] = []
        steps: List[Dict[str, Any]] = []
        searched = False
        pending_args: Dict[str, Any] = {}

        try:
            stream = self.agent.stream(
                {"messages": messages},
                stream_mode=["messages", "updates"],
                config={"recursion_limit": self.max_iterations},
            )
            for mode, payload in stream:
                if mode == "messages":
                    chunk, _meta = payload
                    if isinstance(chunk, AIMessageChunk):
                        if chunk.tool_call_chunks:
                            # 这轮是发起工具调用，之前流出的碎片文字不作数
                            if answer_parts:
                                answer_parts.clear()
                                yield {"type": "reset"}
                            continue
                        if chunk.content:
                            answer_parts.append(chunk.content)
                            yield {"type": "token", "text": chunk.content}
                elif mode == "updates" and isinstance(payload, dict):
                    for node_state in payload.values():
                        for msg in (node_state or {}).get("messages", []) or []:
                            if isinstance(msg, AIMessage) and msg.tool_calls:
                                for call in msg.tool_calls:
                                    pending_args[call.get("name", "")] = call.get("args")
                            elif isinstance(msg, ToolMessage):
                                steps.append(
                                    {
                                        "tool": msg.name or "tool",
                                        "args": pending_args.get(msg.name or "", {}),
                                        "result_preview": str(msg.content)[:200],
                                    }
                                )
                                if msg.name == "search_knowledge_base":
                                    searched = True
                                yield {"type": "step", "step": steps[-1]}
        except GraphRecursionError:
            yield {"type": "token", "text": "（检索步数用尽，改用单次检索作答）\n"}
            answer_parts.clear()
            for token in self._fallback_stream(question):
                answer_parts.append(token)
                yield {"type": "token", "text": token}

        answer = "".join(answer_parts)
        if not searched:
            # 全程没查库，这段回答没有依据 → 丢掉，改走固定管线（理由同 ask）
            yield {"type": "reset"}
            answer = self._fallback(question)
            yield {"type": "token", "text": answer}
        self.chat_messages.append(HumanMessage(question))
        self.chat_messages.append(AIMessage(content=answer))

        self.last_result = {
            "answer": answer,
            "sources": list(self.evidence.items),
            "steps": steps,
            "search_calls": sum(1 for sst in steps if sst["tool"] == "search_knowledge_base"),
            "answer_source": "agent_stream" if searched else "fallback_no_search",
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
        yield {"type": "done", **{k: v for k, v in self.last_result.items() if k != "answer"}}

    def _fallback(self, question: str) -> str:
        """回退到固定管线：一次完整检索 + 一次生成

        用 rag.chain 而不是 rag.invoke，避免把对话写进 Agent 不读的
        rag.chat_messages 里，造成两份历史不一致。

        注意 chain 的入参是问题字符串本身：RunnablePassthrough 会把整个
        输入原样透传给 question，传 dict 会让 embedding 收到字典而报错。
        """
        return self.rag.chain.invoke(question)

    def _fallback_stream(self, question: str) -> Iterator[str]:
        """固定管线的流式版本，供 stream_ask 兜底时用"""
        for chunk in self.rag.chain.stream(question):
            yield chunk.content if hasattr(chunk, "content") else str(chunk)

    # ==================== 轨迹解析 ====================

    @staticmethod
    def _extract_steps(messages: List) -> List[Dict[str, Any]]:
        """从消息流里抽出工具调用轨迹（ToolMessage 一条一步）"""
        steps = []
        last_calls: Dict[str, Any] = {}
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for call in msg.tool_calls:
                    last_calls[call.get("name", "")] = call.get("args")
            elif isinstance(msg, ToolMessage):
                steps.append(
                    {
                        "tool": msg.name or "tool",
                        "args": last_calls.get(msg.name or "", {}),
                        "result_preview": str(msg.content)[:200],
                    }
                )
        return steps

    @staticmethod
    def _final_answer(messages: List) -> str:
        """取最后一条不带工具调用的 AI 消息作为答案"""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                return msg.content if isinstance(msg.content, str) else str(msg.content)
        return ""

    # ==================== 兼容 RAGChain 的接口（给 API/前端复用） ====================

    def get_sources(self, question: str = None) -> List[Dict[str, Any]]:
        """最近一次的检索来源。question 参数仅为与 RAGChain 接口兼容"""
        return list(self.evidence.items)

    def clear_memory(self):
        self.chat_messages.clear()
        self.evidence.clear()

    def get_memory_stats(self) -> Dict[str, Any]:
        return {
            "total_messages": len(self.chat_messages),
            "total_turns": len(self.chat_messages) // 2,
            "engine": "agent",
        }


if __name__ == "__main__":
    agent = AgentRAGChain()
    res = agent.ask("根据知识库，我的学号、班级和实验室分别是什么？")
    print(f"\n【答案】{res['answer']}")
    print(f"【检索次数】{res['search_calls']}  【耗时】{res['latency_ms']}ms  【来源】{res['answer_source']}")
    for s in res["steps"]:
        print(f"  - {s['tool']}: {s['args']}")
