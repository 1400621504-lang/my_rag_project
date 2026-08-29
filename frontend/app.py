"""RAG 知识库问答平台 - DeepSeek 深色风格

启动：streamlit run frontend/app.py
"""
import streamlit as st
import streamlit.components.v1 as components
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ==================== 页面配置 ====================
st.set_page_config(page_title="RAG 知识库问答", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# 强制侧边栏展开
# ==================== DeepSeek 深色主题 ====================
st.markdown("""
<style>
    /* ===== 基础 ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Inter', 'PingFang SC',
                     'Microsoft YaHei', sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    /* 隐藏 Streamlit 默认元素 */
    #MainMenu, header, footer {visibility: hidden;}

    /* ===== 全局深色背景 ===== */
    .stApp {
        background: #0d0d0d;
    }

    /* ===== 侧边栏 ===== */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] ~ div {
        visibility: visible !important;
        opacity: 1 !important;
        transform: none !important;
        display: block !important;
    }
    [data-testid="stSidebar"] {
        background: #161616;
        border-right: 1px solid #2a2a2a;
        min-width: 340px !important;
        width: 340px !important;
        max-width: 340px !important;
        transform: translateX(0) !important;
    }
    [data-testid="stSidebar"] > div {
        padding: 20px 16px;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #e5e5e5;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-top: 18px;
        margin-bottom: 8px;
        text-transform: uppercase;
    }

    /* 侧边栏标题 */
    .sidebar-header {
        padding: 0 0 16px 0;
        margin-bottom: 16px;
        border-bottom: 1px solid #2a2a2a;
    }
    .sidebar-header h1 {
        font-size: 18px !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        margin: 0 !important;
        letter-spacing: -0.01em;
    }
    .sidebar-header p {
        color: #666 !important;
        font-size: 12px !important;
        margin: 4px 0 0 0 !important;
    }

    /* metric 卡片 */
    [data-testid="stMetric"] {
        background: #1e1e1e;
        padding: 12px 14px;
        border-radius: 10px;
        border: 1px solid #2a2a2a;
    }
    [data-testid="stMetricLabel"] {
        color: #666 !important;
        font-size: 11px !important;
        font-weight: 500 !important;
    }
    [data-testid="stMetricValue"] {
        color: #e5e5e5 !important;
        font-size: 22px !important;
        font-weight: 700 !important;
    }

    /* ===== 主区域 ===== */
    .main-area {
        max-width: 800px;
        margin: 0 auto;
        padding: 0 20px;
    }

    /* ===== 空状态 ===== */
    .empty-state {
        text-align: center;
        padding: 100px 0 60px;
    }
    .empty-state .logo {
        font-size: 40px;
        margin-bottom: 16px;
    }
    .empty-state h2 {
        color: #e5e5e5;
        font-size: 24px;
        font-weight: 600;
        margin: 0 0 20px 0;
        letter-spacing: -0.01em;
    }
    .empty-state .hint {
        color: #555;
        font-size: 13px;
        margin: 0;
    }

    /* ===== 对话气泡 ===== */
    .msg-user {
        background: #2a2a2a;
        color: #e5e5e5;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 10px 0 10px auto;
        max-width: 72%;
        font-size: 15px;
        line-height: 1.6;
    }
    .msg-ai {
        background: transparent;
        color: #e5e5e5;
        padding: 4px 0;
        margin: 10px 0;
        max-width: 72%;
        font-size: 15px;
        line-height: 1.6;
    }

    /* ===== 来源标签 ===== */
    .source-tag {
        display: inline-block;
        background: #1e1e1e;
        color: #888;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 11px;
        margin: 2px;
        border: 1px solid #333;
    }

    /* ===== chunk 卡片 ===== */
    .chunk-card {
        background: #1e1e1e;
        border: 1px solid #2a2a2a;
        padding: 10px 14px;
        margin: 5px 0;
        border-radius: 8px;
        font-size: 13px;
        color: #bbb;
        line-height: 1.5;
    }
    .chunk-card:hover {
        border-color: #444;
    }
    .chunk-card b {
        color: #e5e5e5;
    }

    /* ===== 按钮 ===== */
    .stButton > button {
        border-radius: 10px;
        font-weight: 500;
        font-size: 13px;
        transition: all 0.15s ease;
    }
    .stButton > button[kind="primary"] {
        background: #2a2a2a !important;
        border: 1px solid #444 !important;
        color: #e5e5e5 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #333 !important;
    }
    .stButton > button:not([kind="primary"]) {
        background: transparent !important;
        border: 1px solid #333 !important;
        color: #888 !important;
    }
    .stButton > button:not([kind="primary"]):hover {
        border-color: #555 !important;
        color: #e5e5e5 !important;
    }

    /* ===== 展开器 ===== */
    [data-testid="stExpander"] {
        border: 1px solid #2a2a2a;
        border-radius: 10px;
        background: transparent;
    }
    [data-testid="stExpander"] summary {
        font-size: 13px;
        color: #bbb;
        font-weight: 500;
        padding: 10px 14px;
    }
    [data-testid="stExpander"] summary:hover {
        color: #e5e5e5;
    }
    [data-testid="stExpander"] .streamlit-expanderContent {
        padding: 6px 14px 14px;
    }

    /* ===== 输入框 ===== */
    .stChatInput textarea {
        background: #1e1e1e !important;
        border: 1px solid #333 !important;
        border-radius: 12px !important;
        color: #e5e5e5 !important;
        font-size: 15px !important;
        padding: 14px 18px !important;
    }
    .stChatInput textarea:focus {
        border-color: #555 !important;
        box-shadow: none !important;
    }
    .stChatInput textarea::placeholder {
        color: #555 !important;
    }

    /* ===== 滑块 ===== */
    .stSlider label {
        color: #888 !important;
        font-weight: 500 !important;
        font-size: 12px !important;
    }

    /* ===== 选择框 ===== */
    .stSelectbox label {
        color: #888 !important;
        font-weight: 500 !important;
        font-size: 12px !important;
    }
    [data-baseweb="select"] > div {
        background: #1e1e1e !important;
        border-color: #333 !important;
        color: #e5e5e5 !important;
    }

    /* ===== 分割线 ===== */
    hr {
        border-color: #2a2a2a !important;
    }

    /* ===== 进度条 ===== */
    .stProgress > div > div > div > div {
        background: #444;
    }

    /* ===== alert ===== */
    .stAlert {
        background: #1e1e1e !important;
        border: 1px solid #333 !important;
        border-radius: 10px !important;
    }
    .stAlert > div {
        color: #e5e5e5 !important;
    }

    /* ===== 文件上传 ===== */
    [data-testid="stFileUploader"] {
        background: #1e1e1e;
        border: 1px dashed #333;
        border-radius: 10px;
    }
    [data-testid="stFileUploader"] > div:first-child {
        color: #888 !important;
    }

    /* ===== spinner ===== */
    .stSpinner > div {
        border-top-color: #555 !important;
    }
</style>
""", unsafe_allow_html=True)


# ==================== 单例初始化 ====================

@st.cache_resource
def init_services():
    """初始化所有服务（单例）"""
    from backend.services.document_processor import DocumentProcessor
    from backend.services.vectorstore_service import VectorstoreService
    from backend.chains.rag_chain import RAGChain

    processor = DocumentProcessor()
    vs_service = VectorstoreService()
    rag_chain = RAGChain()
    return processor, vs_service, rag_chain


# ==================== 侧边栏 ====================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h1>RAG 知识库</h1>
        <p>检索增强生成问答平台</p>
    </div>
    """, unsafe_allow_html=True)

    processor, vs_service, rag_chain = init_services()

    # ===== 知识库统计 =====
    stats = vs_service.get_stats()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("文档", stats['total_documents'])
    with col2:
        st.metric("文档块", stats['total_chunks'])

    st.markdown("---")

    # ===== 知识库管理 =====
    st.markdown("### 文档")

    # 上传
    with st.expander("上传文档", expanded=False):
        uploaded_files = st.file_uploader(
            "选择文件",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if uploaded_files and st.button("处理入库", use_container_width=True):
            total = 0
            progress = st.progress(0)
            for i, file in enumerate(uploaded_files):
                content = file.read()
                docs = processor.process_bytes(content, file.name)
                if docs:
                    vs_service.add_documents(docs)
                    total += len(docs)
                progress.progress((i + 1) / len(uploaded_files))
            st.success(f"入库 {total} 个文档块")
            st.rerun()

    # 文档列表 + 查看 chunks
    docs = vs_service.list_documents()
    if docs:
        for doc in docs:
            with st.expander(f"📄 {doc['source']} ({doc['chunk_count']})", expanded=False):
                chunks = vs_service.get_chunks_by_source(doc['source'])
                for chunk in chunks:
                    st.markdown(
                        f'<div class="chunk-card">'
                        f'<b>#{chunk["chunk_id"]}</b> [{chunk["chunk_type"]}]<br>'
                        f'{chunk["content"][:300]}{"..." if len(chunk["content"]) > 300 else ""}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                if st.button("删除", key=f"del_{doc['source']}", use_container_width=True):
                    vs_service.delete_by_source(doc['source'])
                    st.rerun()
    else:
        st.markdown('<div style="color:#555; font-size:13px; padding:8px;">知识库为空</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ===== 文档切分参数 =====
    st.markdown("### 切分参数")

    chunk_strategy = st.selectbox(
        "切分策略",
        ["recursive", "parent_child"],
        help="recursive=递归字符切分，parent_child=父子文档切分"
    )
    chunk_size = st.slider("Chunk Size", 100, 2000, 800, 50,
                           help="每个文档块的最大字符数")
    chunk_overlap = st.slider("Overlap", 0, 500, 200, 25,
                              help="相邻文档块的重叠字符数")

    st.markdown("---")

    # ===== 检索参数 =====
    st.markdown("### 检索参数")

    top_k = st.slider("Top K", 1, 20, 5,
                       help="检索返回的文档数量")
    search_type = st.selectbox("检索策略", ["mmr", "similarity"])
    if search_type == "mmr":
        lambda_mult = st.slider("多样性 λ", 0.0, 1.0, 0.7, 0.05)
    else:
        lambda_mult = 0.7

    st.markdown("---")

    # ===== LLM 参数 =====
    st.markdown("### 模型参数")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.slider("Max Tokens", 256, 4096, 2048, 256)

    st.markdown("---")

    # ===== 操作 =====
    col3, col4 = st.columns(2)
    with col3:
        if st.button("清空对话", use_container_width=True):
            rag_chain.clear_memory()
            st.session_state.messages = []
            st.rerun()
    with col4:
        if st.button("清空知识库", use_container_width=True):
            vs_service.clear_all()
            st.rerun()


# ==================== 主区域：对话 ====================

# 初始化
if "messages" not in st.session_state:
    st.session_state.messages = []

chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        # 空状态 - 参考 DeepSeek
        st.markdown("""
        <div class="empty-state">
            <div class="logo">🤖</div>
            <h2>想从哪里开始？</h2>
            <p class="hint">输入问题，从知识库中检索资料生成回答</p>
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="msg-user">{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            content = msg["content"]
            sources = msg.get("sources", [])

            st.markdown(
                f'<div class="msg-ai">{content}</div>',
                unsafe_allow_html=True
            )

            # 来源标签
            if sources:
                source_html = " ".join([
                    f'<span class="source-tag">{s.get("source", "未知")}</span>'
                    for s in sources
                ])
                st.markdown(
                    f'<div style="margin-bottom:16px;">{source_html}</div>',
                    unsafe_allow_html=True
                )

# ==================== 输入区域 ====================

question = st.chat_input("输入问题...")

if question:
    if rag_chain is None:
        st.error("服务未初始化")
    else:
        # 显示用户问题
        st.markdown(
            f'<div class="msg-user">{question}</div>',
            unsafe_allow_html=True
        )
        st.session_state.messages.append({"role": "user", "content": question})

        # 更新检索参数
        rag_chain.retrieval_config = {
            'search_type': search_type,
            'search_kwargs': {
                'k': top_k,
                'fetch_k': max(top_k * 4, 20),
                'lambda_mult': lambda_mult,
            }
        }
        rag_chain.retriever = rag_chain._init_retriever()

        # 生成回答
        try:
            sources = rag_chain.get_sources(question)

            with st.chat_message("assistant"):
                with st.spinner("思考中..."):
                    response_placeholder = st.empty()
                    full_answer = ""

                    for chunk in rag_chain.stream(question):
                        full_answer += chunk
                        response_placeholder.markdown(full_answer + "▌")

                    response_placeholder.markdown(full_answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "sources": sources
            })
        except Exception as e:
            st.error(f"错误：{str(e)}")

# ==================== 绿色悬浮按钮 + 自动展开侧边栏（JS 注入） ====================
components.html("""
<script>
(function() {
    var doc = window.parent.document;

    // 页面加载后自动尝试展开侧边栏
    setTimeout(function() {
        var allBtns = doc.querySelectorAll('button');
        for (var i = 0; i < allBtns.length; i++) {
            var label = (allBtns[i].getAttribute('aria-label') || '').toLowerCase();
            if (label.indexOf('sidebar') !== -1) {
                var sidebar = doc.querySelector('[data-testid="stSidebar"]');
                if (sidebar && sidebar.offsetWidth === 0) {
                    allBtns[i].click();
                }
                return;
            }
        }
    }, 1000);

    // 避免重复注入悬浮按钮
    if (doc.getElementById('sidebar-toggle-fab')) return;

    var btn = doc.createElement('button');
    btn.id = 'sidebar-toggle-fab';
    btn.textContent = '☰';
    Object.assign(btn.style, {
        position: 'fixed',
        top: '12px',
        left: '12px',
        zIndex: '999999',
        width: '40px',
        height: '40px',
        borderRadius: '10px',
        background: '#22c55e',
        border: 'none',
        color: '#fff',
        fontSize: '20px',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        boxShadow: '0 2px 12px rgba(34,197,94,0.4)',
        transition: 'all 0.2s ease',
        fontFamily: 'system-ui'
    });
    btn.onmouseenter = function() { btn.style.background = '#16a34a'; btn.style.transform = 'scale(1.08)'; };
    btn.onmouseleave = function() { btn.style.background = '#22c55e'; btn.style.transform = 'scale(1)'; };
    btn.onclick = function() {
        var allBtns = doc.querySelectorAll('button');
        for (var i = 0; i < allBtns.length; i++) {
            var label = (allBtns[i].getAttribute('aria-label') || '').toLowerCase();
            if (label.indexOf('sidebar') !== -1) {
                allBtns[i].click();
                return;
            }
        }
        // 兜底：找 collapsedControl 容器
        var cc = doc.querySelector('[data-testid="collapsedControl"]');
        if (cc) { cc.click(); return; }
        // 再兜底：点击侧边栏本身让它出现
        var sidebar = doc.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            sidebar.style.display = 'block';
            sidebar.style.width = '380px';
        }
    };
    doc.body.appendChild(btn);
})();
</script>
""", height=0, width=0)

