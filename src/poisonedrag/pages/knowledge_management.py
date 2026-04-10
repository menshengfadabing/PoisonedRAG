import sys
from pathlib import Path
import streamlit as st

from dotenv import load_dotenv
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env")

src_dir = _project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from poisonedrag.config import get_config
from poisonedrag.embeddings import get_embedding_model
from poisonedrag.vectorstore import get_vectorstore
from poisonedrag.data.knowledge_base import create_knowledge_base


def init_session_state():
    """初始化会话状态"""
    pass


# Page config
st.set_page_config(
    page_title="知识库管理",
    page_icon="📚",
    layout="wide",
)

# Sidebar navigation
with st.sidebar:
    st.title("📚 知识库管理")
    st.markdown("---")
    st.subheader("页面导航")
    if st.button("💬 对话主页", use_container_width=True):
        st.switch_page("app.py")
    if st.button("📥 人工审核", use_container_width=True):
        st.switch_page("pages/review.py")
    if st.button("⚙️ 设置", use_container_width=True):
        st.switch_page("pages/settings.py")

# Initialize components
@st.cache_resource
def init_components():
    config = get_config()
    embedding_model = get_embedding_model()
    vectorstore = get_vectorstore(embedding_function=embedding_model.embeddings)
    kb = create_knowledge_base(vectorstore=vectorstore)
    return kb, vectorstore

kb, vectorstore = init_components()

st.title("📚 知识库管理")

# ============================================================
# 知识库状态概览
# ============================================================
doc_count = vectorstore.count()
st.metric("向量存储文档数", doc_count)

# ============================================================
# Tabs
# ============================================================
tab_load, tab_docs, tab_clear = st.tabs(["📥 加载知识库", "📋 文档状态", "🗑️ 清空知识库"])

# ============================================================
# Tab 1: Load
# ============================================================
with tab_load:
    st.subheader("加载知识库")
    st.caption("从 `data/knowledge/` 目录加载所有 JSON 文件到向量存储")

    if st.button("📥 开始加载", type="primary"):
        with st.spinner("正在加载..."):
            try:
                documents = kb.load_from_directory()
                total_files = len(documents)
                if total_files == 0:
                    st.warning("⚠️ `data/knowledge/` 目录下没有找到可加载的文档文件")
                else:
                    indexed_count = kb.index_to_vectorstore(documents)
                    st.success(f"✅ 成功索引 {indexed_count} 个文档")
                    st.rerun()
            except Exception as e:
                st.error(f"加载失败: {e}")

# ============================================================
# Tab 2: Document status
# ============================================================
with tab_docs:
    st.subheader("文档列表")

    if doc_count == 0:
        st.info("知识库为空，请先加载文档")
    else:
        # Get all documents from vectorstore
        all_docs = vectorstore.vectorstore.get()
        ids = all_docs.get("ids", [])
        metadatas = all_docs.get("metadatas", [])
        documents = all_docs.get("documents", [])

        # Pagination
        page_size = 20
        total_pages = max(1, (len(ids) + page_size - 1) // page_size)
        page = st.number_input("页码", min_value=1, max_value=total_pages, value=1)

        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, len(ids))

        for i in range(start_idx, end_idx):
            doc_id = ids[i]
            meta = metadatas[i] if i < len(metadatas) else {}
            content = documents[i] if i < len(documents) else ""

            source = meta.get("source", "未知")
            doc_type = meta.get("type", meta.get("category", "未知"))

            # Status badge
            status_icon = "✅"

            with st.expander(f"{status_icon} [{doc_type}] `{doc_id[:8]}` — {source}"):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"**ID:** `{doc_id}`")
                    st.markdown(f"**来源:** {source}")
                    st.markdown(f"**类型:** {doc_type}")
                    # Show any extra metadata
                    extra_meta = {k: v for k, v in meta.items() if k not in ("source", "type", "category")}
                    if extra_meta:
                        st.markdown("**其他元数据:**")
                        for k, v in extra_meta.items():
                            st.caption(f"{k}: {v}")
                with col2:
                    st.markdown("**内容预览:**")
                    st.text(content[:500] + ("..." if len(content) > 500 else ""))

        st.caption(f"共 {len(ids)} 条文档，第 {page}/{total_pages} 页")

# ============================================================
# Tab 3: Clear
# ============================================================
with tab_clear:
    st.subheader("清空知识库")
    st.caption("清空向量存储中的所有文档")

    current_count = vectorstore.count()
    st.info(f"当前向量存储中共有 **{current_count}** 个文档")

    if current_count > 0:
        if st.button("🗑️ 确认清空", type="primary"):
            with st.spinner("正在清空..."):
                kb.clear()
            st.success("✅ 知识库已清空")
            st.rerun()
    else:
        st.info("知识库已经是空的，无需清空")
