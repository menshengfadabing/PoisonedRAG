import sys
import os
from pathlib import Path
import streamlit as st
from datetime import datetime

from dotenv import load_dotenv
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env")

src_dir = _project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from langchain_core.documents import Document

from poisonedrag.config import get_config
from poisonedrag.embeddings import get_embedding_model
from poisonedrag.vectorstore import get_vectorstore
from poisonedrag.data.knowledge_base import create_knowledge_base
from poisonedrag.resecurity import (
    DocumentReviewer,
    ReviewQueueManager,
    TextSplitter,
    create_document_reviewer,
    create_review_queue_manager,
    create_text_splitter,
    ChunkReviewResult,
    ReviewStatus,
)
from poisonedrag.llm import get_review_llm

# ============================================================
# 三档审查评级映射
# ============================================================

PASS_REVIEW_THRESHOLD = 0.3     # 低于此值 → pass（直接入库）
REVIEW_FAIL_THRESHOLD = 0.7     # 高于等于此值 → fail（拒绝入库）
                                # 中间区间 → review（交给人工审查）


def init_session_state():
    """初始化会话状态"""
    if "review_queue_manager" not in st.session_state:
        st.session_state.review_queue_manager = create_review_queue_manager()

    if "document_reviewer" not in st.session_state:
        st.session_state.document_reviewer = None

    if "text_splitter" not in st.session_state:
        config = get_config()
        st.session_state.text_splitter = create_text_splitter(
            chunk_size=config.review_doc_chunk_size,
            chunk_overlap=config.review_doc_chunk_overlap,
        )

    # 上传结果缓存（避免 rerun 时重复处理）
    if "upload_results" not in st.session_state:
        st.session_state.upload_results = None


def get_document_reviewer() -> DocumentReviewer:
    """获取文档审查器（延迟初始化）"""
    if st.session_state.document_reviewer is None:
        with st.spinner("正在初始化审查模型..."):
            llm = get_review_llm()
            st.session_state.document_reviewer = create_document_reviewer(llm=llm.llm)
    return st.session_state.document_reviewer


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

# Initialize session state
init_session_state()

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
queue_manager = st.session_state.review_queue_manager
stats = queue_manager.get_stats()

col1, col2, col3, col4 = st.columns(4)
col1.metric("向量存储文档", doc_count)
col2.metric("待审核", stats["pending"])
col3.metric("已通过", stats["approved"] + stats["auto_approved"])
col4.metric("已拒绝", stats["rejected"])

# ============================================================
# Tabs
# ============================================================
tab_upload, tab_load, tab_docs, tab_clear = st.tabs([
    "📤 上传语料",
    "📥 加载知识库",
    "📋 文档状态",
    "🗑️ 清空知识库",
])


# ============================================================
# Tab 1: Upload 语料（上传 → 分割 → 审查 → 三档评级）
# ============================================================
with tab_upload:
    st.subheader("📤 上传语料")
    st.caption("上传文件后，系统自动分割、审查，按三档评级处理：")
    st.markdown("""
    | 评级 | 风险分数 | 处理方式 |
    |------|----------|----------|
    | ✅ **pass** | < 0.3 | 直接入库 |
    | ⚠️ **review** | 0.3 ~ 0.7 | 进入人工审核队列 |
    | ❌ **fail** | ≥ 0.7 | 拒绝入库 |
    """)

    uploaded_files = st.file_uploader(
        "选择文件（支持 .txt / .md / .json / .pdf / .docx / .pptx，可多选）",
        type=["txt", "md", "json", "pdf", "docx", "pptx"],
        accept_multiple_files=True,
        key="knowledge_uploader",
    )

    if uploaded_files:
        if st.button("🔍 开始审查并入库", type="primary", use_container_width=True):
            reviewer = get_document_reviewer()
            splitter = st.session_state.text_splitter
            qm = st.session_state.review_queue_manager

            all_chunks = []       # (content, source)
            all_results = []      # ChunkReviewResult

            # 1. 读取文件并提取文本（支持 PDF/DOCX/PPTX/MD/JSON/TXT）
            with st.spinner("正在读取文件并提取文本..."):
                import tempfile
                from poisonedrag.data.document_loader import (
                    load_file as load_doc_file,
                    SUPPORTED_EXTENSIONS as DOC_EXTENSIONS,
                )

                for uf in uploaded_files:
                    ext = Path(uf.name).suffix.lower()
                    if ext not in DOC_EXTENSIONS:
                        st.warning(f"⚠️ 跳过不支持的文件格式: {uf.name}")
                        continue

                    # 将上传的文件保存到临时文件，再用加载器解析
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=ext, dir="/tmp"
                    ) as tmp:
                        tmp.write(uf.read())
                        tmp_path = tmp.name

                    try:
                        documents = load_doc_file(tmp_path)
                        if not documents:
                            st.warning(f"⚠️ {uf.name} 未提取到文本内容")
                            continue

                        # 对每个文档块进行文本分割
                        for doc in documents:
                            chunks = splitter.split_text(doc.page_content)
                            for chunk in chunks:
                                all_chunks.append((chunk, uf.name))
                    except Exception as e:
                        st.error(f"❌ 解析 {uf.name} 失败: {e}")
                    finally:
                        os.unlink(tmp_path)

                st.info(f"共提取并分割出 {len(all_chunks)} 个文本块")

            if not all_chunks:
                st.warning("⚠️ 未提取到有效文本内容")
            else:
                # 2. 批量审查
                with st.spinner("正在调用 LLM 进行安全审查..."):
                    chunk_texts = [c[0] for c in all_chunks]
                    review_results = reviewer.batch_quick_review(
                        chunk_texts, use_parallel=True
                    )
                    all_results = review_results

                # 3. 按三档评级分别处理
                pass_count = 0
                review_count = 0
                fail_count = 0
                fail_details = []
                review_items = []

                for i, result in enumerate(all_results):
                    chunk_text, source_file = all_chunks[i]

                    if result.risk_score < PASS_REVIEW_THRESHOLD:
                        # === pass：直接入库 ===
                        pass_count += 1
                        doc = Document(
                            page_content=chunk_text,
                            metadata={
                                "source": source_file,
                                "type": "knowledge",
                                "reviewed": True,
                                "risk_score": result.risk_score,
                                "upload_time": datetime.now().isoformat(),
                            }
                        )
                        try:
                            vectorstore.add_documents([doc])
                        except Exception as e:
                            st.error(f"pass 文档入库失败: {e}")

                    elif result.risk_score < REVIEW_FAIL_THRESHOLD:
                        # === review：加入人工审核队列 ===
                        review_count += 1
                        # 生成唯一 ID
                        doc_id = (
                            f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
                        )
                        from poisonedrag.resecurity.review_queue_manager import (
                            QueuedDocument,
                            ReviewLog,
                        )
                        queued_doc = QueuedDocument(
                            id=doc_id,
                            content=chunk_text,
                            risk_score=result.risk_score,
                            risk_type=result.risk_type,
                            reason=result.reason,
                            source=source_file,
                            created_at=datetime.now().isoformat(),
                            status=ReviewStatus.PENDING,
                        )
                        qm._queue.append(queued_doc)
                        review_items.append(queued_doc)

                    else:
                        # === fail：拒绝入库 ===
                        fail_count += 1
                        fail_details.append({
                            "source": source_file,
                            "risk_score": result.risk_score,
                            "risk_type": result.risk_type,
                            "reason": result.reason,
                        })
                        # 记录日志
                        from poisonedrag.resecurity.review_queue_manager import ReviewLog
                        doc_id = (
                            f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
                        )
                        qm._logs.append(ReviewLog(
                            timestamp=datetime.now().isoformat(),
                            doc_id=doc_id,
                            action="rejected",
                            risk_score=result.risk_score,
                            risk_type=result.risk_type,
                            reason=result.reason,
                            operator="system",
                            comment="风险分数超过 fail 阈值，自动拒绝",
                        ))

                # 保存队列和日志
                qm._save_queue()
                qm._save_logs()

                # 缓存结果用于展示
                st.session_state.upload_results = {
                    "pass_count": pass_count,
                    "review_count": review_count,
                    "fail_count": fail_count,
                    "review_items": review_items,
                    "fail_details": fail_details,
                    "total_chunks": len(all_chunks),
                }

                st.rerun()

        # 显示上次上传的审查结果
        results = st.session_state.upload_results
        if results:
            st.markdown("---")
            st.subheader("📊 上次审查结果")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("总块数", results["total_chunks"])
            c2.metric("✅ pass（直接入库）", results["pass_count"])
            c3.metric("⚠️ review（待人工审核）", results["review_count"])
            c4.metric("❌ fail（拒绝入库）", results["fail_count"])

            # 展示 review 项目，引导用户去人工审核页面
            if results["review_items"]:
                st.warning(
                    f"有 **{len(results['review_items'])}** 个文本块需要人工审核，"
                    f"请前往 [📥 人工审核页面](pages/review.py) 处理。"
                )
                for item in results["review_items"]:
                    risk_color = "🟡"
                    with st.expander(
                        f"{risk_color} [{item.risk_score:.2f}] {item.source}"
                    ):
                        st.text(item.content[:500])
                        st.caption(f"风险类型: {item.risk_type or '未知'}")
                        if item.reason:
                            st.caption(f"LLM 判断: {item.reason}")

            # 展示 fail 项目
            if results["fail_details"]:
                st.error(
                    f"有 **{len(results['fail_details'])}** 个文本块因高风险被拒绝。"
                )
                for item in results["fail_details"]:
                    with st.expander(
                        f"❌ [{item['risk_score']:.2f}] {item['source']}"
                    ):
                        st.caption(f"风险类型: {item['risk_type'] or '未知'}")
                        st.caption(f"原因: {item['reason'] or '无'}")

            # 清理结果缓存（避免下次打开页面还显示旧结果）
            if st.button("🧹 清除结果展示", key="clear_upload_results"):
                st.session_state.upload_results = None
                st.rerun()


# ============================================================
# Tab 2: Load from directory
# ============================================================
with tab_load:
    st.subheader("加载知识库")
    st.caption("从 `data/knowledge/` 目录加载所有 JSON/MD/TXT 文件到向量存储（**不经过审查**）")

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
# Tab 3: Document status
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
            reviewed = meta.get("reviewed", False)
            risk = meta.get("risk_score", None)

            # Status badge
            status_icon = "✅" if reviewed else "📄"

            title_extra = ""
            if risk is not None:
                if risk < PASS_REVIEW_THRESHOLD:
                    title_extra = " | 🟢 pass"
                elif risk < REVIEW_FAIL_THRESHOLD:
                    title_extra = " | 🟡 review"
                else:
                    title_extra = " | 🔴 fail"

            with st.expander(
                f"{status_icon} [{doc_type}] `{doc_id[:8]}` — {source}{title_extra}"
            ):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"**ID:** `{doc_id}`")
                    st.markdown(f"**来源:** {source}")
                    st.markdown(f"**类型:** {doc_type}")
                    if risk is not None:
                        st.markdown(f"**风险分数:** {risk:.2f}")
                    # Show any extra metadata
                    extra_meta = {
                        k: v for k, v in meta.items()
                        if k not in ("source", "type", "category", "reviewed", "risk_score", "upload_time")
                    }
                    if extra_meta:
                        st.markdown("**其他元数据:**")
                        for k, v in extra_meta.items():
                            st.caption(f"{k}: {v}")
                with col2:
                    st.markdown("**内容预览:**")
                    st.text(content[:500] + ("..." if len(content) > 500 else ""))

        st.caption(f"共 {len(ids)} 条文档，第 {page}/{total_pages} 页")


# ============================================================
# Tab 4: Clear
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
