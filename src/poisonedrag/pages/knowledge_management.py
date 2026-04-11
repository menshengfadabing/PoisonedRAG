import sys
import os
from pathlib import Path
import streamlit as st
import json
import tempfile
from datetime import datetime

from dotenv import load_dotenv
_project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(_project_root / ".env")

src_dir = _project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

from poisonedrag.config import get_config
from poisonedrag.embeddings import get_embedding_model
from poisonedrag.vectorstore import get_vectorstore
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
from poisonedrag.data.document_loader import (
    load_file as load_doc_file,
    SUPPORTED_EXTENSIONS as DOC_EXTENSIONS,
)

# ============================================================
# 审查评级映射
# ============================================================

STATUS_MAP = {
    "p": {"label": "✅ pass", "color": "green", "action": "pass"},
    "r": {"label": "⚠️ review", "color": "orange", "action": "review"},
    "f": {"label": "❌ fail", "color": "red", "action": "fail"},
}

# 系统提示词：仅返回 id + p/r/f，节省 token
BATCH_REVIEW_PROMPT = """你是文档安全审查员。批量审查文档，只返回 JSON。

返回格式（严格遵守）:
{"results": [{"id": 1, "status": "p"}]}

status 只有三个值:
- p: 安全，无风险
- r: 有风险，需要人工复核
- f: 高风险，拒绝入库

检查重点：
1. 可疑包名（仿冒正规库）
2. 硬编码凭证（密码/密钥/token）
3. 危险命令（eval/exec/os.system/rm -rf）
4. 钓鱼欺诈/误导建议/依赖投毒/错误事实注入

返回纯 JSON，不要任何其他文字。"""


def init_session_state():
    """初始化会话状态"""
    if "review_queue_manager" not in st.session_state:
        st.session_state.review_queue_manager = create_review_queue_manager()

    if "document_reviewer" not in st.session_state:
        st.session_state.document_reviewer = None

    if "upload_results" not in st.session_state:
        st.session_state.upload_results = None

    # 入库进度（用于 embedding 阶段展示）
    if "index_progress" not in st.session_state:
        st.session_state.index_progress = None

    # 审查进度持久化（防止切换页面丢失）
    if "review_state" not in st.session_state:
        st.session_state.review_state = None


def _process_review_results(all_chunks, all_results):
    """
    处理审查结果：分类、入库、更新队列和日志。
    可被主上传流程和恢复流程共用。
    """
    qm = st.session_state.review_queue_manager
    pass_docs = []
    review_count = 0
    fail_count = 0
    fail_details = []
    review_items = []

    for i, result in enumerate(all_results):
        chunk_text, source_file = all_chunks[i]
        status = "p" if result.risk_score < 0.2 else "r" if result.risk_score < 0.8 else "f"

        if status == "p":
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
            pass_docs.append(doc)

        elif status == "r":
            review_count += 1
            doc_id = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
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

        else:  # f
            fail_count += 1
            fail_details.append({
                "source": source_file,
                "chunk_id": i,
                "content_preview": chunk_text[:100],
                "status": "fail",
                "reason": result.reason,
            })
            doc_id = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
            qm._logs.append(ReviewLog(
                timestamp=datetime.now().isoformat(),
                doc_id=doc_id,
                action="rejected",
                risk_score=result.risk_score,
                risk_type=result.risk_type,
                reason=result.reason,
                operator="system",
                comment="高风险，拒绝入库",
            ))

    # 批量入库（带进度条）
    pass_count = len(pass_docs)
    if pass_docs:
        st.markdown("---")
        st.subheader("📥 入库进度")
        progress_bar = st.progress(0)
        status_text = st.empty()

        embed_batch_size = 20
        total_batches = (pass_count + embed_batch_size - 1) // embed_batch_size

        for batch_idx in range(0, pass_count, embed_batch_size):
            batch = pass_docs[batch_idx:batch_idx + embed_batch_size]
            batch_num = batch_idx // embed_batch_size + 1
            status_text.info(
                f"⏳ 正在向量化第 {batch_num}/{total_batches} 批 "
                f"({batch_idx + 1}-{min(batch_idx + embed_batch_size, pass_count)}/{pass_count})..."
            )
            try:
                vectorstore.add_documents(batch)
            except Exception as e:
                st.error(f"批量入库失败: {e}")
            progress_bar.progress(min(1.0, (batch_idx + embed_batch_size) / pass_count))

        progress_bar.progress(1.0)
        status_text.success(f"✅ 全部 {pass_count} 篇文档已入库")

    # 保存队列和日志
    qm._save_queue()
    qm._save_logs()

    # 缓存结果
    st.session_state.upload_results = {
        "pass_count": pass_count,
        "review_count": review_count,
        "fail_count": fail_count,
        "review_items": review_items,
        "fail_details": fail_details,
        "total_chunks": len(all_chunks),
        "results": [
            {
                "chunk_id": i,
                "source": all_chunks[i][1],
                "content": all_chunks[i][0],
                "status": "p" if r.risk_score < 0.2 else "r" if r.risk_score < 0.8 else "f",
                "risk_score": r.risk_score,
                "risk_type": r.risk_type,
                "reason": r.reason,
            }
            for i, r in enumerate(all_results)
        ],
    }

    # 清除审查进度（已完成）
    st.session_state.review_state = None

    # 自动刷新页面
    st.rerun()


def get_document_reviewer() -> DocumentReviewer:
    """获取文档审查器"""
    if st.session_state.document_reviewer is None:
        with st.spinner("正在初始化审查模型..."):
            llm = get_review_llm()
            st.session_state.document_reviewer = create_document_reviewer(llm=llm.llm)
    return st.session_state.document_reviewer


def batch_review_stream(
    reviewer: DocumentReviewer,
    chunks: list[str],
    batch_size: int = 20,
    state_key: str = "review_state",
) -> list[ChunkReviewResult]:
    """
    批量审查，带流式输出展示。
    每完成一批就保存进度到 session_state，防止切换页面丢失。
    返回 ChunkReviewResult 列表。
    """
    llm = reviewer.llm
    if llm is None:
        return [
            ChunkReviewResult(chunk_id=i, content=c, risk_score=0.5,
                              risk_type="LLM未初始化", reason="LLM 未配置")
            for i, c in enumerate(chunks)
        ]

    total_batches = (len(chunks) + batch_size - 1) // batch_size
    try:
        status_placeholder = st.empty()
        raw_display = st.empty()
    except Exception:
        status_placeholder = None
        raw_display = None

    all_results = []
    for batch_idx, batch_start in enumerate(range(0, len(chunks), batch_size)):
        batch = chunks[batch_start:batch_start + batch_size]
        chunks_text = "\n---\n".join(
            f"[{i+1}] {c[:reviewer.doc_max_length]}"
            for i, c in enumerate(batch)
        )

        user_prompt = f"""审查以下 {len(batch)} 个文档，返回 JSON:

{chunks_text}

格式: {{"results": [{{"id": 1, "status": "p"}}]}}
只返回 JSON。"""

        messages = [
            SystemMessage(content=BATCH_REVIEW_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        raw_response = ""
        for chunk in llm.stream(messages):
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            if content:
                raw_response += content
                if status_placeholder is not None:
                    status_placeholder.info(
                        f"⏳ 正在审查第 {batch_idx + 1}/{total_batches} 批 "
                        f"({batch_start + 1}-{min(batch_start + batch_size, len(chunks))}/{len(chunks)} 号) — "
                        f"已接收 {len(raw_response)} 字符..."
                    )
                if raw_display is not None:
                    raw_display.expander("📡 审查响应流（实时）", expanded=False).text(
                        raw_response[-500:] if len(raw_response) > 500 else raw_response
                    )

        # 解析 JSON
        batch_results = _parse_batch_response(raw_response, batch, batch_start)
        all_results.extend(batch_results)

        # 每完成一批就保存进度
        completed = len(all_results)
        st.session_state[state_key] = {
            "status": "in_progress",
            "total_chunks": len(chunks),
            "completed_chunks": completed,
            "total_batches": total_batches,
            "completed_batches": batch_idx + 1,
            "results": all_results,
        }

    if status_placeholder is not None:
        status_placeholder.empty()
    if raw_display is not None:
        raw_display.empty()

    # 审查完成，标记状态
    st.session_state[state_key] = {
        "status": "complete",
        "total_chunks": len(chunks),
        "completed_chunks": len(all_results),
        "total_batches": total_batches,
        "completed_batches": total_batches,
        "results": all_results,
    }

    return all_results


def _parse_batch_response(
    raw_content: str,
    chunks: list[str],
    start_id: int,
) -> list[ChunkReviewResult]:
    """解析 LLM 返回的 p/r/f JSON 响应"""
    import re

    json_match = re.search(r'\{[\s\S]*\}', raw_content)
    if not json_match:
        return [
            ChunkReviewResult(
                chunk_id=start_id + i, content=chunk,
                risk_score=0.5, risk_type="解析失败", reason="JSON 格式异常"
            )
            for i, chunk in enumerate(chunks)
        ]

    try:
        data = json.loads(json_match.group())
        results_list = data.get("results", [])
        result_map = {r.get("id"): r for r in results_list}

        score_map = {"p": 0.0, "r": 0.5, "f": 0.9}
        type_map = {
            "p": None,
            "r": "需要人工复核",
            "f": "高风险",
        }

        parsed = []
        for i, chunk in enumerate(chunks):
            doc_id = i + 1
            if doc_id in result_map:
                r = result_map[doc_id]
                status = r.get("status", "r").lower()
                parsed.append(ChunkReviewResult(
                    chunk_id=start_id + i,
                    content=chunk,
                    risk_score=score_map.get(status, 0.5),
                    risk_type=type_map.get(status, "未知"),
                    reason={"p": "安全，无风险", "r": "有风险，需人工复核", "f": "高风险，拒绝入库"}.get(status, "未知"),
                ))
            else:
                parsed.append(ChunkReviewResult(
                    chunk_id=start_id + i, content=chunk,
                    risk_score=0.5, risk_type="未审查", reason="LLM 未返回结果"
                ))

        return parsed

    except (json.JSONDecodeError, ValueError):
        return [
            ChunkReviewResult(
                chunk_id=start_id + i, content=chunk,
                risk_score=0.5, risk_type="解析失败", reason="JSON 解析错误"
            )
            for i, chunk in enumerate(chunks)
        ]


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="知识库管理",
    page_icon="📚",
    layout="wide",
)

# Sidebar
with st.sidebar:
    st.title("📚 知识库管理")
    st.markdown("---")
    st.subheader("页面导航")
    if st.button("💬 对话", use_container_width=True):
        st.switch_page("app.py")
    if st.button("📚 知识库管理", use_container_width=True, type="primary"):
        st.switch_page("pages/knowledge_management.py")
    if st.button("📥 人工审核", use_container_width=True):
        st.switch_page("pages/review.py")
    if st.button("🛡️ 防毒效果", use_container_width=True):
        st.switch_page("pages/visualization.py")
    if st.button("⚙️ 设置", use_container_width=True):
        st.switch_page("pages/settings.py")

# ============================================================
# Initialize
# ============================================================
init_session_state()

@st.cache_resource
def init_components():
    config = get_config()
    embedding_model = get_embedding_model()
    vectorstore = get_vectorstore(embedding_function=embedding_model.embeddings)
    splitter = create_text_splitter(
        chunk_size=config.review_doc_chunk_size,
        chunk_overlap=config.review_doc_chunk_overlap,
    )
    return embedding_model, vectorstore, splitter

embedding_model, vectorstore, splitter = init_components()

st.title("📚 知识库管理")

# ============================================================
# Status overview
# ============================================================
doc_count = vectorstore.count()
qm = st.session_state.review_queue_manager
stats = qm.get_stats()

col1, col2, col3, col4 = st.columns(4)
col1.metric("已入库文档", doc_count)
col2.metric("待人工审核", stats["pending"])
col3.metric("审核已通过", stats["approved"] + stats["auto_approved"])
col4.metric("审核已拒绝", stats["rejected"])

# ============================================================
# Tabs
# ============================================================
tab_upload, tab_docs, tab_clear = st.tabs([
    "📤 上传语料",
    "📋 文档列表",
    "🗑️ 清空知识库",
])

# ============================================================
# Tab 1: Upload + Review + Index
# ============================================================
with tab_upload:
    st.subheader("📤 上传语料")
    st.caption("上传文件 → 自动分割 → LLM 安全审查 → 通过审查的语料自动入库")
    st.markdown("""
    | 评级 | 含义 | 处理方式 |
    |------|------|----------|
    | ✅ **pass** | 安全无风险 | 自动入库 |
    | ⚠️ **review** | 需人工复核 | 进入人工审核队列 |
    | ❌ **fail** | 高风险 | 拒绝入库 |
    """)

    # 检查是否有未完成的审查进度（防止切换页面丢失）
    review_state = st.session_state.review_state
    if review_state and review_state.get("status") == "in_progress":
        completed = review_state.get("completed_batches", 0)
        total = review_state.get("total_batches", 0)
        total_chunks = review_state.get("total_chunks", 0)
        st.warning(
            f"⚠️ 检测到上次审查中断：已完成 **{completed}/{total}** 批（共 **{total_chunks}** 条语料）"
        )
        if st.button("🔄 从上次中断处继续审查", type="primary", use_container_width=True):
            # 继续从上次中断处审查
            reviewer = get_document_reviewer()
            all_chunks = review_state["all_chunks"]
            completed_chunks = review_state.get("completed_chunks", 0)

            # 跳过已完成的批次
            batch_size = reviewer.batch_size if hasattr(reviewer, 'batch_size') else 20
            remaining_chunks = all_chunks[completed_chunks:]
            remaining_texts = [c[0] for c in remaining_chunks]

            st.subheader("🔍 继续审查进度")
            remaining_results = batch_review_stream(
                reviewer, remaining_texts, batch_size=batch_size
            )

            # 合并结果
            all_results = review_state.get("results", []) + remaining_results

            # 继续处理后续流程（入库等）
            _process_review_results(all_chunks, all_results)
        if st.button("🗑️ 放弃进度，重新开始", use_container_width=True):
            st.session_state.review_state = None
            st.rerun()
        st.markdown("---")

    uploaded_files = st.file_uploader(
        "选择文件（支持 .txt / .md / .json / .pdf / .docx / .pptx，可多选）",
        type=["txt", "md", "json", "pdf", "docx", "pptx"],
        accept_multiple_files=True,
        key="knowledge_uploader",
    )

    if uploaded_files:
        config = get_config()
        chunk_size = st.number_input(
            "语料分割大小（字符数）",
            min_value=100,
            max_value=5000,
            value=config.review_doc_chunk_size,
            step=50,
            help="每个语料块的最大字符数",
            key="upload_chunk_size",
        )
        chunk_overlap = st.number_input(
            "语料分割重叠（字符数）",
            min_value=0,
            max_value=500,
            value=config.review_doc_chunk_overlap,
            step=10,
            help="相邻语料块的重叠字符数",
            key="upload_chunk_overlap",
        )

        if st.button("🔍 开始审查并入库", type="primary", use_container_width=True):
            reviewer = get_document_reviewer()
            qm = st.session_state.review_queue_manager

            # 动态创建分割器
            from poisonedrag.resecurity import create_text_splitter as make_splitter
            temp_splitter = make_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            all_chunks = []  # (content, source)

            # 1. 读取文件并提取文本
            with st.spinner("正在读取文件并提取文本..."):
                for uf in uploaded_files:
                    ext = Path(uf.name).suffix.lower()
                    if ext not in DOC_EXTENSIONS:
                        st.warning(f"⚠️ 跳过不支持的文件格式: {uf.name}")
                        continue

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

                        for doc in documents:
                            chunks = temp_splitter.split_text(doc.page_content)
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
                config = get_config()

                # 保存 chunks 到 session_state，防止切换页面丢失
                st.session_state.review_state = {
                    "status": "starting",
                    "total_chunks": len(all_chunks),
                    "completed_chunks": 0,
                    "total_batches": 0,
                    "completed_batches": 0,
                    "results": [],
                    "all_chunks": all_chunks,  # (content, source) 列表
                }

                # 根据入库审查开关决定是否调用 LLM 审查
                if config.enable_ingest_review:
                    # 2. 批量审查（带流式输出）
                    st.subheader("🔍 审查进度")
                    reviewer = get_document_reviewer()
                    chunk_texts = [c[0] for c in all_chunks]
                    batch_size = reviewer.batch_size if hasattr(reviewer, 'batch_size') else 20
                    all_results = batch_review_stream(reviewer, chunk_texts, batch_size=batch_size)
                else:
                    # 跳过 LLM 审查，全部标记为 pass
                    st.info("⚠️ 入库审查已关闭，所有语料将直接入库（未经安全审查）")
                    all_results = [
                        ChunkReviewResult(
                            chunk_id=i,
                            content=c[0],
                            risk_score=0.0,
                            risk_type=None,
                            reason="入库审查已关闭",
                        )
                        for i, c in enumerate(all_chunks)
                    ]

                # 3. 处理审查结果（分类、入库、更新队列）
                _process_review_results(all_chunks, all_results)

        # 显示审查结果
        results = st.session_state.upload_results
        if results:
            st.markdown("---")
            st.subheader("📊 审查结果")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("总块数", results["total_chunks"])
            c2.metric("✅ pass", results["pass_count"])
            c3.metric("⚠️ review", results["review_count"])
            c4.metric("❌ fail", results["fail_count"])

            if results["review_items"]:
                st.warning(
                    f"有 **{len(results['review_items'])}** 个语料块需人工审核，"
                    f"请前往 [📥 人工审核页面](pages/review.py) 处理。"
                )

            # 渲染格式化审查结果表格
            if results.get("results"):
                st.subheader("📋 详细审查结果")

                # 构建表格数据
                table_data = []
                for r in results["results"]:
                    info = STATUS_MAP.get(r["status"], STATUS_MAP["r"])
                    table_data.append({
                        "语料ID": r["chunk_id"],
                        "语料来源": r["source"],
                        "审查评级": info["label"],
                        "语料内容": r["content"][:150] + ("..." if len(r["content"]) > 150 else ""),
                        "原因": r.get("reason", ""),
                    })

                st.table(
                    __import__("pandas", fromlist=["DataFrame"]).DataFrame(table_data)
                )

                # 可展开的详细内容
                for r in results["results"]:
                    info = STATUS_MAP.get(r["status"], STATUS_MAP["r"])
                    with st.expander(
                        f"{info['label']} ID:{r['chunk_id']} | {r['source']}"
                    ):
                        st.text(r["content"])
                        st.caption(f"原因: {r.get('reason', '')}")

            if st.button("🧹 清除结果展示", key="clear_upload_results"):
                st.session_state.upload_results = None
                st.rerun()


# ============================================================
# Tab 2: Document list
# ============================================================
with tab_docs:
    st.subheader("已入库文档")

    if doc_count == 0:
        st.info("知识库为空，请上传语料")
    else:
        all_docs = vectorstore.vectorstore.get()
        ids = all_docs.get("ids", [])
        metadatas = all_docs.get("metadatas", [])
        documents = all_docs.get("documents", [])

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
            risk = meta.get("risk_score", None)

            status_icon = "✅"
            with st.expander(f"{status_icon} `{doc_id[:8]}` — {source}"):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"**ID:** `{doc_id}`")
                    st.markdown(f"**来源:** {source}")
                    if risk is not None:
                        st.markdown(f"**风险分数:** {risk:.2f}")
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
                vectorstore.clear()
            st.success("✅ 知识库已清空")
            st.rerun()
    else:
        st.info("知识库已经是空的，无需清空")
