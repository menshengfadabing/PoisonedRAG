"""
人工审核页面 (Human Review Page)

提供待审核队列查看、文档审批/拒绝、审查日志查看等功能。
"""

import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path

import streamlit as st

# 添加 src 目录到 sys.path
project_root = Path(__file__).parent.parent  # PoisonedRAG/
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# 加载 .env 文件
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from langchain_core.documents import Document

from poisonedrag.config import get_config, reset_config
from poisonedrag.llm import get_review_llm
from poisonedrag.vectorstore import get_vectorstore
from poisonedrag.embeddings import get_embedding_model
from poisonedrag.resecurity import (
    DocumentReviewer,
    ReviewQueueManager,
    TextSplitter,
    create_document_reviewer,
    create_review_queue_manager,
    create_text_splitter,
    ChunkReviewResult,
    QueuedDocument,
    ReviewStatus,
)

# Page config
st.set_page_config(
    page_title="人工审核 - PoisonedRAG",
    page_icon="👤",
    layout="wide",
)


def init_session_state():
    """初始化会话状态"""
    if "review_threshold" not in st.session_state:
        st.session_state.review_threshold = get_config().review_risk_threshold

    if "review_queue_manager" not in st.session_state:
        st.session_state.review_queue_manager = create_review_queue_manager()

    if "document_reviewer" not in st.session_state:
        # 延迟初始化，避免不必要的 API 调用
        st.session_state.document_reviewer = None

    if "text_splitter" not in st.session_state:
        config = get_config()
        st.session_state.text_splitter = create_text_splitter(
            chunk_size=config.review_doc_chunk_size,
            chunk_overlap=config.review_doc_chunk_overlap,
        )


def get_document_reviewer() -> DocumentReviewer:
    """获取文档审查器（延迟初始化）"""
    if st.session_state.document_reviewer is None:
        with st.spinner("正在初始化审查模型..."):
            llm = get_review_llm()
            st.session_state.document_reviewer = create_document_reviewer(llm=llm.llm)
    return st.session_state.document_reviewer


def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.title("👤 人工审核")
        st.markdown("---")

        # 页面导航
        st.subheader("页面导航")
        if st.button("💬 对话主页", use_container_width=True):
            st.switch_page("app.py")
        if st.button("📚 知识库管理", use_container_width=True):
            st.switch_page("pages/knowledge_management.py")
        if st.button("⚙️ 设置", use_container_width=True):
            st.switch_page("pages/settings.py")

        st.markdown("---")

        # 审核设置
        st.subheader("审核设置")
        threshold = st.slider(
            "风险阈值",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.review_threshold,
            step=0.05,
            help="风险分数低于此阈值的文档将自动入库，高于此阈值的文档需要人工审核",
        )
        st.session_state.review_threshold = threshold

        # 阈值说明
        if threshold < 0.3:
            st.info("🔒 严格模式：大部分文档需要人工审核")
        elif threshold > 0.7:
            st.warning("⚠️ 宽松模式：只有高风险文档需要人工审核")
        else:
            st.info("⚖️ 平衡模式")

        st.markdown("---")

        # 统计信息
        st.subheader("审核统计")
        queue_manager = st.session_state.review_queue_manager
        stats = queue_manager.get_stats()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("待审核", stats["pending"])
            st.metric("已通过", stats["approved"] + stats["auto_approved"])
        with col2:
            st.metric("已拒绝", stats["rejected"])
            st.metric("总记录", stats["total"])


def render_pending_queue_tab():
    """渲染待审核队列标签页"""
    st.header("⚠️ 待审核队列")

    queue_manager = st.session_state.review_queue_manager
    pending_docs = queue_manager.get_pending_queue()

    if not pending_docs:
        st.success("✅ 没有待审核的文档")
        return

    st.info(f"共有 {len(pending_docs)} 篇文档待审核")

    # 显示每个待审核文档
    for doc in pending_docs:
        with st.container():
            # 文档卡片
            col1, col2 = st.columns([3, 1])

            with col1:
                # 风险分数颜色
                risk_color = (
                    "🔴" if doc.risk_score >= 0.7
                    else "🟡" if doc.risk_score >= 0.4
                    else "🟢"
                )
                st.markdown(f"**{risk_color} 风险分数: {doc.risk_score:.2f}**")
                st.caption(f"来源: {doc.source} | 风险类型: {doc.risk_type or '未知'}")

                if doc.reason:
                    st.warning(f"LLM 判断: {doc.reason}")

                # 内容预览
                with st.expander("查看内容"):
                    st.text(doc.content)

            with col2:
                # 操作按钮
                if st.button("✅ 通过", key=f"approve_{doc.id}"):
                    handle_approve(doc.id)

                if st.button("❌ 拒绝", key=f"reject_{doc.id}"):
                    handle_reject(doc.id)

                # 拒绝原因输入
                reject_reason = st.text_input(
                    "拒绝原因",
                    key=f"reason_{doc.id}",
                    placeholder="可选",
                )

            st.markdown("---")


def handle_approve(doc_id: str):
    """处理批准入库"""
    queue_manager = st.session_state.review_queue_manager
    queue_manager.approve(doc_id, operator="user")

    # 将文档添加到向量库
    docs = queue_manager.to_documents([doc_id])
    if docs:
        try:
            config = get_config()
            embedding_model = get_embedding_model()
            vectorstore = get_vectorstore(embedding_function=embedding_model.embeddings)
            vectorstore.add_documents(docs)
            st.success("✅ 文档已入库")
        except Exception as e:
            st.error(f"入库失败: {e}")

    st.rerun()


def handle_reject(doc_id: str):
    """处理拒绝入库"""
    queue_manager = st.session_state.review_queue_manager
    reason = st.session_state.get(f"reason_{doc_id}", "")
    queue_manager.reject(doc_id, operator="user", comment=reason)
    st.success("❌ 文档已拒绝")
    st.rerun()


def render_review_logs_tab():
    """渲染审查日志标签页"""
    st.header("📋 审查日志")

    queue_manager = st.session_state.review_queue_manager

    # 筛选
    action_filter = st.selectbox(
        "筛选类型",
        options=["全部", "auto_approved", "approved", "rejected"],
        format_func=lambda x: {
            "全部": "全部",
            "auto_approved": "自动通过",
            "approved": "人工通过",
            "rejected": "已拒绝",
        }.get(x, x),
    )

    # 获取日志
    logs = queue_manager.get_logs(
        action_filter=None if action_filter == "全部" else action_filter,
        limit=100,
    )

    if not logs:
        st.info("暂无审查日志")
        return

    # 显示日志
    for log in logs:
        action_color = {
            "auto_approved": "🟢",
            "approved": "✅",
            "rejected": "❌",
        }.get(log.action, "❓")

        action_text = {
            "auto_approved": "自动通过",
            "approved": "人工通过",
            "rejected": "已拒绝",
        }.get(log.action, log.action)

        with st.container():
            st.markdown(f"**{action_color} {action_text}**")
            st.caption(f"时间: {log.timestamp}")
            st.text(f"文档ID: {log.doc_id}")
            st.text(f"风险分数: {log.risk_score:.2f} | 类型: {log.risk_type or '无'}")

            if log.reason:
                st.warning(f"原因: {log.reason}")

            if log.comment:
                st.info(f"备注: {log.comment}")

            st.markdown("---")


def main():
    """主函数"""
    # 初始化会话状态
    init_session_state()

    # 渲染侧边栏
    render_sidebar()

    # 主界面 - 标签页
    tab1, tab2 = st.tabs([
        "⚠️ 待审核队列",
        "📋 审查日志",
    ])

    with tab1:
        render_pending_queue_tab()

    with tab2:
        render_review_logs_tab()


if __name__ == "__main__":
    main()
