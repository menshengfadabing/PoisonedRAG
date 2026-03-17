"""
知识库管理页面

提供文档上传、审查、入库等功能。
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

    if "upload_progress" not in st.session_state:
        st.session_state.upload_progress = {}

    if "review_results" not in st.session_state:
        st.session_state.review_results = []


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
        st.title("📚 知识库管理")
        st.markdown("---")

        # 风险阈值调节
        st.subheader("审查设置")
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
        st.subheader("统计信息")
        queue_manager = st.session_state.review_queue_manager
        stats = queue_manager.get_stats()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("待审核", stats["pending"])
            st.metric("已通过", stats["approved"] + stats["auto_approved"])
        with col2:
            st.metric("已拒绝", stats["rejected"])
            st.metric("总记录", stats["total"])

        st.markdown("---")

        # 向量库统计
        st.subheader("向量库状态")
        try:
            config = get_config()
            embedding_model = get_embedding_model()
            vectorstore = get_vectorstore(embedding_function=embedding_model.embeddings)
            doc_count = vectorstore.count()
            st.metric("文档数量", doc_count)
        except Exception as e:
            st.error(f"获取向量库状态失败: {e}")

        st.markdown("---")

        # 返回主页
        if st.button("🏠 返回主页", use_container_width=True):
            st.switch_page("app.py")


def render_upload_tab():
    """渲染上传语料标签页"""
    st.header("📤 上传文本语料")

    # 文件上传
    uploaded_files = st.file_uploader(
        "选择文件",
        type=["txt", "md", "json"],
        accept_multiple_files=True,
        help="支持 .txt, .md, .json 格式",
    )

    if not uploaded_files:
        st.info("请上传文本文件进行审查入库")
        return

    # 显示待上传文件
    st.subheader("待处理文件")
    for file in uploaded_files:
        st.text(f"📄 {file.name} ({file.size} 字节)")

    # 分割设置
    with st.expander("分割设置", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input(
                "块大小（字符）",
                min_value=100,
                max_value=2000,
                value=get_config().review_doc_chunk_size,
            )
        with col2:
            chunk_overlap = st.number_input(
                "块重叠（字符）",
                min_value=0,
                max_value=500,
                value=get_config().review_doc_chunk_overlap,
            )

        split_method = st.selectbox(
            "分割方式",
            options=["recursive", "character"],
            format_func=lambda x: "递归分割" if x == "recursive" else "字符分割",
        )

    # 更新分割器
    st.session_state.text_splitter = create_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        split_method=split_method,
    )

    # 开始审查按钮
    if st.button("🔍 开始审查", type="primary"):
        process_uploaded_files(uploaded_files)


def process_uploaded_files(uploaded_files):
    """处理上传的文件"""
    reviewer = get_document_reviewer()
    splitter = st.session_state.text_splitter
    queue_manager = st.session_state.review_queue_manager
    threshold = st.session_state.review_threshold

    # 获取向量库实例
    config = get_config()
    embedding_model = get_embedding_model()
    vectorstore = get_vectorstore(embedding_function=embedding_model.embeddings)

    progress_bar = st.progress(0)
    status_text = st.empty()

    total_chunks = 0
    auto_approved = 0
    need_review = 0

    # 收集自动通过的文档ID，用于批量入库
    auto_approved_ids = []

    for i, file in enumerate(uploaded_files):
        status_text.text(f"正在处理: {file.name}")

        # 读取文件内容
        content = file.read().decode("utf-8")

        # 分割文本
        chunks = splitter.split_text(content)
        total_chunks += len(chunks)

        # 批量审查
        if chunks:
            result = reviewer.review_chunks(chunks)

            if result.success:
                for chunk_result in result.results:
                    # 添加到队列
                    queued_doc = queue_manager.add_to_queue(
                        chunk_result,
                        source=file.name,
                    )

                    # 根据阈值决定是否自动通过
                    if chunk_result.risk_score < threshold:
                        queue_manager.auto_approve(queued_doc.id)
                        auto_approved_ids.append(queued_doc.id)
                        auto_approved += 1
                    else:
                        need_review += 1

        # 更新进度
        progress = (i + 1) / len(uploaded_files)
        progress_bar.progress(progress)

    # 批量将自动通过的文档入库
    if auto_approved_ids:
        status_text.text(f"正在入库 {len(auto_approved_ids)} 个文档...")
        docs = queue_manager.to_documents(auto_approved_ids)
        if docs:
            try:
                vectorstore.add_documents(docs)
            except Exception as e:
                st.error(f"自动入库失败: {e}")

    progress_bar.empty()
    status_text.empty()

    # 显示结果
    st.success(f"处理完成！共 {total_chunks} 个文档块")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("自动入库", auto_approved)
    with col2:
        st.metric("待人工审核", need_review)

    if need_review > 0:
        st.info("请前往「待审核队列」进行人工审核")


def render_queue_tab():
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


def render_knowledge_base_tab():
    """渲染知识库管理标签页"""
    st.header("📚 知识库管理")

    try:
        config = get_config()
        embedding_model = get_embedding_model()
        vectorstore = get_vectorstore(embedding_function=embedding_model.embeddings)

        # 统计信息
        doc_count = vectorstore.count()
        st.metric("文档总数", doc_count)

        if doc_count == 0:
            st.info("知识库为空，请上传文档")
            return

        # 文档列表
        st.subheader("文档列表")

        # 获取所有文档
        all_docs = vectorstore.vectorstore.get()

        # 分页显示
        page_size = 20
        total_pages = (len(all_docs["ids"]) + page_size - 1) // page_size
        page = st.number_input("页码", min_value=1, max_value=max(total_pages, 1), value=1)

        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, len(all_docs["ids"]))

        for i in range(start_idx, end_idx):
            doc_id = all_docs["ids"][i]
            metadata = all_docs["metadatas"][i] if all_docs["metadatas"] else {}
            content = all_docs["documents"][i] if all_docs["documents"] else ""

            source = metadata.get("source", "未知")
            doc_type = metadata.get("type", "未知")

            with st.expander(f"[{doc_type}] {source[:30]}... (ID: {doc_id[:8]})"):
                st.text(content[:500] + "..." if len(content) > 500 else content)

        # 操作按钮
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 导出知识库", use_container_width=True):
                st.info("功能开发中...")

        with col2:
            if st.button("🗑️ 清空知识库", use_container_width=True):
                if st.checkbox("确认清空"):
                    vectorstore.clear()
                    st.success("知识库已清空")
                    st.rerun()

    except Exception as e:
        st.error(f"获取知识库失败: {e}")


def render_logs_tab():
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
    # 页面配置
    st.set_page_config(
        page_title="知识库管理 - PoisonedRAG",
        page_icon="📚",
        layout="wide",
    )

    # 初始化会话状态
    init_session_state()

    # 渲染侧边栏
    render_sidebar()

    # 主界面 - 标签页
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 上传语料",
        "⚠️ 待审核队列",
        "📚 知识库管理",
        "📋 审查日志",
    ])

    with tab1:
        render_upload_tab()

    with tab2:
        render_queue_tab()

    with tab3:
        render_knowledge_base_tab()

    with tab4:
        render_logs_tab()


if __name__ == "__main__":
    main()