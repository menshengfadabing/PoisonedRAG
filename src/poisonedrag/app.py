"""
Streamlit 前端应用

提供交互式界面，支持对话和防护模式配置。
"""

import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path

import streamlit as st

# 加载 .env 文件（LangSmith 配置等）
# 必须在导入 langchain 之前设置环境变量
from dotenv import load_dotenv
_project_root = Path(__file__).parent.parent.parent  # PoisonedRAG/
load_dotenv(_project_root / ".env")

# 添加 src 目录到 sys.path
# __file__ = .../PoisonedRAG/src/poisonedrag/app.py
# 向两级到达 src 目录
src_dir = os.path.dirname(os.path.dirname(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from poisonedrag.config import get_config, Config, ProtectionMode, apply_protection_mode
from poisonedrag.embeddings import EmbeddingModel, get_embedding_model
from poisonedrag.llm import LLMModel, get_llm
from poisonedrag.vectorstore import VectorStore, get_vectorstore
from poisonedrag.rag.retriever import Retriever, create_retriever
from poisonedrag.rag.generator import Generator, create_generator
from poisonedrag.security.filter import ContentFilter, get_content_filter
from poisonedrag.security.validator import ResponseValidator, get_response_validator
from poisonedrag.data.knowledge_base import KnowledgeBase, create_knowledge_base


class ChatApp:
    """
    聊天应用类

    封装 RAG 系统的核心组件，提供统一的接口。
    """

    def __init__(self):
        """初始化聊天应用"""
        self.config = get_config()
        self._initialized = False
        self._components: Dict[str, Any] = {}

    def initialize(self):
        """初始化所有组件"""
        if self._initialized:
            return

        try:
            # 初始化嵌入模型
            self._components["embedding_model"] = get_embedding_model()

            # 初始化 LLM
            self._components["llm"] = get_llm()

            # 初始化向量存储
            self._components["vectorstore"] = get_vectorstore(
                embedding_function=self._components["embedding_model"].embeddings
            )

            # 初始化过滤器
            self._components["content_filter"] = get_content_filter(
                embedding_model=self._components["embedding_model"]
            )

            # 初始化校验器
            self._components["response_validator"] = get_response_validator(
                embedding_model=self._components["embedding_model"]
            )

            # 初始化检索器
            self._components["retriever"] = create_retriever(
                vectorstore=self._components["vectorstore"],
                content_filter=self._components["content_filter"],
            )

            # 初始化生成器
            self._components["generator"] = create_generator(
                llm=self._components["llm"],
                retriever=self._components["retriever"],
                response_validator=self._components["response_validator"],
            )

            # 初始化知识库
            self._components["knowledge_base"] = create_knowledge_base(
                vectorstore=self._components["vectorstore"],
            )

            self._initialized = True

        except Exception as e:
            st.error(f"初始化失败: {e}")
            raise

    @property
    def embedding_model(self) -> EmbeddingModel:
        return self._components["embedding_model"]

    @property
    def llm(self) -> LLMModel:
        return self._components["llm"]

    @property
    def vectorstore(self) -> VectorStore:
        return self._components["vectorstore"]

    @property
    def content_filter(self) -> ContentFilter:
        return self._components["content_filter"]

    @property
    def response_validator(self) -> ResponseValidator:
        return self._components["response_validator"]

    @property
    def retriever(self) -> Retriever:
        return self._components["retriever"]

    @property
    def generator(self) -> Generator:
        return self._components["generator"]

    @property
    def knowledge_base(self) -> KnowledgeBase:
        return self._components["knowledge_base"]


def init_session_state():
    """初始化 Streamlit 会话状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_app" not in st.session_state:
        st.session_state.chat_app = ChatApp()

    # 三阶段独立开关
    if "enable_ingest_review" not in st.session_state:
        st.session_state.enable_ingest_review = True

    if "enable_retrieval_filter" not in st.session_state:
        st.session_state.enable_retrieval_filter = False

    if "enable_generation_validator" not in st.session_state:
        st.session_state.enable_generation_validator = True


def render_sidebar(app: ChatApp):
    """渲染侧边栏"""
    with st.sidebar:
        st.title("🛡️ PoisonedRAG")
        st.markdown("---")

        # === 防护模式设置 ===
        st.subheader("防护模式设置")

        # 防护模式预设
        mode_names = {
            "strict": "🔒 严格模式 (三阶段全开)",
            "standard": "✅ 标准模式 (入库+生成)",
            "performance": "⚡ 性能优先 (仅入库)",
            "development": "🔧 开发模式 (仅生成)",
            "disabled": "❌ 禁用防护",
        }

        current_mode = app.config.protection_mode or "standard"
        selected_mode = st.selectbox(
            "选择防护模式",
            options=list(mode_names.keys()),
            index=list(mode_names.keys()).index(current_mode),
            format_func=lambda x: mode_names.get(x, x),
            help="预设模式会自动配置三阶段开关",
        )

        # 应用模式
        if selected_mode != current_mode:
            app.config.set_protection_mode(selected_mode)
            # 同步到 session state
            mode_config = apply_protection_mode(ProtectionMode(selected_mode))
            st.session_state.enable_ingest_review = mode_config["enable_ingest_review"]
            st.session_state.enable_retrieval_filter = mode_config["enable_retrieval_filter"]
            st.session_state.enable_generation_validator = mode_config["enable_generation_validator"]
            st.rerun()

        st.markdown("---")

        # === 三阶段独立开关 ===
        st.subheader("三阶段防护开关")

        # 入库阶段
        st.session_state.enable_ingest_review = st.toggle(
            "📥 入库审查",
            value=st.session_state.enable_ingest_review,
            help="入库阶段：使用 LLM 对文档进行安全审查，识别潜在风险内容",
        )

        # 检索阶段
        st.session_state.enable_retrieval_filter = st.toggle(
            "🔍 检索过滤",
            value=st.session_state.enable_retrieval_filter,
            help="检索阶段：关键词过滤、语义异常检测（效果有限，可选）",
        )

        # 生成阶段
        st.session_state.enable_generation_validator = st.toggle(
            "📤 生成校验",
            value=st.session_state.enable_generation_validator,
            help="生成阶段：响应安全校验、事实一致性检查",
        )

        # 更新配置
        app.config.enable_ingest_review = st.session_state.enable_ingest_review
        app.config.enable_retrieval_filter = st.session_state.enable_retrieval_filter
        app.config.enable_generation_validator = st.session_state.enable_generation_validator

        # 显示当前防护状态
        active_stages = []
        if st.session_state.enable_ingest_review:
            active_stages.append("入库")
        if st.session_state.enable_retrieval_filter:
            active_stages.append("检索")
        if st.session_state.enable_generation_validator:
            active_stages.append("生成")

        if active_stages:
            st.success(f"🛡️ 已启用: {' → '.join(active_stages)}")
        else:
            st.warning("⚠️ 所有防护已禁用")

        st.markdown("---")

        # 知识库管理
        st.subheader("知识库管理")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("加载知识库", use_container_width=True):
                with st.spinner("加载中..."):
                    try:
                        docs = app.knowledge_base.load_from_directory()
                        count = app.knowledge_base.index_to_vectorstore(docs)
                        st.success(f"已加载 {count} 条文档")
                    except Exception as e:
                        st.error(f"加载失败: {e}")

        with col2:
            if st.button("清空知识库", use_container_width=True):
                app.vectorstore.clear()
                st.success("知识库已清空")

        # 显示知识库统计
        stats = app.knowledge_base.get_stats()
        st.info(f"📚 文档数量: {stats['vectorstore_count']}")

        st.markdown("---")

        # 跳转知识库管理
        if st.button("📚 知识库管理", use_container_width=True):
            st.switch_page("pages/knowledge_management.py")


def render_chat_message(message: Dict[str, Any]):
    """渲染单条聊天消息"""
    role = message["role"]
    content = message["content"]

    with st.chat_message(role):
        st.markdown(content)

        # 显示元信息
        if "metadata" in message and message["metadata"]:
            metadata = message["metadata"]

            if metadata.get("warnings"):
                with st.expander("⚠️ 安全警告", expanded=False):
                    for warning in metadata["warnings"]:
                        st.warning(warning)

            if metadata.get("documents"):
                with st.expander("📄 检索文档", expanded=False):
                    for i, doc in enumerate(metadata["documents"], 1):
                        st.markdown(f"**文档 {i}**")
                        st.text(doc.get("content", "")[:200] + "...")

            if "confidence" in metadata:
                confidence = metadata["confidence"]
                color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                st.markdown(f"置信度: :{color}[{confidence:.2f}]")


def handle_user_input(app: ChatApp, prompt: str):
    """处理用户输入"""
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回复
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                # 根据防护设置调整组件
                protection_enabled = (
                    st.session_state.enable_ingest_review or
                    st.session_state.enable_retrieval_filter or
                    st.session_state.enable_generation_validator
                )

                if protection_enabled:
                    retriever = create_retriever(
                        vectorstore=app.vectorstore,
                        content_filter=app.content_filter,
                    )
                    generator = create_generator(
                        llm=app.llm,
                        retriever=retriever,
                        response_validator=app.response_validator,
                    )
                else:
                    retriever = create_retriever(
                        vectorstore=app.vectorstore,
                        content_filter=None,  # 禁用过滤
                    )
                    generator = create_generator(
                        llm=app.llm,
                        retriever=retriever,
                        response_validator=None,  # 禁用校验
                    )

                # 生成响应
                result = generator.generate(prompt)

                # 显示响应
                st.markdown(result.response)

                # 显示元信息
                metadata = {
                    "warnings": result.warnings,
                    "documents": [
                        {"content": doc.page_content, "metadata": doc.metadata}
                        for doc in result.documents
                    ],
                    "confidence": result.confidence,
                    "is_safe": result.is_safe,
                }

                if result.warnings:
                    with st.expander("⚠️ 安全警告", expanded=False):
                        for warning in result.warnings:
                            st.warning(warning)

                if result.documents:
                    with st.expander("📄 检索文档", expanded=False):
                        for i, doc in enumerate(result.documents, 1):
                            st.markdown(f"**文档 {i}**")
                            st.text(doc.page_content[:200] + "...")

                # 置信度显示
                confidence = result.confidence
                color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                st.markdown(f"置信度: :{color}[{confidence:.2f}]")

                # 保存助手消息
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.response,
                    "metadata": metadata,
                })

            except Exception as e:
                st.error(f"生成失败: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"抱歉，发生了错误: {e}",
                })


def main():
    """主函数"""
    # 页面配置
    st.set_page_config(
        page_title="PoisonedRAG",
        page_icon="🛡️",
        layout="wide",
    )

    # 初始化会话状态
    init_session_state()

    # 获取应用实例
    app = st.session_state.chat_app

    # 初始化组件
    try:
        app.initialize()
    except Exception as e:
        st.error(f"系统初始化失败: {e}")
        st.stop()

    # 渲染侧边栏
    render_sidebar(app)

    # 主界面
    st.title("🛡️ PoisonedRAG 安全对话系统")
    st.markdown("""
    这是一个具有安全防护功能的 RAG 对话系统。
    - 基于知识库进行智能问答
    - 三阶段防护：入库审查 → 检索过滤 → 生成校验
    - 在侧边栏配置防护模式和知识库
    """)

    # 渲染历史消息
    for message in st.session_state.messages:
        render_chat_message(message)

    # 用户输入
    if prompt := st.chat_input("输入您的问题..."):
        handle_user_input(app, prompt)

    # 底部信息
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray;">
        PoisonedRAG - 毕业设计项目 | 安全防护 RAG 系统
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()