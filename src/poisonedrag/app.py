"""
PoisonedRAG - 主页面（对话）

仅保留对话功能，支持多会话记忆。
导航到其他页面使用 st.switch_page。
"""

import os
import sys
from typing import Dict, Any
from pathlib import Path

import streamlit as st

# 加载 .env 文件
from dotenv import load_dotenv
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env")

src_dir = _project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from poisonedrag.config import get_config
from poisonedrag.embeddings import get_embedding_model
from poisonedrag.llm import get_llm
from poisonedrag.vectorstore import get_vectorstore
from poisonedrag.rag.retriever import create_retriever
from poisonedrag.rag.generator import create_generator
from poisonedrag.data.conversation_memory import (
    create_session,
    load_session,
    save_session,
    add_message,
    list_sessions,
    delete_session,
)
from poisonedrag.security.filter import get_content_filter
from poisonedrag.security.validator import get_response_validator


# ============================================================
# 初始化
# ============================================================

def init_session_state():
    """初始化会话状态"""
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 如果有会话 ID 但消息未加载，自动恢复消息
    if st.session_state.current_session_id and not st.session_state.messages:
        sess = load_session(st.session_state.current_session_id)
        if sess:
            st.session_state.messages = sess.get("messages", [])

    if "chat_app" not in st.session_state:
        st.session_state.chat_app = None


class ChatApp:
    """轻量级聊天应用（仅用于主页对话）"""

    def __init__(self):
        self.config = get_config()
        self._initialized = False
        self._components = {}

    def initialize(self):
        if self._initialized:
            return
        try:
            self._components["embedding_model"] = get_embedding_model()
            self._components["llm"] = get_llm()
            self._components["vectorstore"] = get_vectorstore(
                embedding_function=self._components["embedding_model"].embeddings
            )
            self._components["content_filter"] = get_content_filter(
                embedding_model=self._components["embedding_model"]
            )
            self._components["response_validator"] = get_response_validator(
                embedding_model=self._components["embedding_model"]
            )
            self._components["retriever"] = create_retriever(
                vectorstore=self._components["vectorstore"],
                content_filter=self._components["content_filter"],
            )
            self._components["generator"] = create_generator(
                llm=self._components["llm"],
                retriever=self._components["retriever"],
                response_validator=self._components["response_validator"],
            )
            self._initialized = True
        except Exception as e:
            st.error(f"初始化失败: {e}")
            raise

    @property
    def vectorstore(self):
        return self._components["vectorstore"]

    @property
    def llm(self):
        return self._components["llm"]

    @property
    def retriever(self):
        return self._components["retriever"]

    @property
    def response_validator(self):
        return self._components["response_validator"]

    @property
    def generator(self):
        return self._components["generator"]


# ============================================================
# 侧边栏 - 会话记忆
# ============================================================

def render_sidebar(app: ChatApp):
    """渲染侧边栏"""
    with st.sidebar:
        st.title("💬 PoisonedRAG")
        st.markdown("---")

        # === 导航 ===
        st.subheader("📌 导航")
        if st.button("📥 人工审核", use_container_width=True):
            st.switch_page("pages/review.py")
        if st.button("📚 知识库管理", use_container_width=True):
            st.switch_page("pages/knowledge_management.py")
        if st.button("⚙️ 设置", use_container_width=True):
            st.switch_page("pages/settings.py")

        st.markdown("---")

        # === 对话记忆 ===
        st.subheader("💾 对话记忆")

        # 新建对话按钮
        if st.button("＋ 新建对话", use_container_width=True, type="primary"):
            new_id = create_session("新对话")
            st.session_state.current_session_id = new_id
            st.session_state.messages = []
            st.rerun()

        # 会话列表
        sessions = list_sessions(limit=20)
        if not sessions:
            st.info("暂无历史对话")
        else:
            for sess in sessions:
                col1, col2 = st.columns([4, 1])
                with col1:
                    label = f"{sess['title']} ({sess['message_count']})"
                    if st.button(
                        label,
                        key=f"sess_{sess['id']}",
                        use_container_width=True,
                        type="primary" if sess["id"] == st.session_state.current_session_id else "secondary",
                    ):
                        load_chat_session(sess["id"])
                with col2:
                    if st.button("🗑️", key=f"del_{sess['id']}"):
                        delete_session(sess["id"])
                        if st.session_state.current_session_id == sess["id"]:
                            st.session_state.current_session_id = None
                            st.session_state.messages = []
                        st.rerun()

        # 当前会话信息
        if st.session_state.current_session_id:
            sess = load_session(st.session_state.current_session_id)
            if sess:
                st.caption(f"当前: {sess['title']}")


def load_chat_session(session_id: str):
    """加载指定会话到当前状态"""
    sess = load_session(session_id)
    if sess is None:
        st.error("会话不存在")
        return
    st.session_state.current_session_id = session_id
    st.session_state.messages = sess.get("messages", [])
    st.rerun()


# ============================================================
# 对话渲染
# ============================================================

def render_chat_message(message: Dict[str, Any]):
    """渲染单条聊天消息"""
    role = message["role"]
    content = message["content"]

    with st.chat_message(role):
        st.markdown(content)
        metadata = message.get("metadata", {})
        if metadata:
            if metadata.get("warnings"):
                with st.expander("⚠️ 安全警告", expanded=False):
                    for w in metadata["warnings"]:
                        st.warning(w)
            if metadata.get("documents"):
                with st.expander("📄 检索文档", expanded=False):
                    for i, doc in enumerate(metadata["documents"], 1):
                        st.markdown(f"**文档 {i}**")
                        st.text(doc.get("content", "")[:200] + "...")
            if "confidence" in metadata:
                c = metadata["confidence"]
                color = "green" if c > 0.7 else "orange" if c > 0.4 else "red"
                st.markdown(f"置信度: :{color}[{c:.2f}]")


def handle_user_input(app: ChatApp, prompt: str):
    """处理用户输入（真实流式输出 + 安全校验）"""
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # 构建对话历史（排除当前用户消息）
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]

            # 真实流式生成（先检索 + 生成 + 校验，校验通过后流式输出）
            # 1. 检索
            result = app.retriever.retrieve(prompt)
            context = app.retriever._format_documents(result.documents)

            # 2. 构建消息
            system_content = app.generator.system_prompt.format(context=context)
            messages = app.llm.create_messages(
                query=prompt,
                system_prompt=system_content,
                history=history,
            )

            # 3. 真实流式调用 LLM
            display = st.empty()
            full_response = ""
            for chunk in app.llm.llm.stream(messages):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if content:
                    full_response += content
                    display.markdown(full_response)

            # 4. 安全校验（流式完成后）
            validation_result = None
            if app.response_validator:
                validation_result = app.response_validator.validate(
                    query=prompt,
                    response=full_response,
                    documents=result.documents,
                )

            # 5. 如果校验失败，替换为安全警告
            is_safe = True
            warnings = list(result.warnings)
            confidence = 1.0
            if validation_result and not validation_result.is_safe:
                full_response = (
                    "⚠️ 检测到该回答可能存在安全风险或事实不一致，已为您拦截原始响应。"
                    "建议结合其他可靠来源验证该问题的答案。"
                )
                display.markdown(full_response)
                is_safe = False
                warnings.extend(validation_result.warnings)
                confidence = validation_result.confidence
            elif validation_result:
                confidence = validation_result.confidence

            metadata = {
                "warnings": warnings,
                "documents": [
                    {"content": d.page_content, "metadata": d.metadata}
                    for d in result.documents
                ],
                "confidence": confidence,
                "is_safe": is_safe,
            }

            if warnings:
                with st.expander("⚠️ 安全警告", expanded=False):
                    for w in warnings:
                        st.warning(w)
            if result.documents:
                with st.expander("📄 检索文档", expanded=False):
                    for i, doc in enumerate(result.documents, 1):
                        st.markdown(f"**文档 {i}**")
                        st.text(doc.page_content[:200] + "...")
            color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
            st.markdown(f"置信度: :{color}[{confidence:.2f}]")

            assistant_msg = {
                "role": "assistant",
                "content": full_response,
                "metadata": metadata,
            }
            st.session_state.messages.append(assistant_msg)

            # 持久化到本地文件
            if st.session_state.current_session_id:
                add_message(st.session_state.current_session_id, "user", prompt)
                add_message(
                    st.session_state.current_session_id,
                    "assistant",
                    full_response,
                    metadata,
                )
            else:
                sid = create_session(prompt[:30])
                st.session_state.current_session_id = sid
                add_message(sid, "user", prompt)
                add_message(sid, "assistant", full_response, metadata)

        except Exception as e:
            st.error(f"生成失败: {e}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"抱歉，发生了错误: {e}",
            })


# ============================================================
# 主函数
# ============================================================

def main():
    st.set_page_config(
        page_title="PoisonedRAG",
        page_icon="💬",
        layout="wide",
    )

    init_session_state()

    app = ChatApp()
    try:
        app.initialize()
    except Exception as e:
        st.error(f"系统初始化失败: {e}")
        st.stop()
    st.session_state.chat_app = app

    render_sidebar(app)

    # 主界面
    st.title("💬 安全对话")
    st.caption("基于知识库的 RAG 智能问答，三阶段防护：入库审查 → 检索过滤 → 生成校验")

    # 渲染历史消息
    for message in st.session_state.messages:
        render_chat_message(message)

    # 用户输入
    if prompt := st.chat_input("输入您的问题..."):
        handle_user_input(app, prompt)

    # 底部信息
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: gray;">PoisonedRAG - 毕业设计项目 | 安全防护 RAG 系统</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
