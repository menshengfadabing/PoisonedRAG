"""
设置页面 (Settings Page)

提供 API 配置查看、连通性检测、审核策略管理等功能。
"""

import sys
import time
from pathlib import Path

import streamlit as st

from dotenv import load_dotenv
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env")

src_dir = _project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from poisonedrag.config import get_config, ProtectionMode
from poisonedrag.embeddings import get_embedding_model
from poisonedrag.llm import get_llm
from poisonedrag.vectorstore import get_vectorstore


def init_session_state():
    """初始化会话状态中的设置相关字段"""
    if "protection_mode" not in st.session_state:
        config = get_config()
        st.session_state.protection_mode = config.protection_mode or "standard"

    if "enable_ingest_review" not in st.session_state:
        config = get_config()
        st.session_state.enable_ingest_review = config.enable_ingest_review

    if "enable_retrieval_filter" not in st.session_state:
        config = get_config()
        st.session_state.enable_retrieval_filter = config.enable_retrieval_filter

    if "enable_generation_validator" not in st.session_state:
        config = get_config()
        st.session_state.enable_generation_validator = config.enable_generation_validator

    if "risk_threshold" not in st.session_state:
        config = get_config()
        st.session_state.risk_threshold = config.ingest_review_config["risk_threshold"]

    if "review_min_batch_size" not in st.session_state:
        config = get_config()
        st.session_state.review_min_batch_size = config.review_min_batch_size

    if "review_max_batch_size" not in st.session_state:
        config = get_config()
        st.session_state.review_max_batch_size = config.review_max_batch_size


def render_sidebar():
    """渲染侧边栏导航"""
    with st.sidebar:
        st.title("⚙️ 设置")
        st.markdown("---")
        st.subheader("页面导航")
        if st.button("💬 对话主页", use_container_width=True):
            st.switch_page("app.py")
        if st.button("📥 人工审核", use_container_width=True):
            st.switch_page("pages/review.py")
        if st.button("📚 知识库管理", use_container_width=True):
            st.switch_page("pages/knowledge_management.py")


def render_api_config_tab():
    """Tab 1: API 配置（支持前端编辑保存）"""
    st.header("🔑 API 配置")
    st.markdown("可直接在此修改 API 密钥，保存后写入 `.env` 文件并刷新页面生效。")
    st.markdown("---")

    config = get_config()
    _env_path = _project_root / ".env"

    # 初始化 session state 中的编辑值
    if "_edit_deepseek_base" not in st.session_state:
        st.session_state._edit_deepseek_base = config.deepseek_base_url
    if "_edit_deepseek_model" not in st.session_state:
        st.session_state._edit_deepseek_model = config.deepseek_model
    if "_edit_deepseek_key" not in st.session_state:
        st.session_state._edit_deepseek_key = config.deepseek_api_key
    if "_edit_dashscope_key" not in st.session_state:
        st.session_state._edit_dashscope_key = config.dashscope_api_key
    if "_edit_embedding_provider" not in st.session_state:
        st.session_state._edit_embedding_provider = config.embedding_provider

    # LLM 配置
    st.subheader("LLM 配置")
    st.session_state._edit_deepseek_base = st.text_input(
        "DEEPSEEK_API_BASE",
        value=st.session_state._edit_deepseek_base,
        help="API 基础地址，如 DeepSeek 官方或 Volcengine Ark",
    )
    st.session_state._edit_deepseek_model = st.text_input(
        "DEEPSEEK_MODEL",
        value=st.session_state._edit_deepseek_model,
        help="LLM 模型名称",
    )
    st.session_state._edit_deepseek_key = st.text_input(
        "DEEPSEEK_API_KEY",
        value=st.session_state._edit_deepseek_key,
        type="password",
        help="LLM API 密钥",
    )

    st.markdown("---")

    # Embedding 配置
    st.subheader("Embedding 配置")
    st.session_state._edit_embedding_provider = st.selectbox(
        "EMBEDDING_PROVIDER",
        options=["dashscope", "ollama"],
        index=0 if st.session_state._edit_embedding_provider == "dashscope" else 1,
        help="嵌入模型提供商",
    )
    st.session_state._edit_dashscope_key = st.text_input(
        "DASH_SCOPE_API_KEY",
        value=st.session_state._edit_dashscope_key,
        type="password",
        help="DashScope API 密钥（阿里云）",
    )

    st.markdown("---")

    # 保存按钮
    if st.button("💾 保存配置到 .env", type="primary", use_container_width=True):
        if _update_env_file(_env_path):
            st.success("✅ 配置已保存到 .env 文件，页面将在 2 秒后自动刷新...")
            st.rerun()
        else:
            st.error("❌ 保存失败，请检查文件权限")


def _update_env_file(env_path: Path) -> bool:
    """
    更新 .env 文件中的配置值

    Args:
        env_path: .env 文件路径

    Returns:
        是否保存成功
    """
    try:
        # 读取现有 .env 文件
        if env_path.exists():
            lines = env_path.read_text(encoding="utf-8").splitlines()
        else:
            lines = []

        # 构建需要更新的键值对
        updates = {
            "DEEPSEEK_API_BASE": st.session_state._edit_deepseek_base,
            "DEEPSEEK_MODEL": st.session_state._edit_deepseek_model,
            "DEEPSEEK_API_KEY": st.session_state._edit_deepseek_key,
            "EMBEDDING_PROVIDER": st.session_state._edit_embedding_provider,
            "DASH_SCOPE_API_KEY": st.session_state._edit_dashscope_key,
        }

        # 更新已存在的行
        new_lines = []
        updated_keys = set()
        for line in lines:
            stripped = line.strip()
            matched = False
            for key, value in updates.items():
                if stripped.startswith(f"{key}=") and not stripped.startswith("#"):
                    new_lines.append(f"{key}={value}")
                    updated_keys.add(key)
                    matched = True
                    break
            if not matched:
                new_lines.append(line)

        # 添加尚未存在的键
        for key, value in updates.items():
            if key not in updated_keys:
                new_lines.append(f"{key}={value}")

        # 写回文件
        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

        # 重新加载环境变量
        import importlib
        from dotenv import load_dotenv
        load_dotenv(env_path, override=True)

        return True

    except Exception as e:
        st.error(f"保存失败: {e}")
        return False


def render_connectivity_tab():
    """Tab 2: 连通性检测"""
    st.header("🔗 连通性检测")
    st.markdown("测试各组件的连接是否正常，包括响应时间。")
    st.markdown("---")

    # LLM 连接测试
    st.subheader("LLM 连接测试")
    if st.button("测试 LLM 连接", use_container_width=True, key="test_llm"):
        with st.spinner("正在测试 LLM 连接..."):
            try:
                start_time = time.time()
                llm = get_llm()
                response = llm.llm.invoke("你好")
                elapsed_ms = (time.time() - start_time) * 1000

                st.success(f"✅ LLM 连接成功（耗时: {elapsed_ms:.0f} ms）")
                st.text_area("响应内容", str(response), height=100)
            except Exception as e:
                st.error(f"❌ LLM 连接失败: {e}")

    st.markdown("---")

    # Embedding 连接测试
    st.subheader("Embedding 连接测试")
    if st.button("测试 Embedding 连接", use_container_width=True, key="test_embedding"):
        with st.spinner("正在测试 Embedding 连接..."):
            try:
                start_time = time.time()
                embedding_model = get_embedding_model()
                embedding = embedding_model.embed_query("你好")
                elapsed_ms = (time.time() - start_time) * 1000

                st.success(
                    f"✅ Embedding 连接成功（耗时: {elapsed_ms:.0f} ms，维度: {len(embedding)}）"
                )
            except Exception as e:
                st.error(f"❌ Embedding 连接失败: {e}")

    st.markdown("---")

    # 向量库连接测试
    st.subheader("向量库连接测试")
    if st.button("测试向量库连接", use_container_width=True, key="test_vectorstore"):
        with st.spinner("正在测试向量库连接..."):
            try:
                start_time = time.time()
                embedding_model = get_embedding_model()
                vectorstore = get_vectorstore(embedding_function=embedding_model.embeddings)
                doc_count = vectorstore.count()
                elapsed_ms = (time.time() - start_time) * 1000

                st.success(
                    f"✅ 向量库连接成功（耗时: {elapsed_ms:.0f} ms，文档数: {doc_count}）"
                )
            except Exception as e:
                st.error(f"❌ 向量库连接失败: {e}")


def render_review_strategy_tab():
    """Tab 3: 审核策略"""
    st.header("🛡️ 审核策略")
    st.markdown("配置防护模式和各阶段的审查开关。")
    st.markdown("---")

    # 防护模式选择
    st.subheader("防护模式预设")
    mode_descriptions = {
        "strict": "🔒 最严格：入库审查 + 检索过滤 + 生成校验（三阶段全开）",
        "standard": "⚖️ 标准模式：入库审查 + 生成校验（推荐）",
        "performance": "⚡ 性能优先：仅入库审查",
        "development": "🧪 开发测试：仅生成校验",
        "disabled": "❌ 关闭所有防护",
    }

    selected_mode = st.selectbox(
        "选择防护模式",
        options=[mode.value for mode in ProtectionMode],
        format_func=lambda x: mode_descriptions.get(x, x),
        index=[mode.value for mode in ProtectionMode].index(
            st.session_state.protection_mode
        )
        if st.session_state.protection_mode
        in [mode.value for mode in ProtectionMode]
        else 1,
        key="mode_selector",
    )

    st.session_state.protection_mode = selected_mode

    # 显示当前模式描述
    st.info(mode_descriptions.get(selected_mode, ""))

    st.markdown("---")

    # 独立开关
    st.subheader("独立控制开关")
    st.session_state.enable_ingest_review = st.checkbox(
        "入库审查（Ingest Review）",
        value=st.session_state.enable_ingest_review,
        help="在文档入库前进行 LLM 安全审查",
    )
    st.session_state.enable_retrieval_filter = st.checkbox(
        "检索过滤（Retrieval Filter）",
        value=st.session_state.enable_retrieval_filter,
        help="检索时过滤可疑或低可信度的文档",
    )
    st.session_state.enable_generation_validator = st.checkbox(
        "生成校验（Generation Validator）",
        value=st.session_state.enable_generation_validator,
        help="对 LLM 生成的响应进行安全校验",
    )

    st.markdown("---")

    # 风险阈值
    st.subheader("风险阈值")
    st.session_state.risk_threshold = st.slider(
        "风险阈值（Risk Threshold）",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.risk_threshold,
        step=0.05,
        help="风险分数低于此阈值的文档将自动通过，高于此阈值的需要人工审核",
    )

    # 阈值说明
    threshold = st.session_state.risk_threshold
    if threshold < 0.3:
        st.info("🔒 严格模式：大部分文档需要人工审核")
    elif threshold > 0.7:
        st.warning("⚠️ 宽松模式：只有高风险文档需要人工审核")
    else:
        st.info("⚖️ 平衡模式")

    st.markdown("---")

    # 批次大小配置
    st.subheader("审查批次配置")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.review_min_batch_size = st.number_input(
            "最小批次大小",
            min_value=1,
            max_value=st.session_state.review_max_batch_size,
            value=st.session_state.review_min_batch_size,
            help="每次审查的最小文档数量",
        )
    with col2:
        st.session_state.review_max_batch_size = st.number_input(
            "最大批次大小",
            min_value=st.session_state.review_min_batch_size,
            max_value=1000,
            value=st.session_state.review_max_batch_size,
            help="每次审查的最大文档数量",
        )

    st.markdown("---")

    # 保存按钮
    if st.button("💾 保存设置", type="primary", use_container_width=True):
        handle_save_settings()


def handle_save_settings():
    """保存设置到会话状态"""
    config = get_config()

    # 更新防护模式
    config.set_protection_mode(st.session_state.protection_mode)

    # 更新独立开关（确保不受模式覆盖）
    config.enable_ingest_review = st.session_state.enable_ingest_review
    config.enable_retrieval_filter = st.session_state.enable_retrieval_filter
    config.enable_generation_validator = st.session_state.enable_generation_validator

    # 更新风险阈值
    config.ingest_review_config["risk_threshold"] = st.session_state.risk_threshold
    config.review_risk_threshold = st.session_state.risk_threshold

    # 更新批次配置
    config.review_min_batch_size = st.session_state.review_min_batch_size
    config.review_max_batch_size = st.session_state.review_max_batch_size

    st.success("✅ 设置已保存到会话状态（当前会话有效）")


def main():
    """主函数"""
    st.set_page_config(
        page_title="设置 - PoisonedRAG",
        page_icon="⚙️",
        layout="wide",
    )

    # 初始化会话状态
    init_session_state()

    # 渲染侧边栏
    render_sidebar()

    # 主界面标题
    st.title("⚙️ 设置")

    # 标签页
    tab1, tab2, tab3 = st.tabs([
        "🔑 API 配置",
        "🔗 连通性检测",
        "🛡️ 审核策略",
    ])

    with tab1:
        render_api_config_tab()

    with tab2:
        render_connectivity_tab()

    with tab3:
        render_review_strategy_tab()


if __name__ == "__main__":
    main()
