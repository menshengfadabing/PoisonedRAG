"""
配置管理模块

管理 API 密钥、模型参数、向量数据库路径等配置项。
使用环境变量和默认值进行配置。
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")


class ProtectionMode(Enum):
    """防护模式预设"""
    STRICT = "strict"           # 最严格：三阶段全开
    STANDARD = "standard"       # 标准模式：入库 + 生成（推荐）
    PERFORMANCE = "performance" # 性能优先：仅入库审查
    DEVELOPMENT = "development" # 开发测试：仅生成校验
    DISABLED = "disabled"       # 关闭所有防护


def apply_protection_mode(mode: ProtectionMode) -> dict:
    """根据防护模式返回开关配置"""
    configs = {
        ProtectionMode.STRICT: {
            "enable_ingest_review": True,
            "enable_retrieval_filter": True,
            "enable_generation_validator": True,
        },
        ProtectionMode.STANDARD: {
            "enable_ingest_review": True,
            "enable_retrieval_filter": False,
            "enable_generation_validator": True,
        },
        ProtectionMode.PERFORMANCE: {
            "enable_ingest_review": True,
            "enable_retrieval_filter": False,
            "enable_generation_validator": False,
        },
        ProtectionMode.DEVELOPMENT: {
            "enable_ingest_review": False,
            "enable_retrieval_filter": False,
            "enable_generation_validator": True,
        },
        ProtectionMode.DISABLED: {
            "enable_ingest_review": False,
            "enable_retrieval_filter": False,
            "enable_generation_validator": False,
        },
    }
    return configs[mode]


@dataclass
class Config:
    """
    全局配置类

    包含所有系统配置项，支持从环境变量加载配置。
    """

    # LLM API 配置（支持 DeepSeek / Volcengine Ark 等 OpenAI 兼容接口）
    deepseek_api_key: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "")
    )
    deepseek_base_url: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    )
    deepseek_model: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    )

    # 文档审查专用 API 配置（使用独立的 API Key，未设置时复用主 Key）
    review_api_key: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_REVIEW_API_KEY", "") or os.getenv("DEEPSEEK_API_KEY", "")
    )
    review_base_url: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    )
    review_model: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_REVIEW_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    )
    review_risk_threshold: float = 0.5  # 风险阈值

    # ============================================================
    # 批量审查配置（用户可调参数）
    # ============================================================

    # --- 上下文窗口配置 ---
    llm_context_window: int = 128000    # LLM 上下文窗口大小（tokens），根据 API 文档设置

    # --- 文档处理配置 ---
    review_doc_max_length: int = 800    # 每个文档最大字符数（超出截断）
    review_doc_chunk_size: int = 500    # 文档分割块大小（字符数）
    review_doc_chunk_overlap: int = 50  # 文档分割块重叠（字符数）

    # --- Token 估算配置 ---
    # 中文约 1.5 字符/token，英文约 4 字符/token，混合取中间值
    review_chars_per_token: float = 2.0     # 每个 token 约等于多少字符
    review_system_prompt_tokens: int = 600  # 系统提示词 token 数
    review_user_prompt_overhead: int = 200  # 用户提示词固定开销（"请审查以下文档..."等）
    review_output_per_doc_tokens: int = 80  # 每个文档输出 token 数（JSON 结果）
    review_safety_margin: float = 0.9       # 安全边际系数（使用 90% 的窗口）

    # --- 批量处理配置 ---
    review_min_batch_size: int = 10     # 最小批次大小
    review_max_batch_size: int = 100    # 最大批次大小
    review_max_workers: int = 4         # 并行处理线程数

    # --- 功能开关 ---
    review_dynamic_batch: bool = True   # 是否启用动态批次计算

    # ============================================================
    # 嵌入模型配置（支持多种 Provider：dashscope / ollama）
    # ============================================================

    # Provider 选择：dashscope（API）或 ollama（本地）
    embedding_provider: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "dashscope")
    )

    # DashScope API 配置（阿里云）
    dashscope_api_key: str = field(
        default_factory=lambda: os.getenv("DASH_SCOPE_API_KEY", "")
    )
    dashscope_embedding_model: str = field(
        default_factory=lambda: os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3")
    )

    # Ollama 嵌入模型配置（本地）
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_embedding_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    )

    # 向量数据库配置
    chroma_persist_directory: str = field(
        default_factory=lambda: os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            # config.py -> poisonedrag/ -> src/ -> PoisonedRAG/
            "data",
            "chroma"
        )
    )
    chroma_collection_name: str = "poisonedrag_knowledge"

    # 检索配置
    retriever_top_k: int = 5
    retriever_score_threshold: float = 0.5

    # === 三阶段审查开关（独立控制）===
    # 入库阶段：LLM 文档安全审查
    enable_ingest_review: bool = True

    # 检索阶段：内容过滤器（关键词/语义异常）
    enable_retrieval_filter: bool = False  # 默认关闭，效果有限

    # 生成阶段：响应校验器（一致性检查）
    enable_generation_validator: bool = True

    # 防护模式（可选，用于快速设置三阶段开关）
    protection_mode: Optional[str] = "standard"  # strict, standard, performance, development, disabled

    # === 各阶段详细配置 ===

    # 入库审查配置
    ingest_review_config: dict = field(default_factory=lambda: {
        "risk_threshold": 0.5,          # 风险阈值
        "batch_size": 20,               # 批量审查大小
        "auto_approve_safe": True,      # 安全文档自动入库
        "require_manual_review": True,  # 高风险文档需人工复核
    })

    # 检索过滤配置
    retrieval_filter_config: dict = field(default_factory=lambda: {
        "use_keyword_filter": True,     # 关键词过滤
        "use_semantic_anomaly": False,  # 语义异常检测（耗资源）
        "use_source_trust": False,      # 来源可信度检查
    })

    # 生成校验配置
    generation_validator_config: dict = field(default_factory=lambda: {
        "check_safety_rules": True,     # 安全规则检查
        "check_consistency": True,      # 事实一致性检查
        "consistency_threshold": 0.5,   # 一致性阈值
    })

    # 安全配置（兼容旧配置）
    enable_content_filter: bool = True
    enable_response_validator: bool = True

    # 关键词过滤配置 - 敏感词列表（中英文）
    sensitive_keywords: list = field(default_factory=lambda: [
        # 危险操作关键词（中文）
        "rm -rf", "del /", "format", "mkfs",
        "DROP TABLE", "DELETE FROM", "TRUNCATE",
        # 恶意引导关键词（中文）
        "忽略之前的", "忽视之前", "绕过", "请执行",
        # 敏感信息关键词（中文）
        "密码是", "密钥是", "令牌是", "api key",
        # 危险操作关键词（英文）
        "eval(", "exec(", "system(", "subprocess",
        "os.system(", "subprocess.call",
        # 恶意引导关键词（英文）
        "ignore previous", "disregard", "bypass",
        "execute this", "run the following",
        # 敏感信息关键词（英文）
        "password is", "api_key is", "secret key is",
        "your token is", "your credential",
    ])

    # 语义异常检测阈值
    semantic_anomaly_threshold: float = 0.3

    # 来源可信度配置
    trusted_sources: list = field(default_factory=lambda: [
        "official_docs",
        "verified_knowledge",
        "curated_content",
        "python_basics",
        "python_security",
        "python_best_practices",
        "python_advanced",
    ])

    def __post_init__(self):
        """初始化后处理，确保目录存在并应用防护模式"""
        os.makedirs(self.chroma_persist_directory, exist_ok=True)

        # 应用防护模式（如果设置了）
        if self.protection_mode:
            self._apply_protection_mode(self.protection_mode)

    def _apply_protection_mode(self, mode: str):
        """
        应用防护模式配置

        Args:
            mode: 防护模式名称 (strict, standard, performance, development, disabled)
        """
        try:
            protection_mode = ProtectionMode(mode)
            mode_config = apply_protection_mode(protection_mode)

            # 只有当开关未被手动覆盖时才应用模式设置
            # 这里我们始终应用模式设置，因为 protection_mode 是显式设置的
            self.enable_ingest_review = mode_config["enable_ingest_review"]
            self.enable_retrieval_filter = mode_config["enable_retrieval_filter"]
            self.enable_generation_validator = mode_config["enable_generation_validator"]
        except ValueError:
            # 无效的模式，忽略
            pass

    def set_protection_mode(self, mode: str):
        """
        动态设置防护模式

        Args:
            mode: 防护模式名称
        """
        self.protection_mode = mode
        self._apply_protection_mode(mode)

    def get_protection_status(self) -> dict:
        """获取当前防护状态"""
        return {
            "mode": self.protection_mode,
            "enable_ingest_review": self.enable_ingest_review,
            "enable_retrieval_filter": self.enable_retrieval_filter,
            "enable_generation_validator": self.enable_generation_validator,
        }

    def calculate_max_batch_size(
        self,
        doc_length: Optional[int] = None,
        verbose: bool = False,
    ) -> int:
        """
        根据上下文窗口计算最大批次大小

        计算公式：
        可用 tokens = (context_window * safety_margin) - system_prompt - user_prompt_overhead
        每文档输入 tokens = doc_length / chars_per_token
        每文档输出 tokens = output_per_doc_tokens
        最大文档数 = 可用 tokens / (每文档输入 + 每文档输出)

        Args:
            doc_length: 文档平均长度（字符），默认使用 review_doc_max_length
            verbose: 是否打印详细信息

        Returns:
            计算得出的最大批次大小
        """
        if doc_length is None:
            doc_length = self.review_doc_max_length

        # 可用 token 数
        total_available = self.llm_context_window * self.review_safety_margin
        input_available = total_available - self.review_system_prompt_tokens - self.review_user_prompt_overhead

        # 每个文档消耗的 token
        doc_input_tokens = doc_length / self.review_chars_per_token
        doc_output_tokens = self.review_output_per_doc_tokens
        doc_total_tokens = doc_input_tokens + doc_output_tokens

        # 计算最大文档数
        max_docs = int(input_available / doc_total_tokens)

        # 应用边界限制
        max_docs = max(self.review_min_batch_size, min(max_docs, self.review_max_batch_size))

        if verbose:
            print(f"=== 批次大小计算 ===")
            print(f"上下文窗口: {self.llm_context_window:,} tokens")
            print(f"安全边际: {self.review_safety_margin * 100}%")
            print(f"系统提示词: {self.review_system_prompt_tokens} tokens")
            print(f"用户提示词开销: {self.review_user_prompt_overhead} tokens")
            print(f"可用输入 tokens: {input_available:,.0f}")
            print(f"文档长度: {doc_length} 字符")
            print(f"每文档输入: {doc_input_tokens:.1f} tokens")
            print(f"每文档输出: {doc_output_tokens} tokens")
            print(f"每文档总计: {doc_total_tokens:.1f} tokens")
            print(f"计算批次大小: {max_docs}")
            print(f"===================")

        return max_docs

    def get_review_batch_info(self, doc_length: Optional[int] = None) -> dict:
        """
        获取审查批次详细信息

        Args:
            doc_length: 文档平均长度

        Returns:
            包含批次计算详情的字典
        """
        if doc_length is None:
            doc_length = self.review_doc_max_length

        max_batch = self.calculate_max_batch_size(doc_length)

        total_available = self.llm_context_window * self.review_safety_margin
        input_available = total_available - self.review_system_prompt_tokens - self.review_user_prompt_overhead
        doc_input_tokens = doc_length / self.review_chars_per_token
        doc_total_tokens = doc_input_tokens + self.review_output_per_doc_tokens

        return {
            "context_window": self.llm_context_window,
            "safety_margin": self.review_safety_margin,
            "total_available_tokens": total_available,
            "input_available_tokens": input_available,
            "doc_length": doc_length,
            "doc_input_tokens": doc_input_tokens,
            "doc_output_tokens": self.review_output_per_doc_tokens,
            "doc_total_tokens": doc_total_tokens,
            "calculated_batch_size": max_batch,
            "min_batch_size": self.review_min_batch_size,
            "max_batch_size": self.review_max_batch_size,
        }

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量创建配置实例"""
        return cls()

    def get_llm_config(self) -> dict:
        """获取 LLM 配置字典"""
        return {
            "api_key": self.deepseek_api_key,
            "base_url": self.deepseek_base_url,
            "model": self.deepseek_model,
        }

    def get_embedding_config(self) -> dict:
        """获取嵌入模型配置字典"""
        return {
            "provider": self.embedding_provider,
            "dashscope_api_key": self.dashscope_api_key,
            "dashscope_model": self.dashscope_embedding_model,
            "ollama_base_url": self.ollama_base_url,
            "ollama_model": self.ollama_embedding_model,
        }

    def get_vectorstore_config(self) -> dict:
        """获取向量数据库配置字典"""
        return {
            "persist_directory": self.chroma_persist_directory,
            "collection_name": self.chroma_collection_name,
        }

    def get_review_config(self) -> dict:
        """获取文档审查配置字典"""
        return {
            "api_key": self.review_api_key,
            "base_url": self.deepseek_base_url,
            "model": self.review_model,
            "risk_threshold": self.review_risk_threshold,
            # 上下文窗口
            "context_window": self.llm_context_window,
            # 文档处理
            "doc_max_length": self.review_doc_max_length,
            "doc_chunk_size": self.review_doc_chunk_size,
            "doc_chunk_overlap": self.review_doc_chunk_overlap,
            # Token 估算
            "chars_per_token": self.review_chars_per_token,
            "system_prompt_tokens": self.review_system_prompt_tokens,
            "user_prompt_overhead": self.review_user_prompt_overhead,
            "output_per_doc_tokens": self.review_output_per_doc_tokens,
            "safety_margin": self.review_safety_margin,
            # 批量处理
            "min_batch_size": self.review_min_batch_size,
            "max_batch_size": self.review_max_batch_size,
            "max_workers": self.review_max_workers,
            "dynamic_batch": self.review_dynamic_batch,
            # 计算得出的批次大小
            "calculated_batch_size": self.calculate_max_batch_size(),
        }


# 全局配置实例
_config: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置实例（单例模式）"""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def reset_config():
    """重置配置（用于测试）"""
    global _config
    _config = None