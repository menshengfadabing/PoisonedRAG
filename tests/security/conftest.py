"""
pytest 配置和 fixtures

提供测试所需的共享 fixtures 和测试结果数据类。
"""

import pytest
import json
import tempfile
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class ProtectionTestResult:
    """防护测试结果"""
    sample_id: str
    sample_content: str
    poison_type: str
    stealth_level: str

    # 各阶段检测结果
    ingest_detected: bool = False
    ingest_risk_score: float = 0.0
    retrieval_blocked: bool = False
    generation_flagged: bool = False

    # 最终结果
    attack_success: bool = True  # 默认攻击成功（未检测到）
    detection_stage: str = ""      # 在哪个阶段被检测到

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "poison_type": self.poison_type,
            "stealth_level": self.stealth_level,
            "ingest_detected": self.ingest_detected,
            "ingest_risk_score": self.ingest_risk_score,
            "retrieval_blocked": self.retrieval_blocked,
            "generation_flagged": self.generation_flagged,
            "attack_success": self.attack_success,
            "detection_stage": self.detection_stage,
        }


@pytest.fixture
def protection_test_result_class():
    """返回测试结果类"""
    return ProtectionTestResult


@pytest.fixture
def reset_config():
    """重置配置 fixture"""
    from poisonedrag.config import reset_config
    reset_config()
    yield
    reset_config()


@pytest.fixture
def protection_mode_config(reset_config):
    """防护模式配置 fixture"""
    from poisonedrag.config import get_config
    return get_config()


@pytest.fixture
def poison_samples():
    """加载投毒样本"""
    project_root = Path(__file__).parent.parent.parent
    samples_file = project_root / "data" / "poison" / "stealth_samples.json"

    if not samples_file.exists():
        pytest.skip(f"投毒样本文件不存在: {samples_file}")

    with open(samples_file, 'r', encoding='utf-8') as f:
        categories = json.load(f)

    all_samples = []
    for category in categories:
        for sample in category.get("samples", []):
            sample["category"] = category.get("category", "")
            all_samples.append(sample)

    return all_samples


@pytest.fixture
def low_stealth_samples(poison_samples):
    """低隐蔽性样本"""
    return [s for s in poison_samples if s.get("stealth_level") == "low"]


@pytest.fixture
def medium_stealth_samples(poison_samples):
    """中隐蔽性样本"""
    return [s for s in poison_samples if s.get("stealth_level") == "medium"]


@pytest.fixture
def high_stealth_samples(poison_samples):
    """高隐蔽性样本"""
    return [s for s in poison_samples if s.get("stealth_level") == "high"]


@pytest.fixture
def clean_vectorstore():
    """清洁的测试用向量数据库"""
    from poisonedrag.vectorstore import VectorStore
    from poisonedrag.embeddings import get_embedding_model
    import tempfile
    import os

    # 使用临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db_path = os.path.join(tmpdir, "test_chroma")

        # 获取嵌入模型
        try:
            embedding = get_embedding_model()
        except Exception:
            # 如果无法连接 Ollama，使用 None（某些测试不需要向量存储）
            embedding = None

        vectorstore = VectorStore(
            embedding_function=embedding,
            persist_directory=test_db_path,
            collection_name="test_collection"
        )
        yield vectorstore
        # 临时目录会自动清理


@pytest.fixture
def mock_llm():
    """模拟 LLM 响应"""
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage

    mock = MagicMock()
    # 默认返回安全评分
    mock.invoke.return_value = AIMessage(content='{"results": [{"id": 1, "risk_score": 0.1}]}')
    return mock


@pytest.fixture
def metrics_collector():
    """指标收集器"""
    from tests.utils.metrics import MetricsCollector
    return MetricsCollector()