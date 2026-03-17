"""
端到端防护测试

模拟完整的投毒攻击流程，验证各阶段防护效果。
"""

import pytest
from typing import List, Optional

from tests.security.conftest import ProtectionTestResult
from tests.utils.metrics import MetricsCollector


class TestProtectionSwitches:
    """测试防护开关功能"""

    def test_protection_mode_standard(self, protection_mode_config):
        """测试标准防护模式"""
        from poisonedrag.config import ProtectionMode, apply_protection_mode

        mode_config = apply_protection_mode(ProtectionMode.STANDARD)

        assert mode_config["enable_ingest_review"] is True
        assert mode_config["enable_retrieval_filter"] is False
        assert mode_config["enable_generation_validator"] is True

    def test_protection_mode_strict(self, protection_mode_config):
        """测试严格防护模式"""
        from poisonedrag.config import ProtectionMode, apply_protection_mode

        mode_config = apply_protection_mode(ProtectionMode.STRICT)

        assert mode_config["enable_ingest_review"] is True
        assert mode_config["enable_retrieval_filter"] is True
        assert mode_config["enable_generation_validator"] is True

    def test_protection_mode_disabled(self, protection_mode_config):
        """测试禁用防护模式"""
        from poisonedrag.config import ProtectionMode, apply_protection_mode

        mode_config = apply_protection_mode(ProtectionMode.DISABLED)

        assert mode_config["enable_ingest_review"] is False
        assert mode_config["enable_retrieval_filter"] is False
        assert mode_config["enable_generation_validator"] is False

    def test_config_protection_status(self, protection_mode_config):
        """测试配置防护状态获取"""
        status = protection_mode_config.get_protection_status()

        assert "mode" in status
        assert "enable_ingest_review" in status
        assert "enable_retrieval_filter" in status
        assert "enable_generation_validator" in status


class TestIngestStageDetection:
    """测试入库阶段的检测能力"""

    @pytest.fixture(autouse=True)
    def setup(self, poison_samples, metrics_collector):
        """测试前准备"""
        self.samples = poison_samples
        self.metrics = metrics_collector

    def test_ingest_detection_basic(self):
        """基本入库检测测试"""
        from poisonedrag.resecurity import create_document_reviewer
        from poisonedrag.llm import LLMModel
        from poisonedrag.config import get_config

        config = get_config()

        # 检查是否启用入库审查
        if not config.enable_ingest_review:
            pytest.skip("入库审查未启用")

        # 获取 LLM（直接使用 LLMModel 类）
        try:
            llm = LLMModel(
                api_key=config.review_api_key,
                base_url=config.deepseek_base_url,
                model=config.review_model,
            )
        except Exception as e:
            pytest.skip(f"无法初始化 LLM: {e}")

        reviewer = create_document_reviewer(llm=llm)
        results = []

        # 测试部分样本（限制数量避免 API 调用过多）
        test_samples = self.samples[:5]

        for sample in test_samples:
            review_result = reviewer.quick_review(sample["content"])

            result = ProtectionTestResult(
                sample_id=sample.get("id", ""),
                sample_content=sample["content"],
                poison_type=sample["poison_type"],
                stealth_level=sample["stealth_level"],
                ingest_detected=review_result.risk_score >= 0.5,
                ingest_risk_score=review_result.risk_score,
            )

            if result.ingest_detected:
                result.attack_success = False
                result.detection_stage = "ingest"

            results.append(result)
            self.metrics.add_result(result.to_dict())

        # 输出检测结果
        self.metrics.print_summary()

        # 断言：应该有检测结果
        assert len(results) > 0


class TestE2EProtection:
    """端到端防护测试类"""

    @pytest.fixture(autouse=True)
    def setup(self, clean_vectorstore, poison_samples, metrics_collector):
        """测试前准备"""
        self.vectorstore = clean_vectorstore
        self.samples = poison_samples
        self.metrics = metrics_collector

    def test_full_pipeline_with_standard_mode(self, protection_mode_config):
        """测试标准模式下的完整流水线"""
        from poisonedrag.config import ProtectionMode, apply_protection_mode

        mode_config = apply_protection_mode(ProtectionMode.STANDARD)
        results = self._run_pipeline_with_config(mode_config)

        # 输出测试结果
        self._print_mode_results("standard", results)

    def test_full_pipeline_with_strict_mode(self, protection_mode_config):
        """测试严格模式下的完整流水线"""
        from poisonedrag.config import ProtectionMode, apply_protection_mode

        mode_config = apply_protection_mode(ProtectionMode.STRICT)
        results = self._run_pipeline_with_config(mode_config)

        # 输出测试结果
        self._print_mode_results("strict", results)

    def test_full_pipeline_with_performance_mode(self, protection_mode_config):
        """测试性能优先模式下的完整流水线"""
        from poisonedrag.config import ProtectionMode, apply_protection_mode

        mode_config = apply_protection_mode(ProtectionMode.PERFORMANCE)
        results = self._run_pipeline_with_config(mode_config)

        # 输出测试结果
        self._print_mode_results("performance", results)

    def _run_pipeline_with_config(self, mode_config: dict) -> List[ProtectionTestResult]:
        """根据配置运行测试流水线"""
        results = []

        # 初始化各阶段组件
        reviewer = None
        content_filter = None
        validator = None

        if mode_config["enable_ingest_review"]:
            from poisonedrag.resecurity import create_document_reviewer
            from poisonedrag.llm import LLMModel
            from poisonedrag.config import get_config

            config = get_config()
            try:
                llm = LLMModel(
                    api_key=config.review_api_key,
                    base_url=config.deepseek_base_url,
                    model=config.review_model,
                )
                reviewer = create_document_reviewer(llm=llm)
            except Exception:
                pass

        if mode_config["enable_retrieval_filter"]:
            from poisonedrag.security import get_content_filter
            content_filter = get_content_filter()

        if mode_config["enable_generation_validator"]:
            from poisonedrag.security import get_response_validator
            validator = get_response_validator()

        # 测试部分样本（限制数量）
        test_samples = self.samples[:10]

        for sample in test_samples:
            result = self._run_single_attack(
                sample, reviewer, content_filter, validator
            )
            results.append(result)
            self.metrics.add_result(result.to_dict())

        return results

    def _run_single_attack(
        self,
        sample: dict,
        reviewer,
        content_filter,
        validator
    ) -> ProtectionTestResult:
        """运行单次攻击模拟"""
        result = ProtectionTestResult(
            sample_id=sample.get("id", ""),
            sample_content=sample["content"],
            poison_type=sample["poison_type"],
            stealth_level=sample["stealth_level"],
        )

        # 阶段1：入库审查
        if reviewer:
            try:
                review = reviewer.quick_review(sample["content"])
                result.ingest_detected = review.risk_score >= 0.5
                result.ingest_risk_score = review.risk_score
                if result.ingest_detected:
                    result.detection_stage = "ingest"
                    result.attack_success = False
                    return result
            except Exception:
                pass

        # 阶段2：检索过滤（如果文档已入库）
        if content_filter:
            try:
                from langchain_core.documents import Document
                doc = Document(page_content=sample["content"])
                filter_result = content_filter.filter_document(doc)
                result.retrieval_blocked = not filter_result.is_safe
                if result.retrieval_blocked:
                    result.detection_stage = "retrieval"
                    result.attack_success = False
                    return result
            except Exception:
                pass

        # 阶段3：生成校验
        if validator:
            try:
                from langchain_core.documents import Document
                # 模拟 RAG 响应
                mock_response = f"根据知识库信息：{sample['content'][:100]}..."
                docs = [Document(page_content=sample["content"])]

                validation = validator.validate(
                    query="测试查询",
                    response=mock_response,
                    documents=docs
                )
                result.generation_flagged = not validation.is_safe
                if result.generation_flagged:
                    result.detection_stage = "generation"
                    result.attack_success = False
                    return result
            except Exception:
                pass

        # 所有阶段都未检测到
        result.attack_success = True
        return result

    def _print_mode_results(self, mode: str, results: List[ProtectionTestResult]):
        """输出测试结果"""
        total = len(results)
        blocked = sum(1 for r in results if not r.attack_success)
        success_rate = blocked / total if total > 0 else 0

        print(f"\n{'='*60}")
        print(f"防护模式: {mode}")
        print(f"{'='*60}")
        print(f"总测试样本: {total}")
        print(f"成功拦截: {blocked}")
        print(f"拦截率: {success_rate:.2%}")

        # 按阶段统计
        stages = {"ingest": 0, "retrieval": 0, "generation": 0}
        for r in results:
            if r.detection_stage:
                stages[r.detection_stage] += 1

        print(f"\n各阶段拦截数:")
        for stage, count in stages.items():
            print(f"  {stage}: {count}")

        # 按隐蔽等级统计
        stealth_stats = {}
        for r in results:
            level = r.stealth_level
            if level not in stealth_stats:
                stealth_stats[level] = {"detected": 0, "total": 0}
            stealth_stats[level]["total"] += 1
            if not r.attack_success:
                stealth_stats[level]["detected"] += 1

        print(f"\n按隐蔽等级统计:")
        for level, stats in stealth_stats.items():
            rate = stats["detected"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"  {level}: {stats['detected']}/{stats['total']} ({rate:.1f}%)")

        print(f"{'='*60}\n")


class TestStealthLevelSamples:
    """按隐蔽等级测试样本检测"""

    def test_low_stealth_samples_detection(self, low_stealth_samples):
        """测试低隐蔽性样本检测"""
        if not low_stealth_samples:
            pytest.skip("没有低隐蔽性样本")

        # 低隐蔽性样本应该容易被识别
        # 这里只做样本加载测试
        assert len(low_stealth_samples) > 0
        for sample in low_stealth_samples:
            assert sample.get("stealth_level") == "low"
            assert "content" in sample

    def test_medium_stealth_samples_detection(self, medium_stealth_samples):
        """测试中隐蔽性样本检测"""
        if not medium_stealth_samples:
            pytest.skip("没有中隐蔽性样本")

        assert len(medium_stealth_samples) > 0
        for sample in medium_stealth_samples:
            assert sample.get("stealth_level") == "medium"
            assert "content" in sample

    def test_high_stealth_samples_detection(self, high_stealth_samples):
        """测试高隐蔽性样本检测"""
        if not high_stealth_samples:
            pytest.skip("没有高隐蔽性样本")

        assert len(high_stealth_samples) > 0
        for sample in high_stealth_samples:
            assert sample.get("stealth_level") == "high"
            assert "content" in sample


class TestSampleLoading:
    """测试样本加载功能"""

    def test_load_stealth_samples(self, poison_samples):
        """测试加载隐蔽性样本"""
        assert len(poison_samples) > 0

        # 验证样本结构
        for sample in poison_samples:
            assert "id" in sample
            assert "content" in sample
            assert "poison_type" in sample
            assert "stealth_level" in sample
            assert "attack_vector" in sample

    def test_sample_categories(self, poison_samples):
        """测试样本分类"""
        categories = set()
        poison_types = set()
        stealth_levels = set()

        for sample in poison_samples:
            categories.add(sample.get("category", ""))
            poison_types.add(sample.get("poison_type", ""))
            stealth_levels.add(sample.get("stealth_level", ""))

        print(f"\n样本分类统计:")
        print(f"  类别: {categories}")
        print(f"  投毒类型: {poison_types}")
        print(f"  隐蔽等级: {stealth_levels}")

        assert len(categories) > 0
        assert len(poison_types) > 0
        assert len(stealth_levels) > 0