"""
测试指标收集模块

收集和统计防护测试的各项指标。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import json


@dataclass
class TestMetrics:
    """测试指标"""
    total_samples: int = 0
    detected_samples: int = 0
    attack_success: int = 0

    # 按阶段统计
    ingest_detections: int = 0
    retrieval_blocks: int = 0
    generation_flags: int = 0

    # 按隐蔽等级统计
    stealth_level_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # 按投毒类型统计
    poison_type_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def detection_rate(self) -> float:
        """检测率"""
        return self.detected_samples / self.total_samples if self.total_samples > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "detected_samples": self.detected_samples,
            "attack_success": self.attack_success,
            "detection_rate": f"{self.detection_rate:.2%}",
            "stage_breakdown": {
                "ingest": self.ingest_detections,
                "retrieval": self.retrieval_blocks,
                "generation": self.generation_flags,
            },
            "stealth_level_stats": self.stealth_level_stats,
            "poison_type_stats": self.poison_type_stats,
        }


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.metrics = TestMetrics()

    def add_result(self, result: Dict[str, Any]):
        """添加测试结果"""
        self.results.append(result)
        self._update_metrics(result)

    def _update_metrics(self, result: Dict[str, Any]):
        """更新指标"""
        self.metrics.total_samples += 1

        if result.get("attack_success") is False:
            self.metrics.detected_samples += 1
        else:
            self.metrics.attack_success += 1

        # 按阶段更新
        stage = result.get("detection_stage", "")
        if stage == "ingest":
            self.metrics.ingest_detections += 1
        elif stage == "retrieval":
            self.metrics.retrieval_blocks += 1
        elif stage == "generation":
            self.metrics.generation_flags += 1

        # 按隐蔽等级更新
        stealth = result.get("stealth_level", "unknown")
        if stealth not in self.metrics.stealth_level_stats:
            self.metrics.stealth_level_stats[stealth] = {"detected": 0, "missed": 0}

        if result.get("attack_success") is False:
            self.metrics.stealth_level_stats[stealth]["detected"] += 1
        else:
            self.metrics.stealth_level_stats[stealth]["missed"] += 1

        # 按投毒类型更新
        poison_type = result.get("poison_type", "unknown")
        if poison_type not in self.metrics.poison_type_stats:
            self.metrics.poison_type_stats[poison_type] = {"detected": 0, "missed": 0}

        if result.get("attack_success") is False:
            self.metrics.poison_type_stats[poison_type]["detected"] += 1
        else:
            self.metrics.poison_type_stats[poison_type]["missed"] += 1

    def generate_report(self) -> str:
        """生成测试报告"""
        return json.dumps(self.metrics.to_dict(), indent=2, ensure_ascii=False)

    def save_report(self, filepath: str):
        """保存测试报告"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.generate_report())

    def print_summary(self):
        """打印测试摘要"""
        print(f"\n{'='*60}")
        print("防护测试报告")
        print(f"{'='*60}")
        print(f"总测试样本: {self.metrics.total_samples}")
        print(f"成功检测: {self.metrics.detected_samples}")
        print(f"攻击成功: {self.metrics.attack_success}")
        print(f"检测率: {self.metrics.detection_rate:.2%}")

        print(f"\n各阶段拦截数:")
        print(f"  入库阶段: {self.metrics.ingest_detections}")
        print(f"  检索阶段: {self.metrics.retrieval_blocks}")
        print(f"  生成阶段: {self.metrics.generation_flags}")

        if self.metrics.stealth_level_stats:
            print(f"\n按隐蔽等级统计:")
            for level, stats in self.metrics.stealth_level_stats.items():
                total = stats["detected"] + stats["missed"]
                rate = stats["detected"] / total * 100 if total > 0 else 0
                print(f"  {level}: 检测 {stats['detected']}/{total} ({rate:.1f}%)")

        print(f"{'='*60}\n")