# PoisonedRAG 测试文档

本文档介绍 PoisonedRAG 项目的测试方法、测试内容和预期效果。

---

## 测试概述

### 测试目标

PoisonedRAG 的测试主要验证以下能力：

1. **三阶段防护开关** - 独立控制各阶段防护功能的启用/禁用
2. **入库审查能力** - LLM 对投毒语料的检测能力
3. **检索过滤能力** - 关键词和语义过滤的有效性
4. **生成校验能力** - 响应安全性和一致性检查
5. **端到端防护** - 完整攻击链路的防护效果
6. **性能指标统计** - 拦截率、召回率、误报率等关键指标

### 关键性能指标

| 指标 | 说明 | 计算方式 |
|------|------|---------|
| **拦截率** | 投毒语料被成功拦截的比例 | 拦截数 / 投毒样本总数 |
| **召回率** | 正常语料被正确入库的比例 | 入库数 / 正常样本总数 |
| **误报率** | 正常语料被误判为投毒的比例 | 误判数 / 正常样本总数 |
| **攻击成功率** | 投毒语料绕过所有防线的比例 | 攻击成功数 / 投毒样本总数 |

### 防护模式预设

| 模式 | 入库审查 | 检索过滤 | 生成校验 | 适用场景 |
|------|---------|---------|---------|---------|
| `strict` | ✅ | ✅ | ✅ | 最高安全要求 |
| `standard` | ✅ | ❌ | ✅ | 推荐默认模式 |
| `performance` | ✅ | ❌ | ❌ | 性能优先场景 |
| `development` | ❌ | ❌ | ✅ | 开发测试阶段 |
| `disabled` | ❌ | ❌ | ❌ | 关闭所有防护 |

---

## 测试框架结构

```
tests/
├── README.md                   # 本文档
├── security/
│   ├── conftest.py              # pytest 配置和 fixtures
│   ├── test_e2e_protection.py   # 端到端防护测试
│   └── test_protection_performance.py  # 性能测试
└── utils/
    ├── metrics.py               # 测试指标收集
    └── report_generator.py      # HTML 报告生成
```

### 测试文件说明

| 文件 | 功能 | 是否需要 Ollama |
|------|------|----------------|
| `test_e2e_protection.py` | 防护开关、样本加载测试 | ❌ 不需要 |
| `test_protection_performance.py` | 性能测试（拦截率、召回率等） | ✅ 需要 |

### 测试类说明

| 测试类 | 功能 | 测试数量 |
|-------|------|---------|
| `TestProtectionSwitches` | 防护开关功能测试 | 4 |
| `TestIngestStageDetection` | 入库阶段检测测试 | 1 |
| `TestE2EProtection` | 端到端防护测试 | 3 |
| `TestStealthLevelSamples` | 隐蔽等级样本测试 | 3 |
| `TestSampleLoading` | 样本加载测试 | 2 |
| `TestProtectionPerformance` | 防护性能测试 | 拦截率、召回率 |
| `TestFullRAGPipeline` | 完整 RAG 流程测试 | 端到端效果 |

---

## 测试样本库

### 样本文件位置

```
data/poison/
├── misinformation.json         # 虚假信息样本
├── malicious_guidance.json     # 恶意引导样本
├── stealth_samples.json        # 隐蔽性投毒样本（22条）
└── expanded_samples.json       # 扩展样本集（100条）
```

### 扩展样本集分类

`expanded_samples.json` 包含 23 个分类，共 100 条投毒样本，覆盖：

- Python/Django/FastAPI/Flask 安全陷阱
- 数据库安全、依赖投毒、隐蔽后门
- JavaScript、数据科学、容器安全
- 云服务、API、密码学安全
- 日志监控、认证授权、文件处理
- 网络请求、性能优化、Git 安全

### 隐蔽等级定义

| 等级 | 特征 | LLM 识别难度 |
|------|------|-------------|
| `low` | 明显的恶意命令、荒谬的虚假信息 | 容易直接识别 |
| `medium` | 混合真实与虚假、不完整的安全配置 | 需要一定专业知识 |
| `high` | 精心设计的代码注入、依赖投毒、隐蔽后门 | 需要代码审查和安全知识 |

---

## 运行测试

### 前置条件

1. 安装项目依赖
2. 确保 Ollama 服务运行（性能测试需要）

### 运行命令

**运行所有测试：**
```bash
uv run pytest tests/security/ -v -s
```

**运行基础测试（不需要 Ollama）：**
```bash
uv run pytest tests/security/test_e2e_protection.py -v -s
```

**运行性能测试（需要 Ollama）：**
```bash
uv run pytest tests/security/test_protection_performance.py -v -s
```

**运行特定测试类：**
```bash
# 防护开关测试
uv run pytest tests/security/test_e2e_protection.py::TestProtectionSwitches -v -s

# 投毒拦截率测试
uv run pytest tests/security/test_protection_performance.py::TestProtectionPerformance -v -s
```

**生成测试报告：**
```bash
uv run pytest tests/security/ -v -s --html=reports/protection_test.html --self-contained-html
```

---

## 预期测试结果

### 性能测试输出示例

```
============================================================
性能测试结果（批量审查模式）
============================================================

【投毒拦截率】
  总样本: 100
  成功拦截: 100
  攻击成功: 0
  拦截率: 100.00%

【阶段分布】
  入库阶段拦截: 100
  检索阶段拦截: 0
  生成阶段拦截: 0

【性能统计】
  总耗时: 89s
  平均每条: 0.89s
============================================================
```

### 结果解读

| 指标 | 说明 | 理想值 |
|------|------|--------|
| 拦截率 | 投毒语料被拦截的比例 | ≥ 90% |
| 召回率 | 正常语料被正确入库的比例 | ≥ 90% |
| 误报率 | 正常语料被误判的比例 | ≤ 10% |
| 平均耗时 | 每条样本的处理时间 | ≤ 1s（批量模式）|

---

## 常见问题

### Q: 测试跳过 `SKIPPED` 是什么原因？

可能原因：
1. 入库审查未启用
2. 无法初始化 LLM（API Key 无效或网络问题）
3. 样本文件不存在

### Q: 如何调试单个样本的检测效果？

使用 `batch_quick_review` 方法批量审查，或使用 `quick_review` 方法审查单个文本。

### Q: 如何验证特定防护模式的效果？

通过 `config.set_protection_mode()` 设置模式后运行测试。

---

## 测试最佳实践

1. **定期运行完整测试** - 代码变更后运行全量测试
2. **关注高隐蔽性样本** - 更接近真实攻击场景
3. **对比不同模式效果** - 使用 `standard` 和 `strict` 模式对比
4. **查看检测阶段分布** - 入库阶段检测优于检索/生成阶段
5. **持续更新样本库** - 发现新型攻击时添加新样本