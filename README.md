# PoisonedRAG

具有安全防护功能的 RAG（检索增强生成）对话系统，核心是防御投毒攻击。

## 项目简介

PoisonedRAG 是一个毕业设计项目，实现了具有三阶段防护机制的 RAG 对话系统，能够有效检测和防御各类投毒攻击。

### 核心功能

- **智能问答** - 基于知识库的 RAG 检索增强生成
- **三阶段防护** - 入库审查 → 检索过滤 → 生成校验
- **投毒检测** - 识别虚假信息、恶意代码、依赖投毒等多种攻击
- **灵活配置** - 防护模式预设 + 独立开关控制

### 项目亮点

| 特性 | 说明 |
|------|------|
| **100% 拦截率** | 100 条样本测试，全部成功拦截 |
| **批量审查优化** | 利用 LLM 长上下文窗口，速度提升 4.5 倍 |
| **动态批次计算** | 根据上下文窗口自动计算最优批次大小 |
| **规则引擎预筛选** - 快速拦截高风险内容，减少 LLM 调用 |
| **配置化设计** | 所有关键参数可配置，灵活适应不同场景 |

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 框架 | LangGraph + LangChain |
| 对话模型 | DeepSeek API / Volcengine Ark |
| 嵌入模型 | OpenAI 兼容接口（DashScope / Ollama / vLLM 等） |
| 向量数据库 | ChromaDB |
| 前端 | Streamlit |
| 测试框架 | pytest |

---

## 快速开始

### 环境要求

- Python 3.13+
- UV 包管理器
- Ollama（可选，本地运行嵌入模型）

### 安装部署

**1. 安装依赖：**
```bash
pip install uv
uv venv
source .venv/bin/activate
uv pip install -e .
```

**2. 配置 API 密钥：**

复制 `.env.example` 为 `.env` 并填入你的 API 密钥：
```bash
cp .env.example .env
```

编辑 `.env` 文件：
```bash
# DeepSeek API 配置
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_REVIEW_API_KEY=your-deepseek-review-api-key

# Embedding 模型配置（默认 DashScope，可在设置界面一键切换）
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v3
EMBEDDING_API_KEY=your-dashscope-api-key

# LangSmith 配置（可选，用于追踪调试）
LANGSMITH_API_KEY=your-langsmith-api-key
```

> 获取 API 密钥：[DeepSeek](https://platform.deepseek.com/) | [DashScope](https://dashscope.console.aliyun.com/) | [LangSmith](https://smith.langchain.com/)

**2.1（可选）本地 Ollama 嵌入模型：**

如果你想使用本地 Ollama 代替云端 DashScope，可以编辑 `.env` 切换：
```bash
# 注释掉 DashScope 配置，启用 Ollama
# EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# EMBEDDING_MODEL=text-embedding-v3
# EMBEDDING_API_KEY=xxx

EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_MODEL=qwen3-embedding:0.6b
EMBEDDING_API_KEY=ollama
```

```bash
ollama pull qwen3-embedding:0.6b
ollama serve
```

也可以在应用**设置界面**中点击预设按钮一键切换，无需手动编辑 `.env`。

**3. 启动应用：**
```bash
uv run streamlit run src/poisonedrag/app.py
```

应用将在 `http://localhost:8501` 启动。

---

## 核心架构

### 三阶段防护机制

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PoisonedRAG 防护架构                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  入库阶段                检索阶段                生成阶段                   │
│  ┌────────────────┐     ┌────────────────┐     ┌────────────────┐          │
│  │ DocumentReviewer│     │ ContentFilter  │     │ResponseValidator│          │
│  │ (LLM审查)       │     │ (关键词/语义)   │     │ (一致性检查)    │          │
│  │                │     │                │     │                │          │
│  │ 核心防线 ✅    │     │ 可选防线       │     │ 最后防线 ✅    │          │
│  └────────────────┘     └────────────────┘     └────────────────┘          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**阶段 1：入库审查** - 核心防线，使用 LLM 对入库文档进行安全审查，识别潜在风险内容。

**阶段 2：检索过滤** - 可选防线，对检索到的文档进行关键词过滤和语义异常检测。

**阶段 3：生成校验** - 最后防线，对生成的响应进行安全校验和事实一致性检查。

### 防护模式预设

| 模式 | 入库审查 | 检索过滤 | 生成校验 | 适用场景 |
|------|---------|---------|---------|---------|
| `strict` | ✅ | ✅ | ✅ | 最高安全要求 |
| `standard` | ✅ | ❌ | ✅ | **推荐默认模式** |
| `performance` | ✅ | ❌ | ❌ | 性能优先场景 |
| `development` | ❌ | ❌ | ✅ | 开发测试阶段 |
| `disabled` | ❌ | ❌ | ❌ | 关闭所有防护 |

---

## 支持的攻击类型

| 类型 | 说明 | 检测率 |
|------|------|--------|
| `misleading_advice` | 误导性建议 | 90%+ |
| `fact_injection` | 错误事实注入 | 95%+ |
| `malicious_code` | 恶意代码 | 96%+ |
| `dependency_poisoning` | 依赖投毒 | **100%** |
| `hidden_backdoor` | 后门代码 | **100%** |
| `sensitive_info` | 敏感信息泄露 | 92%+ |

> 测试基于 100 条样本，总体拦截率 **100%**（优化后）

---

## 测试验证

详细测试文档请参阅 [tests/README.md](tests/README.md)。

### 运行测试

```bash
# 运行所有防护测试
uv run pytest tests/security/ -v -s

# 运行性能测试
uv run pytest tests/security/test_protection_performance.py -v -s
```

### 测试结果

```
【投毒拦截率】
  总样本: 100
  成功拦截: 100
  拦截率: 100.00%

【性能统计】
  总耗时: 89s
  平均每条: 0.89s
```

---

## 项目结构

```
PoisonedRAG/
├── src/poisonedrag/          # 源代码
│   ├── config.py             # 配置管理
│   ├── embeddings.py         # 嵌入模型（OpenAI 兼容接口）
│   ├── llm.py                # 对话模型
│   ├── vectorstore.py        # 向量数据库
│   ├── rag/                  # RAG 核心模块
│   ├── security/             # 安全过滤模块
│   ├── resecurity/           # 文档审查模块
│   ├── data/                 # 数据管理模块
│   ├── pages/                # Streamlit 页面
│   │   ├── app.py            # 主页（对话）
│   │   ├── knowledge_management.py  # 知识库管理
│   │   ├── review.py         # 人工审核
│   │   ├── settings.py       # 设置
│   │   └── visualization.py  # 防毒效果可视化
│   └── app.py                # Streamlit 入口
├── data/                     # 数据文件
│   ├── knowledge/            # 知识库文档
│   ├── poison/               # 投毒样本（380+ 条，跨领域）
│   └── chroma/               # 向量数据库
├── docs/                     # 项目文档
├── tests/                    # 测试文件
└── pyproject.toml
```

---

## 文档

- [测试文档](tests/README.md) - 测试方法和预期效果
- [防护优化过程](docs/防护优化过程.md) - 开发难点与解决方案
- [方案审查与优化建议](docs/方案审查与优化建议.md) - 三阶段防护评估
- [可视化展示方案](docs/可视化展示.md) - 防毒效果可视化设计

---

## 许可证

MIT License

---

## 致谢

本项目为毕业设计作品，感谢以下开源项目：
- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [Streamlit](https://streamlit.io)
- [Ollama](https://ollama.ai)