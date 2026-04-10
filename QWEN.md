# PoisonedRAG - Project Context

## Project Overview

PoisonedRAG is a **RAG (Retrieval-Augmented Generation) system with three-stage protection against poisoning attacks**. It is a graduation project that defends against various poisoning attack types including misleading advice, fact injection, malicious code, dependency poisoning, hidden backdoors, and sensitive information leakage.

### Core Architecture

**Three-Stage Protection Pipeline:**
```
Ingest Stage (入库) → Retrieval Stage (检索) → Generation Stage (生成)
  DocumentReviewer     ContentFilter           ResponseValidator
  (LLM Review)         (Keyword/Semantic)      (Consistency Check)
```

- **Stage 1 - Ingest Review**: Core defense using LLM to review documents before they enter the knowledge base
- **Stage 2 - Retrieval Filter**: Optional keyword filtering and semantic anomaly detection on retrieved documents
- **Stage 3 - Generation Validator**: Safety and fact-consistency validation on generated responses

### Protection Modes

| Mode | Ingest Review | Retrieval Filter | Generation Validator | Use Case |
|------|:---:|:---:|:---:|---------|
| `strict` | Yes | Yes | Yes | Maximum security |
| `standard` | Yes | No | Yes | **Default (recommended)** |
| `performance` | Yes | No | No | Performance-first |
| `development` | No | No | Yes | Dev/testing |
| `disabled` | No | No | No | All protections off |

## Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | LangGraph + LangChain |
| LLM | DeepSeek API (`deepseek-chat`) |
| Embedding | Ollama (`qwen3-embedding:0.6b`) |
| Vector DB | ChromaDB |
| Frontend | Streamlit |
| Testing | pytest |
| Package Manager | UV |

## Key Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Run Streamlit application
uv run streamlit run src/poisonedrag/app.py

# Run all tests
uv run pytest tests/security/ -v -s

# Run basic tests (no Ollama needed)
uv run pytest tests/security/test_e2e_protection.py -v -s

# Run performance tests (requires Ollama)
uv run pytest tests/security/test_protection_performance.py -v -s

# Lint code
uv run ruff check .
```

## Project Structure

```
PoisonedRAG/
├── src/poisonedrag/          # Main source code
│   ├── app.py                # Streamlit application entry
│   ├── config.py             # Centralized configuration (singleton pattern)
│   ├── embeddings.py         # Ollama embedding wrapper
│   ├── llm.py                # DeepSeek LLM wrapper
│   ├── vectorstore.py        # ChromaDB vector store
│   ├── rag/                  # RAG core (retriever + generator via LangGraph)
│   ├── security/             # Security filters & validators
│   ├── resecurity/           # Document review module
│   ├── data/                 # Data management
│   └── pages/                # Streamlit app pages
├── data/
│   ├── knowledge/            # Knowledge base documents (JSON format)
│   ├── poison/               # Poison samples for testing (100 samples)
│   └── chroma/               # ChromaDB persistent storage
├── tests/
│   ├── security/             # Security-focused tests
│   └── utils/                # Test utilities (metrics, report generator)
├── docs/                     # Project documentation
├── .env.example              # Environment variable template
└── pyproject.toml            # Project dependencies
```

## Configuration

Configuration is centralized in `src/poisonedrag/config.py` using a singleton pattern via `get_config()`. Key configuration points:

- **LLM**: `DEEPSEEK_API_KEY` and `DEEPSEEK_REVIEW_API_KEY` from `.env`
- **Embeddings**: Ollama at `localhost:11434`, model `qwen3-embedding:0.6b`
- **Vector DB**: ChromaDB persisted at `data/chroma/`
- **Batch Review**: Dynamic batch size calculation based on context window (128K tokens)

### Environment Variables (`.env`)

```bash
DEEPSEEK_API_KEY=your-api-key
DEEPSEEK_REVIEW_API_KEY=your-review-api-key
LANGSMITH_API_KEY=your-langsmith-key  # Optional
```

## Data Formats

**Knowledge Documents** (`data/knowledge/*.json`):
```json
[{"content": "...", "metadata": {"source": "...", "category": "..."}}]
```

**Poison Samples** (`data/poison/*.json`):
```json
[{"content": "...", "poison_type": "...", "description": "..."}]
```

## Development Conventions

- **Package management**: Use `uv` (not `pip` directly) for dependency management
- **Testing**: Tests are in `tests/` directory, organized by category (`security/`, `utils/`)
- **Configuration**: Use the `Config` dataclass and `get_config()` singleton; do not access env vars directly
- **Error handling**: The system uses graceful degradation - if a protection stage fails, it logs and continues
- **Language**: Code comments and documentation are primarily in Chinese; code itself follows Python conventions

## Module Dependencies

```
config.py ← embeddings.py, llm.py, vectorstore.py
embeddings.py ← vectorstore.py
vectorstore.py ← rag/retriever.py
retriever.py ← rag/generator.py
security/filter.py ← rag/retriever.py (injected)
security/validator.py ← rag/generator.py (injected)
```

## Prerequisites

- **Python 3.13+**
- **UV** package manager
- **Ollama** running locally for embeddings (`ollama pull qwen3-embedding:0.6b && ollama serve`)
- **DeepSeek API key** for LLM operations
