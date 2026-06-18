# ARIA — AI-Powered Engineering Assistant

**ARIA** is a terminal-based AI engineering assistant that helps you research, architect, code, review, debug, and orchestrate complex software projects — all from your CLI.

```
  ╔══════════════════════════════════════════╗
  ║   ARIA v1.0.0 — Engineering Assistant    ║
  ╚══════════════════════════════════════════╝
```

## Features

### 🧪 R&D Mode
- **Deep Research** — Multi-source synthesis on any topic
- **Comparison** — Side-by-side technology comparison with feature matrices
- **Feasibility Analysis** — Risk assessment, timeline, and viability checks
- **Competitive Analysis** — SWOT analysis and market landscape

### 🛠️ Engineer Mode
- **Architect** — Design system architecture, component diagrams, and data flow
- **Code** — Generate multi-file projects from natural language descriptions
- **Review** — Static analysis for bugs, style issues, and security concerns
- **Debug** — Identify and fix issues in existing code with apply confirmation
- **Scan** — Analyze project structure, dependencies, and imports
- **Edit** — Make precise edits to existing code with preview before applying

### 🔀 Orchestrate Mode
- Decompose complex tasks into sub-agents, execute them, and synthesize results

### 🧠 Memory & Knowledge
- **Persistent memory** with SQLite-backed conversation history
- **Semantic search** using ChromaDB + Ollama embeddings (fully local)
- **Keyword search** using FTS5 full-text search
- **Fact storage** — Save preferences and context about your projects
- **Knowledge recall** — `knowledge [query]` for semantic search across all past content

### 💬 Chat
- Streaming responses with Rich markdown rendering
- Switch between **cloud providers** (Groq, NVIDIA, OpenRouter) and **local Ollama**
- Smart model routing: Groq for chat/research, NVIDIA for deep reasoning, OpenRouter for any model

## Quick Start

### Cloud Mode (API Keys)

```bash
# Prerequisites: Python 3.10+ and an API key from a provider
#   Groq: https://console.groq.com
#   NVIDIA: https://build.nvidia.com
#   OpenRouter: https://openrouter.ai

git clone https://github.com/24b81a0416-oss/ARIA_V3.git
cd ARIA_V3
pip install -r requirements.txt
pip install -e .
```

Create a `.env` file:

```env
GROQ_API_KEY=gsk_your_key_here
NVIDIA_API_KEY=nvapi-your-key-here
OPENROUTER_API_KEY=sk-or-your-key-here
```

```bash
aria
```

### Local Mode (Fully Offline with Ollama)

```bash
# 1. Install Ollama from https://ollama.com/download/windows
# 2. Pull a model (qwen2.5-coder is recommended for coding):
ollama pull qwen2.5-coder:7b

# 3. Pull an embedding model (for the knowledge base):
ollama pull nomic-embed-text

# 4. Run ARIA:
aria
```

At the ARIA prompt, type `mode local` to switch to fully offline mode using Ollama.

### Switching Modes

```
mode local     → 🖥️ Switch to local Ollama (fully offline)
mode cloud     → ☁️ Switch back to cloud providers (Groq, NVIDIA, OpenRouter)
mode auto      → Back to default routing (auto-select by task)
```

ARIA auto-detects Ollama at startup. If Ollama is running, you'll see it in the status and can use `mode local` anytime.

## Commands

| Command | Description |
|---------|-------------|
| `mode local` | 🖥️ Switch to local Ollama (offline) |
| `mode cloud` | ☁️ Switch to cloud providers |
| `mode rd` / `1` | Enter R&D mode |
| `mode engineer` / `2` | Enter Engineer mode |
| `mode orchestrate` / `3` | Enter Orchestrate mode |
| `mode auto` | Back to normal auto-routing |
| `rd [topic]` | Deep research on a topic |
| `rd compare A vs B` | Compare technologies |
| `rd feasibility [idea]` | Feasibility analysis |
| `rd competitive [market]` | Competitive landscape |
| `research [topic]` | Quick web research |
| `engineer [desc]` | Full project generation pipeline |
| `architect [desc]` | System architecture design |
| `code [desc]` | Multi-file code generation |
| `review [path]` | Code review |
| `debug [path]` | Debug and fix code |
| `scan [path]` | Analyze project structure |
| `edit [desc]` | Precise code edits |
| `orchestrate [task]` | Multi-agent task orchestration |
| `ask [question]` | Deep reasoning question |
| `explain [topic]` | Detailed explanation |
| `bash [command]` | Run terminal command |
| `knowledge [query]` | Semantic search across memory |
| `memory` | View/manage memory |
| `model list` | List available models |
| `help` | Show help |
| `status` | Show provider status |
| `exit` / `quit` | Shut down |

## Configuration

All configuration is via environment variables (in `.env` or system):

### Cloud Providers

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Your Groq API key |
| `NVIDIA_API_KEY` | Your NVIDIA NIM API key |
| `OPENROUTER_API_KEY` | Your OpenRouter API key |
| `PRIMARY_PROVIDER` | Default: `auto` (options: groq, nvidia, openrouter, auto) |

### Local (Ollama)

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Default LLM model for chat/code | `qwen2.5-coder:7b` |
| `OLLAMA_EMBED_MODEL` | Model for vector embeddings | `nomic-embed-text` |

## Architecture

```
ARIA_V3/
├── aria.py                 # Main CLI entry point
├── pyproject.toml          # Python packaging config
├── requirements.txt        # Dependencies
├── test_stress.py          # 95-test stress suite
├── agents/                 # AI agent modules
│   ├── architect_agent.py  # System design agent
│   ├── coder_agent.py      # Code generation agent
│   ├── debugger_agent.py   # Bug fixing agent
│   ├── reviewer_agent.py   # Code review agent
│   ├── research_agent.py   # Web research agent
│   ├── engineering_agent.py# Full pipeline agent
│   ├── rd_agent.py         # R&D research agent
│   ├── orchestrator_agent.py# Multi-agent orchestration
│   ├── bash_agent.py       # Command execution agent
│   └── editor_agent.py     # Code editing agent
├── utils/                  # Core utilities
│   ├── config.py           # Configuration & providers
│   ├── llm.py              # LLM client abstraction
│   ├── model_router.py     # Smart model routing
│   ├── memory.py           # SQLite persistent memory
│   ├── vector_store.py     # ChromaDB with Ollama embeddings
│   ├── researcher.py       # Web search & extraction
│   └── skill_manager.py    # Skill system
└── templates/              # Document templates
```

### Provider Routing

| Mode | Task | Provider |
|------|------|----------|
| Cloud (default) | Chat, Research, Docs | Groq |
| Cloud (default) | Complex Coding, Deep Reasoning | NVIDIA |
| Cloud (default) | Any model (Claude, GPT, Gemini) | OpenRouter |
| **Local** | **All tasks** | **Ollama (offline)** |

## Testing

```bash
# Run the full 95-test stress suite
python test_stress.py
```

## License

MIT
