# ARIA — AI-Powered Engineering Assistant

**ARIA** is a terminal-based AI engineering assistant with 3 work modes. Enter a mode, then type naturally — ARIA detects your intent.

```
  ╔══════════════════════════════════════════╗
  ║   ARIA v1.0.0 — Engineering Assistant    ║
  ╚══════════════════════════════════════════╝
```

## 3 Work Modes — Just Type Naturally

| Mode | Enter | Description |
|------|-------|-------------|
| 🧪 **Research & Analysis** | `mode 1` or `mode research` | Deep research, comparisons, feasibility studies, competitive analysis |
| 🛠️ **Design & Build** | `mode 2` or `mode plan` | Architecture, code generation, review, debug, scan, edit, full pipeline |
| 🔀 **Orchestrate** | `mode 3` or `mode orchestrate` | Complex multi-agent task decomposition and synthesis |

**No keywords to remember.** Just enter a mode and type naturally:
- In Research mode: *"Flask vs FastAPI"* → comparison · *"is this idea feasible?"* → feasibility analysis
- In Design & Build mode: *"fix the login bug"* → debug · *"build a chat app"* → full project
- In Orchestrate mode: *"set up a FastAPI project with Docker and CI/CD"* → multi-agent pipeline

## Quick Start

### Cloud Mode (API Keys Required)

```bash
# Prerequisites: Python 3.10+ and an API key from a provider
#   Groq: https://console.groq.com
#   NVIDIA: https://build.nvidia.com
#   OpenRouter: https://openrouter.ai

git clone https://github.com/24b81a0416-oss/ARIA_V1.git
cd ARIA_V1
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

## Features

### 🧪 Mode 1 — Research & Analysis
- **Deep Research** — Multi-source synthesis on any topic
- **Comparison** — Side-by-side technology comparison with feature matrices
- **Feasibility Analysis** — Risk assessment, timeline, and viability checks
- **Competitive Analysis** — SWOT analysis and market landscape
- **Web Research** — Quick structured research reports

### 🛠️ Mode 2 — Design & Build
- **Architect** — Design system architecture, component diagrams, and data flow
- **Code** — Generate multi-file projects from natural language descriptions
- **Review** — Static analysis for bugs, style issues, and security concerns
- **Debug** — Identify and fix issues in existing code with apply confirmation
- **Scan** — Analyze project structure, dependencies, and imports
- **Edit** — Make precise edits to existing code with preview before applying
- **Full Pipeline** — Generate + review in one go

### 🔀 Mode 3 — Orchestrate
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
- Smart model routing: Groq for chat/research, NVIDIA for deep reasoning

## Commands

### System Commands (Work in ALL Modes)

| Command | Description |
|---------|-------------|
| `help` | Show help |
| `status` | Show provider & model status |
| `clear` | Clear screen |
| `exit` / `quit` / `bye` | Shut down ARIA |
| `save` | Save last report to markdown |
| `save [filename]` | Save last report to a specific file |

### Memory & Knowledge

| Command | Description |
|---------|-------------|
| `memory` | View memory overview |
| `memory recall [query]` | Search past conversations |
| `memory fact key: value` | Save a fact |
| `memory stats` | View statistics |
| `knowledge [query]` | Semantic search across all indexed content |

### Models & Provider

| Command | Description |
|---------|-------------|
| `model list` | List available models |
| `model [name]` | Switch to a specific model |
| `model auto` | Auto-route by task type |
| `mode local` | 🖥️ Switch to Ollama (offline) |
| `mode cloud` | ☁️ Switch to cloud providers |
| `mode groq` | Force Groq |
| `mode nvidia` | Force NVIDIA |

### Utilities

| Command | Description |
|---------|-------------|
| `bash [command]` | Run terminal command (with safety check) |
| `bash! [command]` | Run terminal command (skip confirmation) |
| `read [file]` | Read PDF, DOCX, or XLSX |
| `write [file]` | Write last report to DOCX or XLSX |
| `skill list` | List available skills |
| `skill show [name]` | View skill details |
| `skill create [name]` | Create a new skill |

## Configuration

All configuration is via environment variables (in `.env` or system):

### Cloud Providers

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Your Groq API key |
| `NVIDIA_API_KEY` | Your NVIDIA NIM API key |
| `OPENROUTER_API_KEY` | Your OpenRouter API key |

### Local (Ollama)

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Default LLM model for chat/code | `qwen2.5-coder:7b` |
| `OLLAMA_EMBED_MODEL` | Model for vector embeddings | `nomic-embed-text` |

## Architecture

```
ARIA_V1/
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
