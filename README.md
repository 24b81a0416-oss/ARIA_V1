# ARIA — AI-Powered Engineering Assistant

**ARIA** is a terminal-based AI engineering assistant that helps you research, architect, code, review, debug, and orchestrate complex software projects — all from your CLI.

```
  ╔══════════════════════════════════════════╗
  ║   ARIA v1.0.0 — Engineering Assistant   ║
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
- **Semantic search** using ChromaDB + sentence-transformers vector store
- **Keyword search** using FTS5 full-text search
- **Fact storage** — Save preferences and context about your projects
- **Knowledge recall** — `knowledge [query]` for semantic search across all past content

### 💬 Chat
- Streaming responses with Rich markdown rendering
- Smart model routing: Groq for chat/research, NVIDIA for deep reasoning, OpenRouter for any model
- Multi-provider support: Groq, NVIDIA, OpenRouter

## Quick Start

### Prerequisites
- Python 3.10+
- API key from at least one provider: [Groq](https://console.groq.com), [NVIDIA](https://build.nvidia.com), or [OpenRouter](https://openrouter.ai)

### Installation

```bash
# Clone the repo
git clone https://github.com/24b81a0416-oss/ARIA_V3.git
cd ARIA_V3

# Install dependencies
pip install -r requirements.txt

# Optionally install as a CLI tool
pip install -e .
```

### Setup

Create a `.env` file in the project root:

```env
# At least one API key is required
GROQ_API_KEY=gsk_your_key_here
NVIDIA_API_KEY=nvapi-your-key-here
OPENROUTER_API_KEY=sk-or-your-key-here
```

### Usage

```bash
# Run interactively
python aria.py

# Or if installed as a CLI tool
aria

# Check version
aria --version
```

## Commands

| Command | Description |
|---------|-------------|
| `mode rd` / `1` | Enter R&D mode |
| `mode engineer` / `2` | Enter Engineer mode |
| `mode orchestrate` / `3` | Enter Orchestrate mode |
| `mode auto` | Exit work mode back to chat |
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
│   ├── vector_store.py     # ChromaDB semantic search
│   ├── researcher.py       # Web search & extraction
│   └── skill_manager.py    # Skill system
└── templates/              # Document templates
```

### Provider Routing

The router automatically selects the best provider for each task:

| Task | Preferred Provider |
|------|-------------------|
| Chat, Research, Docs | Groq (fast/cheap) |
| Complex Coding, Deep Reasoning | NVIDIA (powerful) |
| Any model (Claude, GPT, Gemini) | OpenRouter |

## Testing

```bash
# Run the full 95-test stress suite
python test_stress.py
```

## License

MIT
