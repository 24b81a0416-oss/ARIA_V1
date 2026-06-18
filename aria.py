#!/usr/bin/env python3
"""
ARIA — Your Engineering Assistant

A rich CLI with streaming responses, smart model routing,
and beautiful terminal output.

Usage:
    python aria.py
    python aria.py --help
"""

from __future__ import annotations

import os
import sys
import io
import re
import time
from pathlib import Path
from typing import Optional

import signal

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.traceback import install as install_rich_traceback

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from utils.config import load_config, print_provider_status, AriaConfig, get_available_providers, get_all_known_models, DEFAULT_MODELS
from utils.model_router import ModelRouter
from utils.llm import create_client, GroqClient  # create_client used by router; GroqClient for engineer pipeline
from agents.research_agent import run_research, format_report
from agents.engineering_agent import run_engineering_pipeline, format_engineering_result
from agents.rd_agent import run_rd_research, format_rd_report
from agents.editor_agent import plan_edits, apply_edits
from utils.document_processor import read_document, write_document, SUPPORTED_EXTENSIONS
from utils.codebase_scanner import scan_project, format_project_context
from utils.skill_manager import list_skills, load_skill, create_skill
from agents.orchestrator_agent import run_orchestration, format_orchestration_result
from agents.bash_agent import run_command, format_result, is_command_safe
from agents.architect_agent import run_architecture, format_architecture_result
from agents.coder_agent import run_coding, format_coding_result
from agents.reviewer_agent import run_review, format_review_result
from agents.debugger_agent import run_debug, format_debug_result
from utils.memory import (
    start_session, end_session, save_message, get_recent_conversations,
    search_conversations, save_fact, get_facts, get_stats,
    forget_message, clear_all, get_relevant_context,
)
from utils.vector_store import search_memory, is_available as vector_available


PROJECT_ROOT = Path(__file__).parent.resolve()
VERSION = "1.0.0"

console = Console()

HELP_TEXT = """
## ARIA Commands

**Work Modes — activate & type naturally:**
  `mode rd` / 1              — 🧪 R&D Mode: research, compare, analyze
  `mode engineer` / 2        — 🛠️ Engineer Mode: build, code, review, debug
  `mode orchestrate` / 3     — 🔀 Orchestrate Mode: complex multi-agent tasks
  `mode auto`                — Back to normal chat (exits work mode)

  *In R&D mode, typing "Flask vs FastAPI" → auto-routes to `rd compare`.*
  *In Engineer mode, typing "fix the login bug" → auto-routes to `debug`.*

**🧪 R&D:**
  `rd [topic]`                — Deep research: multi-angle, multi-source synthesis
  `rd compare [A] vs [B]`     — Technology comparison with feature matrix
  `rd feasibility [idea]`     — Feasibility analysis with risks & timeline
  `rd competitive [market]`   — Competitive landscape with SWOT
  `research [topic]`          — Quick web research with structured report
  `save`                      — Save last report to `research-[topic].md`
  `save [filename]`           — Save last report to a specific file

**🛠️ Engineer:**
  `engineer [desc]`       — Full pipeline: generate + review in one go
  `architect [desc]`      — Design system architecture & components
  `code [desc]`           — Generate multi-file code projects
  `review [path]`         — Review code for bugs, style, security
  `debug [path]`          — Debug & fix issues in code files
  `scan`                  — Analyze current directory structure
  `scan [path]`           — Analyze a specific project path
  `edit [desc]`           — Make precise edits to existing code

**🔀 Orchestrate:**
  `orchestrate [task]`   — Decompose complex tasks, execute sub-agents, synthesize

**💬 Chat:**
  `ask [question]`        — Deep reasoning question (routes to NVIDIA)
  `explain [topic]`       — Detailed explanation (routes to Groq)

**⚙️ System (works in ALL modes):**
  `bash [command]`        — Run terminal command (with safety confirmation)
  `bash! [command]`       — Run terminal command (no confirmation)
  `help`                  — Show this help
  `status`                — Show provider & model status
  `clear`                 — Clear screen
  `exit` / `quit`         — Shut down ARIA
  `memory`                — View, search, and manage memory (FTS5 keyword)
  `knowledge [query]`     — Semantic search across all past conversations & research
  `skill list`            — List available skills
  `skill list`            — List available skills
  `skill show [name]`     — Show skill details & instructions
  `skill create [name]`   — Create a new skill
  `read [file]`           — Read PDF, DOCX, or XLSX files
  `write [file]`          — Write to DOCX or XLSX

**📡 Models & Provider:**
  `model list`            — List all available models by provider
  `model [model name]`    — Switch to a specific model
  `model auto`            — Auto-route by task type (default)
  `mode groq`             — Force all tasks to use Groq
  `mode nvidia`           — Force all tasks to use NVIDIA
  `mode local`            — 🖥️ Force all tasks to use Ollama (fully offline)
  `mode cloud`            — ☁️ Switch back to cloud providers

**🌐 Local vs Cloud:**
  `mode local`            — Run entirely offline using Ollama
  `mode cloud`            — Use cloud providers (Groq, NVIDIA, OpenRouter)
  `mode auto`             — Back to default routing
  *ARIA auto-detects Ollama at startup. If running, you can use `mode local` anytime.*

**Router — Task-to-Provider Mapping:**
  Groq → chat, research, docs, code review
  NVIDIA → complex coding, deep reasoning, architecture
  OpenRouter → any model (Claude, GPT, Gemini, DeepSeek)
  Ollama → fully local, any model you have pulled (qwen2.5-coder, llama3.2, etc.)
"""


class ARIA:
    """ARIA — your engineering assistant CLI."""

    def __init__(self, config: AriaConfig):
        self.config = config
        self.router = ModelRouter(config)
        self.running = True
        self.force_mode: Optional[str] = None
        self.work_mode: Optional[str] = None  # 'rd', 'engineer', 'orchestrate', or None
        self.start_time = time.time()
        self.last_research: Optional[dict] = None  # Stored for 'save' command
        self.session_id: str = ""  # Memory session ID
        self.memory_initialized: bool = False
        self.last_architecture: Optional[dict] = None  # Cached architect output for code pipeline

    # ── Intent Parsing ─────────────────────────────────────────────────

    def parse_intent(self, user_input: str) -> dict:
        """Parse user input and return intent dict."""
        text = user_input.strip().lower()

        if text in ("exit", "quit", "bye"):
            return {"action": "exit", "args": ""}
        if text in ("help", "?"):
            return {"action": "help", "args": ""}
        if text in ("status", "health"):
            return {"action": "status", "args": ""}
        if text == "clear":
            return {"action": "clear", "args": ""}
        if text == "save":
            return {"action": "save", "args": ""}
        if text.startswith("save "):
            return {"action": "save", "args": user_input.strip()[5:].strip()}
        if text.startswith("rd compare "):
            return {"action": "rd", "args": user_input[11:].strip(), "mode": "compare"}
        if text.startswith("rd feasibility "):
            return {"action": "rd", "args": user_input[15:].strip(), "mode": "feasibility"}
        if text.startswith("rd competitive "):
            return {"action": "rd", "args": user_input[15:].strip(), "mode": "competitive"}
        if text == "rd":
            return {"action": "rd", "args": "", "mode": "deep"}
        if text.startswith("rd "):
            return {"action": "rd", "args": user_input[3:].strip(), "mode": "deep"}
        if text.startswith("memory recall "):
            return {"action": "memory", "args": user_input[13:].strip(), "memory_action": "recall"}
        if text.startswith("memory forget "):
            return {"action": "memory", "args": user_input[13:].strip(), "memory_action": "forget"}
        if text == "memory stats":
            return {"action": "memory", "args": "stats", "memory_action": "stats"}
        if text == "memory clear":
            return {"action": "memory", "args": "clear", "memory_action": "clear"}
        if text.startswith("memory fact "):
            # Save a fact: "memory fact prefers python over javascript"
            return {"action": "memory", "args": user_input[12:].strip(), "memory_action": "fact"}
        if text == "memory":
            return {"action": "memory", "args": "", "memory_action": "show"}
        if text.startswith("knowledge "):
            return {"action": "knowledge", "args": user_input[10:].strip()}
        if text == "knowledge":
            return {"action": "knowledge", "args": ""}
        if text.startswith("orchestrate "):
            return {"action": "orchestrate", "args": user_input[12:].strip()}
        if text == "skill list":
            return {"action": "skill", "args": "list"}
        if text.startswith("skill show "):
            return {"action": "skill", "args": user_input[10:].strip(), "skill_action": "show"}
        if text.startswith("skill create "):
            return {"action": "skill", "args": user_input[13:].strip(), "skill_action": "create"}
        if text.startswith("skill "):
            return {"action": "skill", "args": user_input[6:].strip()}
        if text.startswith("scan "):
            return {"action": "scan", "args": user_input[5:].strip()}
        if text == "scan":
            return {"action": "scan", "args": "."}
        if text.startswith("edit "):
            return {"action": "edit", "args": user_input[5:].strip()}
        if text.startswith("engineer "):
            return {"action": "engineer", "args": user_input[9:].strip()}
        if text.startswith("read "):
            return {"action": "read", "args": user_input[5:].strip()}
        if text.startswith("write "):
            return {"action": "write", "args": user_input[6:].strip()}
        if text.startswith("research "):
            return {"action": "research", "args": user_input[9:].strip()}
        if text.startswith("ask "):
            return {"action": "chat", "args": user_input[4:].strip(), "task": "deep_reasoning"}
        if text.startswith("explain "):
            return {"action": "chat", "args": f"Explain {user_input[8:].strip()} in detail", "task": "research"}
        if text == "mode groq":
            return {"action": "set_mode", "args": "groq"}
        if text == "mode nvidia":
            return {"action": "set_mode", "args": "nvidia"}
        if text == "mode local":
            return {"action": "set_mode", "args": "local"}
        if text == "mode cloud":
            return {"action": "set_mode", "args": "cloud"}
        if text == "mode auto":
            return {"action": "set_mode", "args": "auto"}
        if text in ("mode rd", "mode 1", "mode1"):
            return {"action": "set_mode", "args": "rd"}
        if text in ("mode engineer", "mode 2", "mode2"):
            return {"action": "set_mode", "args": "engineer"}
        if text in ("mode orchestrate", "mode 3", "mode3"):
            return {"action": "set_mode", "args": "orchestrate"}
        if text.startswith("architect "):
            return {"action": "architect", "args": user_input[10:].strip()}
        if text.startswith("code "):
            return {"action": "code", "args": user_input[5:].strip()}
        if text.startswith("review "):
            return {"action": "review", "args": user_input[7:].strip()}
        if text.startswith("debug "):
            return {"action": "debug", "args": user_input[6:].strip()}
        if text == "model list":
            return {"action": "models", "args": "list"}
        if text.startswith("model "):
            return {"action": "models", "args": user_input[6:].strip()}
        if text.startswith("bash! "):
            return {"action": "bash", "args": user_input[6:].strip(), "confirm": False}
        if text.startswith("bash "):
            return {"action": "bash", "args": user_input[5:].strip(), "confirm": True}
        if text:
            # Check if we're in a work mode — intelligently detect intent
            if self.work_mode:
                return self._route_by_work_mode(user_input.strip())
            return {"action": "chat", "args": user_input.strip(), "task": "chat"}

        return {"action": "none", "args": ""}

    # ── Intent Handlers ────────────────────────────────────────────────

    def handle_intent(self, intent: dict) -> Optional[str]:
        """Handle parsed intent. Returns response string or None/empty if already printed."""
        action = intent["action"]

        if action == "exit":
            return self._handle_exit()
        elif action == "help":
            return self._handle_help()
        elif action == "status":
            return self._handle_status()
        elif action == "clear":
            return self._handle_clear()
        elif action == "set_mode":
            return self._handle_set_mode(intent["args"])
        elif action == "rd":
            return self._handle_rd(intent["args"], intent.get("mode", "deep"))
        elif action == "memory":
            return self._handle_memory(intent["args"], intent.get("memory_action", "show"))
        elif action == "knowledge":
            return self._handle_knowledge(intent["args"])
        elif action == "orchestrate":
            return self._handle_orchestrate(intent["args"])
        elif action == "skill":
            return self._handle_skill(intent["args"], intent.get("skill_action", ""))
        elif action == "scan":
            return self._handle_scan(intent["args"])
        elif action == "edit":
            return self._handle_edit(intent["args"])
        elif action == "engineer":
            return self._handle_engineer(intent["args"])
        elif action == "read":
            return self._handle_read(intent["args"])
        elif action == "write":
            return self._handle_write(intent["args"])
        elif action == "save":
            return self._handle_save(intent["args"])
        elif action == "bash":
            return self._handle_bash(intent["args"], intent.get("confirm", True))
        elif action == "architect":
            return self._handle_architect(intent["args"])
        elif action == "code":
            return self._handle_code(intent["args"])
        elif action == "review":
            return self._handle_review(intent["args"])
        elif action == "debug":
            return self._handle_debug(intent["args"])
        elif action == "models":
            return self._handle_models(intent["args"])
        elif action == "research":
            return self._handle_research(intent["args"])
        elif action == "chat":
            return self._handle_chat(intent["args"], intent.get("task", "chat"))
        return None

    # ── Provider Resolution Helper ───────────────────────────────────

    def _resolve_client(self, task_type: str = "chat"):
        """
        Get the best client+provider for a task using the router.
        If force_mode is set (e.g. 'ollama' for local mode), bypass router.
        Returns (client, provider_name) or raises RuntimeError.
        """
        # If force_mode is set, use that provider directly (bypass router)
        if self.force_mode:
            available = get_available_providers(self.config)
            if self.force_mode not in available:
                raise RuntimeError(
                    f"Provider '{self.force_mode}' is not available. "
                    f"Available: {', '.join(available) if available else 'none'}"
                )
            try:
                client = create_client(self.config, self.force_mode)
                return client, self.force_mode
            except ValueError as e:
                raise RuntimeError(str(e))

        # Normal router-based resolution
        available = get_available_providers(self.config)
        if not available:
            raise RuntimeError("No LLM providers available. Add an API key to .env")

        try:
            return self.router.get_client(task_type)
        except RuntimeError:
            raise

    # ── Work Mode Smart Routing ─────────────────────────────────────

    def _route_by_work_mode(self, user_input: str) -> dict:
        """
        Intelligently route input based on the active work mode.
        Detects intent from keywords in the input rather than requiring explicit keywords.
        """
        text = user_input.lower()

        if self.work_mode == "rd":
            # R&D Mode: detect compare / feasibility / competitive / deep
            if re.search(r'\b(vs\.?|versus|compare|comparison|difference|better|alternative)\b', text):
                return {"action": "rd", "args": user_input, "mode": "compare"}
            if re.search(r'\b(feasibl|feasible|viab|can we|is it possib|showstopper|worth|practical|realistic)\b', text):
                return {"action": "rd", "args": user_input, "mode": "feasibility"}
            if re.search(r'\b(competitiv|market|competitor|landscape|swot|industry|rival|player|vendor)\b', text):
                return {"action": "rd", "args": user_input, "mode": "competitive"}
            # Default: deep research
            return {"action": "rd", "args": user_input, "mode": "deep"}

        elif self.work_mode == "engineer":
            # Engineer Mode: detect architect / code / review / debug / scan / edit / engineer
            if re.search(r'\b(architect|design.*(?:system|app|arch(?:itecture)?|struct)|component.*diagram|design.*architecture)\b', text):
                return {"action": "architect", "args": user_input}
            if re.search(r'\b(review|audit|inspect|quality|check.*code|code.*check|bug.*find)\b', text):
                return {"action": "review", "args": user_input}
            if re.search(r'\b(debug|fix|bug|error|crash|broken|issue.*code|failing|exception|stack.*trace)\b', text):
                return {"action": "debug", "args": user_input}
            if re.search(r'\b(scan|analy[sz]e|analyz|analys|explor|structure|dependenc|project.*map)\b', text):
                return {"action": "scan", "args": user_input}
            if re.search(r'\b(edit|change|update|modif|refactor|improve|rewrite|add|remove)\b', text):
                return {"action": "edit", "args": user_input}
            if re.search(r'\b(code|generat|creat|implement|write.*(?:code|app|script)|build|make|develop|produc)\b', text):
                return {"action": "code", "args": user_input}
            # Default: full engineer pipeline
            return {"action": "engineer", "args": user_input}

        elif self.work_mode == "orchestrate":
            # Orchestrate Mode: everything routes to multi-agent orchestration
            return {"action": "orchestrate", "args": user_input}

        # Fallback (no work mode active)
        return {"action": "chat", "args": user_input, "task": "chat"}

    # ── Memory Context Helper ────────────────────────────────────────

    def _build_memory_context(self, task: str = "") -> str:
        """Build user facts string from memory for injection into agent prompts."""
        if not self.memory_initialized:
            return ""
        mem_context = get_relevant_context(task)
        facts = mem_context.get("facts", "")
        if facts and facts != "No saved facts yet.":
            return facts
        return ""

    def _handle_memory(self, args: str, action: str = "show") -> str:
        """Handle memory commands — recall, forget, stats, clear, fact."""
        if action == "recall":
            if not args:
                return "What should I search for? Usage: `memory recall [query]`"

            # Try semantic search first (if vector store available)
            lines = [f"## Memories matching: {args}", ""]
            found = False

            if vector_available():
                semantic_results = search_memory(args, limit=8)
                if semantic_results:
                    found = True
                    lines.append("### Semantic Matches")
                    lines.append("")
                    for r in semantic_results:
                        score = r.get("score", 0)
                        bar = "█" * max(1, int(score * 10))
                        source_icon = {"chat": "💬", "research": "📚", "rd": "🧪",
                                       "project": "📦", "fact": "📝"}.get(r["source"], "📄")
                        lines.append(f"{source_icon} [{r['source']}] {bar} ({score:.0%})")
                        lines.append(f"   {r['content'][:200]}")
                        if r.get("time_str"):
                            lines.append(f"   _— {r['time_str']}_")
                        lines.append("")

            # Also do keyword search (complementary)
            kw_results = search_conversations(args, limit=5)
            if kw_results:
                found = True
                lines.append("### Keyword Matches")
                lines.append("")
                for r in kw_results:
                    role_icon = "👤" if r["role"] == "user" else "🤖"
                    highlighted = r.get("highlighted", "") or r["content"][:200]
                    lines.append(f"{role_icon} [{r['time_str']}] {highlighted}")
                    lines.append("")

            if not found:
                return f"No memories found for: {args}"

            lines.append("---")
            lines.append("Keywords match exact terms; Semantic matches find related meaning.")
            return "\n".join(lines)

        if action == "forget":
            try:
                msg_id = int(args)
                if forget_message(msg_id):
                    return f"Forgot message #{msg_id}."
                return f"Message #{msg_id} not found."
            except ValueError:
                return "Usage: `memory forget [message_id]` — use `memory recall` to find IDs."

        if action == "stats":
            stats = get_stats()
            return f"""## Memory Stats

- **Conversations:** {stats['conversations']} messages
- **Facts:** {stats['facts']}
- **Sessions:** {stats['sessions']}
- **Last active:** {stats['last_active']}
- **Database:** {stats['database_size']}
"""

        if action == "clear":
            clear_all()
            return "All memory cleared. Conversations, facts, and sessions deleted."

        if action == "fact":
            if not args or ":" not in args:
                return "Usage: `memory fact key: value` — e.g., `memory fact prefers: react over vue`"
            key, value = args.split(":", 1)
            save_fact(key.strip(), value.strip())
            return f"Saved fact: **{key.strip()}**: {value.strip()}"

        # Default: show memory overview
        stats = get_stats()
        facts = get_facts()
        recent = get_recent_conversations(5)

        lines = ["## Memory", ""]
        lines.append(f"**{stats['conversations']}** messages across **{stats['sessions']}** sessions")
        lines.append("")

        if facts:
            lines.append("### Facts")
            for f in facts[:5]:
                lines.append(f"- **{f['key']}**: {f['value']}")
            lines.append("")

        if recent:
            lines.append("### Recent")
            for r in recent:
                who = "👤" if r["role"] == "user" else "🤖"
                lines.append(f"{who} [{r['time_str']}] {r['content'][:100]}")

        lines.extend([
            "",
            "---",
            "`memory recall [query]` — Search past conversations",
            "`memory fact key: value` — Save a fact about yourself",
            "`memory stats` — Show memory statistics",
            "`memory forget [id]` — Delete a specific message",
        ])
        return "\n".join(lines)

    def _handle_knowledge(self, query: str) -> str:
        """Handle knowledge command — semantic search across all indexed content."""
        if not query:
            return "What do you want to know? Usage: `knowledge [question]` — e.g., `knowledge what did we learn about FastAPI`"

        if not vector_available():
            return "Vector knowledge base not available. Install chromadb and sentence-transformers to enable."

        console.print(f"  Searching knowledge base for: {query}")

        try:
            results = search_memory(query, limit=10)
            if not results:
                return f"No relevant knowledge found for: {query}\n\nTry `memory recall {query}` for keyword search instead."

            lines = [f"## Knowledge: {query}", ""]
            for r in results:
                score = r.get("score", 0)
                bar = "█" * max(1, int(score * 8))
                source_icon = {"chat": "💬", "research": "📚", "rd": "🧪",
                               "project": "📦", "fact": "📝"}.get(r["source"], "📄")
                lines.append(f"{source_icon} [{r['source']}] {bar} ({score:.0%})")
                lines.append(f"   {r['content'][:300]}")
                if r.get("time_str"):
                    lines.append(f"   _— {r['time_str']}_")
                lines.append("")

            lines.append("---")
            lines.append(f"Found {len(results)} results. _Vector store auto-indexes every conversation and research report._")
            return "\n".join(lines)

        except Exception as e:
            return f"Knowledge search failed: {e}"

    def _handle_orchestrate(self, task: str) -> Optional[str]:
        """Handle orchestrate command — multi-agent task decomposition."""
        if not task:
            return "What should I orchestrate? Describe the complex task."

        try:
            client, provider = self._resolve_client("orchestration")
            # Try to get a deep reasoning client too if available
            deep_client = None
            if self.config.nvidia_available:
                try:
                    deep_client, _ = self._resolve_client("deep_reasoning")
                except Exception:
                    pass
            if not deep_client and self.config.openrouter_available:
                try:
                    # Use OpenRouter with a strong model as fallback deep client
                    from utils.llm import OpenRouterClient
                    deep_client = OpenRouterClient(self.config, "google/gemini-2.0-flash:free")
                except Exception:
                    pass
        except RuntimeError as e:
            return str(e)

        user_ctx = self._build_memory_context(task)

        try:
            result = run_orchestration(task, client, deep_client, user_context=user_ctx)
            if "error" in result:
                return f"Orchestration failed: {result['error']}"

            report = format_orchestration_result(result)
            console.print("[bold green]ARIA >[/bold green]")
            md = Markdown(report)
            console.print(md)

            if self.memory_initialized:
                save_message(self.session_id, "assistant", report[:2000])

            return ""
        except Exception as e:
            return f"Orchestration failed: {e}"

    def _handle_skill(self, args: str, skill_action: str = "") -> str:
        """Handle skill commands — list, show, create."""
        if args == "list" or skill_action == "list":
            skills = list_skills()
            if not skills:
                return "No skills found. Create one with `skill create [name]`."

            lines = ["## Available Skills", ""]
            for s in skills:
                lines.append(f"- **{s['name']}** — {s.get('description', 'No description')} (v{s.get('version', '?')})")
            return "\n".join(lines)

        if skill_action == "show" or args.startswith("show "):
            name = args[5:].strip() if args.startswith("show ") else args
            skill = load_skill(name)
            if not skill:
                return f"Skill not found: {name}. Use `skill list` to see available skills."
            return f"""## {skill['name']}

**Version:** {skill.get('version', '?')}
**Description:** {skill.get('description', 'No description')}

### Instructions

```
{skill.get('instructions', 'No instructions')[:1500]}
```
"""

        if skill_action == "create" or args.startswith("create "):
            name = args[7:].strip() if args.startswith("create ") else args
            if not name:
                return "Skill name required. Usage: `skill create [name]`"

            safe_name = "".join(c if c.isalnum() or c == "-" else "_" for c in name.lower().replace(" ", "-"))
            skill_file = create_skill(safe_name, f"Skill for {safe_name}")
            if skill_file:
                return f"Skill **{safe_name}** created at `{skill_file}`"
            else:
                return f"Skill **{safe_name}** already exists."

        return """Skill commands:
- `skill list` — List all skills
- `skill show [name]` — Show skill details
- `skill create [name]` — Create a new skill"""

    def _handle_scan(self, path: str) -> Optional[str]:
        """Handle scan command — analyze project structure."""
        scan_path = Path(path).resolve() if path else Path.cwd()
        if not scan_path.exists():
            return f"Path not found: {scan_path}"

        console.print(f"  Scanning: {scan_path.name}...")

        try:
            result = scan_project(scan_path)
            if "error" in result:
                return f"Scan failed: {result['error']}"

            context = format_project_context(result)
            console.print("[bold green]ARIA >[/bold green]")
            md = Markdown(context)
            console.print(md)

            if self.memory_initialized:
                save_message(self.session_id, "assistant", f"Project scan of {scan_path.name}: {result.get('summary', '')}")

            return ""
        except Exception as e:
            return f"Scan failed: {e}"

    def _handle_edit(self, description: str) -> Optional[str]:
        """Handle edit command — make precise edits to existing code."""
        if not description:
            return "What should I edit? Describe the change you want."

        try:
            client, provider = self._resolve_client("planning")
        except RuntimeError as e:
            return str(e)

        user_ctx = self._build_memory_context(description)

        try:
            plan = plan_edits(description, Path.cwd(), client, user_context=user_ctx)

            if "error" in plan:
                return f"Edit failed: {plan['error']}"

            # Show the plan
            console.print("[bold green]ARIA >[/bold green]")
            md = Markdown(plan.get("preview", "Planning complete."))
            console.print(md)

            # Ask for confirmation
            confirm = Prompt.ask("[yellow]Apply these changes?[/yellow]", default="no")
            if confirm.strip().lower() not in ("yes", "y"):
                return "Edit cancelled."

            # Apply the edits
            console.print("  Applying changes...")
            result = apply_edits(Path.cwd(), plan.get("files", []))

            console.print("[bold green]ARIA >[/bold green]")
            md = Markdown(result.get("summary", "Changes applied."))
            console.print(md)

            if self.memory_initialized:
                save_message(self.session_id, "assistant", result.get("summary", "Edits applied.")[:2000])

            return ""
        except Exception as e:
            return f"Edit failed: {e}"

    def _handle_rd(self, topic: str, mode: str = "deep") -> Optional[str]:
        """Handle R&D research command — deep research, comparisons, feasibility, competitive."""
        if not topic:
            return "What should I research? Use: `rd [topic]`, `rd compare A vs B`, `rd feasibility [idea]`, or `rd competitive [market]`"

        try:
            client, provider = self._resolve_client("research")
        except RuntimeError as e:
            return str(e)

        user_ctx = self._build_memory_context(topic)

        try:
            mode_label = {"compare": "Comparing", "feasibility": "Analyzing feasibility",
                          "competitive": "Analyzing market", "deep": "Researching"}
            with Progress(
                SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(), console=console, transient=True,
            ) as p:
                p.add_task(f"{mode_label.get(mode, 'Researching')}: {topic[:50]}...", total=None)
                result = run_rd_research(topic, client, mode=mode, max_sources=8, user_context=user_ctx)

            report = format_rd_report(result)

            console.print("[bold green]ARIA >[/bold green]")
            md = Markdown(report)
            console.print(md)

            if self.memory_initialized:
                save_message(self.session_id, "assistant", report[:2000])

            return ""
        except Exception as e:
            return f"R&D research failed: {e}"

    def _handle_architect(self, description: str) -> Optional[str]:
        """Handle architect command — design system architecture."""
        if not description:
            return "What should I design the architecture for? Usage: `architect [description]`"

        try:
            client, provider = self._resolve_client("architecture")
        except RuntimeError as e:
            return str(e)

        user_ctx = self._build_memory_context(description)

        console.print(f"  [architect] Designing: {description}")

        try:
            result = run_architecture(description, client, user_context=user_ctx)
            if "error" not in result:
                self.last_architecture = result  # Cache for code pipeline
            report = format_architecture_result(result)

            console.print("[bold green]ARIA >[/bold green]")
            md = Markdown(report)
            console.print(md)

            if self.memory_initialized:
                save_message(self.session_id, "assistant", report[:2000])

            return ""
        except Exception as e:
            return f"Architecture design failed: {e}"

    def _handle_code(self, description: str) -> Optional[str]:
        """Handle code command — generate code from description.

        Automatically chains with the architect: if a recent architecture exists for
        the same description, it's passed as context to the coder. Otherwise,
        architecture is generated first.
        """
        if not description:
            return "What code should I generate? Usage: `code [description]`"

        # Build memory context once, reuse in both phases
        user_ctx = self._build_memory_context(description)

        # ── Architecture Phase ──
        # Check if we have a cached architecture for this description
        arch_text = None
        if self.last_architecture and self.last_architecture.get("description", "").lower() == description.lower():
            arch_text = self.last_architecture.get("architecture", "")
            console.print(f"  [pipeline] Using cached architecture for: {description}")
        else:
            # Auto-run architecture first
            console.print(f"  [pipeline] Running architecture phase first...")
            try:
                arch_client, _ = self._resolve_client("architecture")
                arch_result = run_architecture(description, arch_client, user_context=user_ctx)
                if "error" not in arch_result:
                    self.last_architecture = arch_result
                    arch_text = arch_result.get("architecture", "")
                    console.print(f"  [pipeline] Architecture complete, proceeding to code generation")
                else:
                    console.print(f"  [yellow]Architecture phase warning: {arch_result['error']}[/yellow]")
            except Exception as e:
                console.print(f"  [yellow]Architecture phase failed (continuing without): {e}[/yellow]")

        # ── Code Generation Phase ──
        client, provider = self._resolve_client("complex_code")

        console.print(f"  [coder] Generating code: {description}")

        try:
            result = run_coding(
                description,
                client,
                architecture=arch_text,
                user_context=user_ctx,
            )
            report = format_coding_result(result)

            console.print("[bold green]ARIA >[/bold green]")
            md = Markdown(report)
            console.print(md)

            if self.memory_initialized:
                save_message(self.session_id, "assistant", report[:2000])

            return ""
        except Exception as e:
            return f"Code generation failed: {e}"

    def _handle_review(self, target_path: str) -> Optional[str]:
        """Handle review command — review code files."""
        if not target_path:
            return "What should I review? Usage: `review [path]`"

        try:
            client, provider = self._resolve_client("code_review")
        except RuntimeError as e:
            return str(e)

        user_ctx = self._build_memory_context(target_path)

        console.print(f"  [reviewer] Reviewing: {target_path}")

        try:
            result = run_review(target_path, client, user_context=user_ctx)
            report = format_review_result(result)

            console.print("[bold green]ARIA >[/bold green]")
            md = Markdown(report)
            console.print(md)

            if self.memory_initialized:
                save_message(self.session_id, "assistant", report[:2000])

            return ""
        except Exception as e:
            return f"Review failed: {e}"

    def _handle_debug(self, target_path: str) -> Optional[str]:
        """Handle debug command — debug and fix code issues."""
        if not target_path:
            return "What should I debug? Usage: `debug [path]`"

        try:
            client, provider = self._resolve_client("bug_fixing")
        except RuntimeError as e:
            return str(e)

        user_ctx = self._build_memory_context(target_path)

        console.print(f"  [debugger] Analyzing: {target_path}")

        try:
            result = run_debug(target_path, client, user_context=user_ctx)
            report = format_debug_result(result)

            console.print("[bold green]ARIA >[/bold green]")
            md = Markdown(report)
            console.print(md)

            # Ask for confirmation before applying fixes
            if result.get("bug_count", 0) > 0 and result.get("fixes", []):
                confirm = Prompt.ask("[yellow]Apply these fixes?[/yellow]", default="no")
                if confirm.strip().lower() in ("yes", "y"):
                    console.print("  Applying fixes...")
                    # Re-run with apply=True
                    result = run_debug(target_path, client, user_context=user_ctx, apply_fixes=True)
                    report = format_debug_result(result)
                    console.print("[bold green]ARIA >[/bold green]")
                    md = Markdown(report)
                    console.print(md)

            if self.memory_initialized:
                save_message(self.session_id, "assistant", report[:2000])

            return ""
        except Exception as e:
            return f"Debug failed: {e}"

    def _handle_bash(self, command: str, confirm: bool = True) -> Optional[str]:
        """Handle bash command — execute terminal commands with safety."""
        if not command:
            return "What command should I run? Usage: `bash [command]`"

        if confirm:
            # Check safety first before asking user
            safe, reason = is_command_safe(command)
            if not safe:
                return f"⛔ {reason}"

            console.print(f"  [yellow]Run:[/yellow] `{command}`")
            confirm_input = Prompt.ask("[yellow]Execute this command?[/yellow]", default="no")
            if confirm_input.strip().lower() not in ("yes", "y"):
                return "Command cancelled."
        else:
            # bash! mode — check safety but skip user confirmation
            safe, reason = is_command_safe(command)
            if not safe:
                return f"⛔ {reason}"

        console.print(f"  Running: {command}")

        try:
            result = run_command(command)
            report = format_result(result)

            console.print("[bold green]ARIA >[/bold green]")
            console.print(report)

            if self.memory_initialized:
                summary = f"Command `{command}` exited with code {result.get('return_code', -1)}"
                save_message(self.session_id, "assistant", summary)

            return ""
        except Exception as e:
            return f"Command execution failed: {e}"

    def _handle_exit(self) -> str:
        self.running = False
        return "Shutting down ARIA. Goodbye."

    def _handle_help(self) -> str:
        return HELP_TEXT

    def _handle_status(self) -> str:
        runtime = time.time() - self.start_time
        hours, remainder = divmod(int(runtime), 3600)
        minutes, seconds = divmod(remainder, 60)

        provider_mode = self.force_mode or "auto"
        active_model = self.router.active_model_name
        lines = [
            f"**ARIA v{VERSION}**",
            "",
            f"Runtime: {hours}h {minutes}m {seconds}s",
            f"Provider: {provider_mode}",
            f"Model: `{active_model}`",
            "",
        ]
        if self.work_mode:
            mode_labels = {"rd": "R&D 🧪", "engineer": "Engineer 🛠️", "orchestrate": "Orchestrate 🔀"}
            label = mode_labels.get(self.work_mode, self.work_mode)
            lines.append(f"Work Mode: **{label}** — non-command inputs routed to {self.work_mode}")
            lines.append("")
        lines.extend([
            "**Providers:**",
            f"🌐 Groq: {'✅ Connected' if self.config.groq_available else '⏸️  No API key'}",
            f"🌐 NVIDIA: {'✅ Connected' if self.config.nvidia_available else '⏸️  No API key'}",
            f"🌐 OpenRouter: {'✅ Connected' if self.config.openrouter_available else '⏸️  No API key'}",
            f"🖥️ Ollama (Local): {'✅ Running' if self.config.ollama_available else '⏸️  Not detected'}",
        ])
        if self.config.ollama_available:
            model_name = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
            lines.append(f"     Model: `{model_name}` — use `mode local` for offline mode")
        return "\n".join(lines)

    def _handle_clear(self) -> str:
        console.clear()
        return ""

    def _handle_models(self, args: str) -> Optional[str]:
        """Handle model commands — list, switch, auto."""
        if args == "list":
            models = self.router.list_models()
            if not models:
                return "No models available. Configure at least one API key in .env"

            lines = ["## Available Models", ""]
            current_provider = ""
            for m in models:
                if m["provider"] != current_provider:
                    current_provider = m["provider"]
                    lines.append(f"### {current_provider.title()}")
                    lines.append("")
                active_mark = " **← active**" if m["active"] else ""
                default_mark = " *(default)*" if m.get("default") else ""
                lines.append(f"- `{m['model']}` — {m['description']}{default_mark}{active_mark}")
            lines.append("")
            lines.append("---")
            lines.append("`model [model name]` — Switch to a specific model")
            lines.append("`model auto` — Auto-route by task type")
            return "\n".join(lines)

        if args == "auto":
            return self.router.clear_override()

        if args:
            return self.router.set_model(args)

        return f"Current model: **{self.router.active_model_name}** ({self.router.active_provider})"

    def _handle_set_mode(self, mode: str) -> str:
        if mode == "auto":
            self.force_mode = None
            self.work_mode = None
            return "Mode set to **auto** — back to normal routing."
        elif mode == "local":
            if not self.config.ollama_available:
                return "Ollama is not running. Start Ollama first with `ollama serve` or `ollama run [model]`."
            self.force_mode = "ollama"
            model_name = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
            return f"🖥️ Mode set to **local** — all tasks will use Ollama ({model_name}). Fully offline.\n\nType `mode cloud` or `mode auto` to switch back."
        elif mode == "cloud":
            self.force_mode = None
            self.work_mode = None
            return "☁️ Mode set to **cloud** — using your configured cloud providers (Groq, NVIDIA, OpenRouter)."
        elif mode in ("groq", "nvidia"):
            self.force_mode = mode
            return f"Provider mode set to **{mode}** — all tasks will use {mode.title()}."
        elif mode == "openrouter":
            self.force_mode = mode
            return "Provider mode set to **openrouter** — all tasks will use OpenRouter."
        elif mode == "rd":
            self.work_mode = "rd"
            self.force_mode = None
            return "Mode set to **R&D** 🧪 — any input will trigger deep multi-angle research.\n\nType `mode auto` to exit."
        elif mode == "engineer":
            self.work_mode = "engineer"
            self.force_mode = None
            return "Mode set to **Engineer** 🛠️ — any input will generate a complete project.\n\nType `mode auto` to exit."
        elif mode == "orchestrate":
            self.work_mode = "orchestrate"
            self.force_mode = None
            return "Mode set to **Orchestrate** 🔀 — complex tasks will be decomposed into sub-agents.\n\nType `mode auto` to exit."
        return f"Unknown mode: {mode}.\n\nProvider modes: `groq`, `nvidia`, `openrouter`, `local`, `cloud`, `auto`\nWork modes: `rd` / `1`, `engineer` / `2`, `orchestrate` / `3`"

    def _handle_research(self, topic: str) -> Optional[str]:
        """Handle research command — web search + structured report."""
        if not topic:
            return "What should I research?"

        try:
            client, provider = self._resolve_client("research")
        except RuntimeError as e:
            return str(e)

        console.print(f"  [{provider}] researching: {topic}")

        user_ctx = self._build_memory_context(topic)

        try:
            with Progress(
                SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(), console=console, transient=True,
            ) as p:
                p.add_task(f"Researching: {topic[:50]}...", total=None)
                result = run_research(topic, client, max_sources=5, user_context=user_ctx)

            self.last_research = result  # Store for 'save' command
            report = format_report(result)

            console.print("[bold green]ARIA >[/bold green]")
            md = Markdown(report)
            console.print(md)

            # Save to memory
            if self.memory_initialized:
                save_message(self.session_id, "assistant", report[:2000])

            return ""

        except Exception as e:
            return f"Research failed: {e}"

    def _handle_chat(self, message: str, task_type: str = "chat") -> Optional[str]:
        """Handle chat with streaming — tokens appear as generated."""
        if not message:
            return "What would you like to know?"

        # ── Resolve provider via router ──
        # Map chat task types to router task types
        router_task = task_type if task_type in ("chat", "research", "deep_reasoning") else "chat"
        client, actual_provider = self._resolve_client(router_task)

        # ── Inject memory context ──
        mem_context = get_relevant_context(message)
        facts = mem_context.get("facts", "")

        base_prompt = (
            "You are ARIA, a brilliant engineering partner. "
            "Talk to the user like a person — warm, direct, and natural. "
            "No bullet points unless they genuinely help clarity. "
            "No robot language. No listing features like a spec sheet. "
            "Instead, explain things the way a smart colleague would: "
            "start with a friendly acknowledgment, give the key insight, "
            "then offer next steps if relevant. Use markdown for code."
            "\n\n"
            "## Your Capabilities\n"
            "You have access to these tools through the ARIA CLI — USE them when the user asks for action:\n\n"
            "- **`bash [command]`** — Run ANY terminal command. Use this to create files (`bash! echo 'content' > file.py`), install packages (`bash! pip install flask`), run code (`bash! python app.py`), list directories, check versions, run tests, do git operations, etc.\n"
            "- **`bash! [command]`** — Same as bash but skips the safety confirmation. Use for quick, safe commands.\n"
            "- **`scan [path]`** — Analyze a project's structure, dependencies, and imports.\n"
            "- **`edit [description]`** — Make precise edits to existing code files.\n"
            "- **`research [topic]`** — Web research with structured report.\n"
            "- **`rd [topic]`** — Deep multi-source research with synthesis.\n"
            "- **`engineer [desc]`** — Generate a complete multi-file project.\n"
            "- **`orchestrate [task]`** — Decompose complex tasks into sub-agents.\n"
            "\n"
            "## How to Handle User Requests\n"
            "When the user asks you to **DO** something practical (create a file, build an app, save something, install something, run code):\n"
            "1. First, chat briefly to confirm what they want (1-2 sentences max)\n"
            "2. Then, output the exact `bash` commands they need to run inside a markdown code block\n"
            "3. Tell them they can copy-paste or let you run it\n"
            "\n"
            "Example — user says \"make a flask app on my desktop\":\n"
            "```\n"
            "I'll create a Flask app on your Desktop. Here are the commands to run:\n"
            "\n"
            "bash! cd %USERPROFILE%/Desktop && mkdir -p my_flask_app\n"
            "bash! cd %USERPROFILE%/Desktop/my_flask_app && echo \"from flask import Flask\" > app.py\n"
            "bash! pip install flask\n"
            "```\n"
            "\n"
            "## Handling Confirmation Responses\n"
            "If the user responds with just \"yes\", \"sure\", \"ok\", \"do it\", \"go ahead\", or similar confirmation after you've suggested commands:\n"
            "- Do NOT treat this as a new conversation topic\n"
            "- Instead, recognize it as confirmation to proceed with the commands you JUST suggested\n"
            "- Respond with: \"Say `bash! [first command]` to run it, or type each command one by one.\"\n"
            "- Repeat the exact commands you suggested so they can copy-paste them\n"
            "\n"
            "Example flow:\n"
            "User: \"make a flask app on my desktop\"\n"
            "You: suggest bash commands\n"
            "User: \"yes\"\n"
            "You: \"Great! Run these commands to create your app:\" + repeat the commands\n"
            "\n"
            "Important: Always use `bash!` (no-confirm) for file operations since they're safe. Use `bash` (with confirm) for package installs or commands that could fail.\n"
            "\n"
            "When the user just wants to CHAT (ask questions, discuss ideas, get explanations), just talk naturally without suggesting commands."
        )
        if facts and facts != "No saved facts yet.":
            base_prompt += f"\n\n## About the user\n{facts}"

        # ── Stream the response ──
        try:
            console.print("[bold green]ARIA >[/bold green]")
            buffer = ""
            is_tty = sys.stdout.isatty()

            if is_tty:
                # Interactive mode: stream tokens live using Rich's Live display
                with Live(console=console, refresh_per_second=12, vertical_overflow="visible") as live:
                    for token in client.generate_stream(
                        message,
                        system_prompt=base_prompt,
                        max_tokens=2048,
                    ):
                        buffer += token
                        live.update(Markdown(buffer))
            else:
                # Piped mode: accumulate silently, print once at end
                for token in client.generate_stream(
                    message,
                    system_prompt=base_prompt,
                    max_tokens=2048,
                ):
                    buffer += token

            if not buffer.strip():
                return "Received empty response from the model."

            # Print final response (piped mode skips Live display)
            if not is_tty:
                md = Markdown(buffer.strip())
                console.print(md)

            # Save response to memory
            if self.memory_initialized:
                save_message(self.session_id, "assistant", buffer.strip())

            return ""

        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                return f"Authentication error with {actual_provider}. Check your API key."
            if "429" in error_msg or "rate limit" in error_msg.lower():
                return f"Rate limited by {actual_provider}. Waiting before retry..."
            return f"Error from {actual_provider}: {error_msg}"

    def _handle_engineer(self, description: str) -> Optional[str]:
        """Handle engineer command — full project pipeline."""
        if not description:
            return "What should I build? Describe the project."

        # Use router to get gen + review clients
        try:
            gen_client, gen_provider = self._resolve_client("complex_code")
        except RuntimeError as e:
            return str(e)

        # Try to get a different provider for review
        review_client = None
        try:
            # If gen was NVIDIA, try Groq for review (cheaper)
            if gen_provider == "nvidia" and self.config.groq_available:
                review_client = GroqClient(self.config)
            elif gen_provider == "groq" and self.config.nvidia_available:
                review_client, _ = self._resolve_client("code_review")
            else:
                review_client = gen_client
        except Exception:
            review_client = gen_client  # Use same client for both

        console.print(f"  [engineering] {description}")

        user_ctx = self._build_memory_context(description)

        try:
            result = run_engineering_pipeline(
                problem=description,
                llm_arch=gen_client,
                llm_review=review_client,
                user_context=user_ctx,
            )
            report = format_engineering_result(result)

            console.print("[bold green]ARIA >[/bold green]")
            md = Markdown(report)
            console.print(md)

            if self.memory_initialized:
                save_message(self.session_id, "assistant", report[:2000])

            return ""

        except Exception as e:
            return f"Engineering pipeline failed: {e}"

    def _handle_read(self, filepath: str) -> str:
        """Read a document (PDF, DOCX, XLSX) and return its content."""
        if not filepath:
            exts = ", ".join(SUPPORTED_EXTENSIONS.keys())
            return f"What file should I read? Supported: {exts}"

        console.print(f"  Reading: {filepath}")

        result = read_document(filepath)
        if "error" in result:
            return f"Error: {result['error']}"

        content = result.get("content", "")
        doc_type = result.get("type", "unknown")

        if not content:
            return "The file appears to be empty or contains no extractable text."

        # For short content, show inline. For long content, summarize.
        if len(content) < 2000:
            preview = content
        else:
            preview = content[:2000] + f"\n\n... *({len(content)} total characters)*"

        info = f"**{filepath}** ({doc_type.upper()})"
        if result.get("pages"):
            info += f" · {result['pages']} pages"
        elif result.get("sheets"):
            info += f" · Sheets: {', '.join(result['sheets'])}"

        return f"{info}\n\n{preview}"

    def _handle_write(self, filepath: str) -> str:
        """Write the last research report to a document (DOCX or XLSX)."""
        if not filepath:
            return "What file should I write to? (.docx or .xlsx)"

        if not self.last_research:
            return "No content to write. Run a `research` first."

        report_text = format_report(self.last_research)

        console.print(f"  Writing: {filepath}")

        # DOCX accepts string, XLSX needs structured data
        if filepath.endswith(".xlsx"):
            # Convert markdown report to structured rows
            lines = report_text.split("\n")
            data = []
            for line in lines:
                if line.strip():
                    data.append({"content": line.strip()})
            result = write_document(filepath, data)
        else:
            result = write_document(filepath, report_text)

        if "error" in result:
            return f"Error: {result['error']}"

        return f"Written to **{result['path']}**"

    def _handle_save(self, filename: str) -> str:
        """Save the last research report to a markdown file."""
        if not self.last_research:
            return "No research report to save. Run a research first."

        topic = self.last_research.get("topic", "report")
        # Sanitize filename
        safe_topic = "".join(c if c.isalnum() or c in " -_" else "_" for c in topic)
        safe_topic = safe_topic.strip().replace(" ", "-")[:50]

        if filename:
            path = Path(filename)
            if not path.suffix:
                path = path.with_suffix(".md")
        else:
            path = Path(f"research-{safe_topic}.md")

        report_text = format_report(self.last_research)

        try:
            path.write_text(report_text, encoding="utf-8")
            return f"Report saved to **{path.name}**"
        except Exception as e:
            return f"Failed to save report: {e}"

    # ── Main Loop ──────────────────────────────────────────────────────

    def run(self) -> None:
        """Main REPL loop."""
        # Startup banner
        console.print()
        console.rule(f"[bold cyan]ARIA[/bold cyan] [dim]v{VERSION} — Engineering Assistant[/dim]")
        console.print()
        print_provider_status(self.config)
        console.print()
        console.print("[dim]Type 'help' for commands, 'exit' to quit.[/dim]")

        if not self.config.any_provider_available:
            if self.config.ollama_available:
                console.print("  [green]Ollama detected! Use `mode local` to run fully offline.[/green]")
            else:
                console.print("\n[yellow]No API keys found. Add GROQ_API_KEY, NVIDIA_API_KEY, and/or OPENROUTER_API_KEY to .env[/yellow]")
                console.print("  [dim]Or install Ollama (ollama.com) for fully local operation.[/dim]")
                return

        # Initialize persistent memory
        self.session_id = start_session()
        self.memory_initialized = True
        console.print(f"  [dim]Memory: session {self.session_id}[/dim]")

        while self.running:
            try:
                user_input = Prompt.ask("\n[bold]You[/bold]")
            except (EOFError, KeyboardInterrupt):
                print()
                console.print("\n[yellow]Shutting down ARIA. Goodbye.[/yellow]")
                break

            if not user_input:
                continue

            # Save user message to memory
            if self.memory_initialized:
                save_message(self.session_id, "user", user_input)

            intent = self.parse_intent(user_input)
            response = self.handle_intent(intent)

            # Save assistant response to memory
            if self.memory_initialized and response and response != "":
                save_message(self.session_id, "assistant", response)

            if response is not None and response != "":
                console.print("[bold green]ARIA >[/bold green]")
                md = Markdown(response.strip())
                console.print(md)


# ── Entry Point ─────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    # Handle --help and --version before anything else
    if "--help" in args or "-h" in args:
        console.print(Markdown(HELP_TEXT))
        return

    if "--version" in args or "-V" in args:
        print(f"ARIA v{VERSION}")
        return

    # ── Install Rich traceback handler for beautiful error display ──
    install_rich_traceback(show_locals=False)

    # ── Register signal handlers for graceful shutdown (scoped to main) ──
    _shutting_down = False

    def _signal_handler(signum, frame):
        nonlocal _shutting_down
        if _shutting_down:
            sys.exit(1)
        _shutting_down = True
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    print(f"  [ARIA] Starting...")
    print(f"  [ARIA] Project: {PROJECT_ROOT}")

    config = load_config()

    aria = ARIA(config)

    try:
        aria.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]ARIA interrupted. Goodbye.[/yellow]")
    except Exception:
        console.print("\n[red]ARIA encountered an error:[/red]")
        # Rich traceback handler already installed, this will be pretty-printed
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
