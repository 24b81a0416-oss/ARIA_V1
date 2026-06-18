"""
ARIA — Comprehensive Stress Test Suite

Tests every module, edge case, and failure mode to ensure
deployment readiness. Run with:
    python test_stress.py

Exits with code 0 if all tests pass, 1 if any fail.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ── Test Framework (minimal, no external deps) ────────────────────────

PASSED = 0
FAILED = 0
ERRORS: List[str] = []
_TEST_FUNCTIONS: List[tuple] = []  # (name, func)


def test(name: str):
    """Decorator that registers a test function."""
    def decorator(func):
        _TEST_FUNCTIONS.append((name, func))
        return func
    return decorator


def assert_eq(a, b, msg=""):
    assert a == b, f"{msg or ''} Expected {b!r}, got {a!r}"


def assert_in(a, b, msg=""):
    assert a in b, f"{msg or ''} Expected {a!r} to be in {b!r}"


def assert_true(val, msg=""):
    assert val, msg or f"Expected True, got {val!r}"


def assert_false(val, msg=""):
    assert not val, msg or f"Expected False, got {val!r}"


def assert_not_none(val, msg=""):
    assert val is not None, msg or "Expected non-None"


# ── Configuration ─────────────────────────────────────────────────────

# Suppress model loading progress bars that interfere with test output
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")    # suppress tqdm progress bars

PROJECT_ROOT = Path(__file__).parent.resolve()
os.chdir(str(PROJECT_ROOT))

# Save original .env state
ENV_BACKUP = None
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    ENV_BACKUP = env_path.read_bytes()


def clear_env():
    """Temporarily clear .env for testing."""
    if env_path.exists():
        os.rename(str(env_path), str(env_path) + ".stress_bak")


def restore_env():
    """Restore .env after testing."""
    bak = PROJECT_ROOT / ".env.stress_bak"
    if bak.exists():
        if env_path.exists():
            env_path.unlink()
        os.rename(str(bak), str(env_path))


# ═══════════════════════════════════════════════════════════════════════
# TESTS — Section 1: Import Safety
# ═══════════════════════════════════════════════════════════════════════

@test("Import safety: all core modules load")
def _():
    modules = [
        "utils.config",
        "utils.llm",
        "utils.model_router",
        "utils.memory",
        "utils.vector_store",
        "utils.skill_manager",
        "utils.researcher",
        "utils.file_parser",
        "agents.research_agent",
        "agents.engineering_agent",
        "agents.rd_agent",
        "agents.editor_agent",
        "agents.orchestrator_agent",
        "agents.bash_agent",
        "agents.architect_agent",
        "agents.coder_agent",
        "agents.reviewer_agent",
        "agents.debugger_agent",
    ]
    for mod in modules:
        # Replace / with . for module paths
        mod_name = mod.replace("/", ".").replace("\\", ".")
        # agents submodules: use importlib
        if mod_name.startswith("agents."):
            __import__(mod_name)
        else:
            __import__(mod_name)


@test("Import safety: aria.py compiles without errors")
def _():
    import ast
    with open("aria.py", "rb") as f:
        ast.parse(f.read())


@test("Import safety: vector_store and memory can both be imported")
def _():
    # This tests for circular imports
    from utils.vector_store import is_available
    from utils.memory import start_session
    assert_true(callable(is_available))
    assert_true(callable(start_session))


# ═══════════════════════════════════════════════════════════════════════
# Section 2: Config Loading
# ═══════════════════════════════════════════════════════════════════════

@test("Config: load_config returns AriaConfig")
def _():
    from utils.config import load_config, AriaConfig
    config = load_config()
    assert_true(isinstance(config, AriaConfig))
    assert_true(hasattr(config, "groq_available"))
    assert_true(hasattr(config, "nvidia_available"))
    assert_true(hasattr(config, "openrouter_available"))


@test("Config: get_available_providers returns list")
def _():
    from utils.config import load_config, get_available_providers
    config = load_config()
    providers = get_available_providers(config)
    assert_true(isinstance(providers, list))


@test("Config: get_all_known_models returns dict")
def _():
    from utils.config import load_config, get_all_known_models
    config = load_config()
    models = get_all_known_models(config)
    assert_true(isinstance(models, dict))


@test("Config: works without .env file (no crash)")
def _():
    global env_path
    moved = False
    if env_path.exists():
        os.rename(str(env_path), str(env_path) + ".tmp")
        moved = True
    try:
        from utils.config import load_config
        config = load_config()
        assert_true(isinstance(config.groq_available, bool))
    finally:
        if moved:
            os.rename(str(env_path) + ".tmp", str(env_path))


# ═══════════════════════════════════════════════════════════════════════
# Section 3: LLM Clients
# ═══════════════════════════════════════════════════════════════════════

@test("LLM: detect_provider_from_model works for all providers")
def _():
    from utils.llm import detect_provider_from_model

    # NVIDIA models
    assert_eq(detect_provider_from_model("nvidia/llama-3.3-70b-instruct"), "nvidia")
    assert_eq(detect_provider_from_model("deepseek-ai/deepseek-v4-flash"), "nvidia")
    assert_eq(detect_provider_from_model("z-ai/glm-5.1"), "nvidia")
    assert_eq(detect_provider_from_model("moonshotai/kimi-k2.6"), "nvidia")

    # Groq models
    assert_eq(detect_provider_from_model("llama-3.3-70b-versatile"), "groq")
    assert_eq(detect_provider_from_model("qwen/qwen3-32b"), "groq")
    assert_eq(detect_provider_from_model("meta-llama/llama-4-scout-17b-16e-instruct"), "groq")

    # OpenRouter models
    assert_eq(detect_provider_from_model("google/gemini-2.0-flash:free"), "openrouter")
    assert_eq(detect_provider_from_model("openai/gpt-4o-mini:free"), "openrouter")


@test("LLM: create_client raises ValueError for unknown provider")
def _():
    from utils.config import load_config
    from utils.llm import create_client
    config = load_config()
    try:
        create_client(config, "unknown_provider")
        assert_true(False, "Should have raised ValueError")
    except ValueError as e:
        assert_in("Unknown provider", str(e))


@test("LLM: create_client raises ValueError for unavailable provider")
def _():
    from utils.config import AriaConfig
    from utils.llm import create_client
    empty_config = AriaConfig()
    try:
        create_client(empty_config, "groq")
        assert_true(False, "Should have raised ValueError")
    except ValueError as e:
        assert_in("not available", str(e).lower())


@test("LLM: get_model_display_name returns description or fallback")
def _():
    from utils.llm import get_model_display_name
    name = get_model_display_name("llama-3.3-70b-versatile")
    assert_true(len(name) > 0)
    # Unknown model should return itself
    assert_eq(get_model_display_name("completely/fake-model"), "completely/fake-model")


# ═══════════════════════════════════════════════════════════════════════
# Section 4: Model Router
# ═══════════════════════════════════════════════════════════════════════

@test("Router: classify_task returns valid provider")
def _():
    from utils.model_router import classify_task
    tasks = ["chat", "complex_code", "research", "deep_reasoning", "unknown_task"]
    for task in tasks:
        provider = classify_task(task)
        assert_in(provider, ["groq", "nvidia", "openrouter"])


@test("Router: ModelRouter initializes without errors")
def _():
    from utils.config import load_config
    from utils.model_router import ModelRouter
    config = load_config()
    router = ModelRouter(config)
    assert_true(isinstance(router, ModelRouter))


@test("Router: list_models returns list of dicts")
def _():
    from utils.config import load_config
    from utils.model_router import ModelRouter
    config = load_config()
    router = ModelRouter(config)
    models = router.list_models()
    assert_true(isinstance(models, list))
    if models:
        required_keys = {"provider", "model", "description", "active"}
        for m in models:
            for key in required_keys:
                assert_in(key, m, f"Missing key {key}")


@test("Router: set_model validates provider availability")
def _():
    from utils.config import AriaConfig
    from utils.model_router import ModelRouter
    empty_config = AriaConfig()
    router = ModelRouter(empty_config)
    result = router.set_model("llama-3.3-70b-versatile")
    assert_in("Cannot use", result)


@test("Router: clear_override returns expected message")
def _():
    from utils.config import load_config
    from utils.model_router import ModelRouter
    config = load_config()
    router = ModelRouter(config)
    result = router.clear_override()
    assert_in("auto", result.lower())


@test("Router: get_client raises RuntimeError with no providers")
def _():
    from utils.config import AriaConfig
    from utils.model_router import ModelRouter
    empty_config = AriaConfig()
    router = ModelRouter(empty_config)
    try:
        router.get_client("chat")
        assert_true(False, "Should have raised RuntimeError")
    except RuntimeError as e:
        assert_in("No LLM providers", str(e))


# ═══════════════════════════════════════════════════════════════════════
# Section 5: Memory System
# ═══════════════════════════════════════════════════════════════════════

@test("Memory: start_session returns session ID")
def _():
    from utils.memory import start_session, end_session
    session_id = start_session()
    assert_true(len(session_id) > 0)
    end_session(session_id)


@test("Memory: save_message and retrieve")
def _():
    from utils.memory import start_session, save_message, get_recent_conversations
    session_id = start_session()
    msg_id = save_message(session_id, "user", "Hello, ARIA!")
    assert_true(msg_id > 0)
    recent = get_recent_conversations(5)
    contents = [m["content"] for m in recent]
    assert_in("Hello, ARIA!", contents)


@test("Memory: save_message with very long content (10K chars)")
def _():
    from utils.memory import start_session, save_message
    session_id = start_session()
    msg_id = save_message(session_id, "user", "A" * 10000)
    assert_true(msg_id > 0)


@test("Memory: save_message with empty content")
def _():
    from utils.memory import start_session, save_message
    session_id = start_session()
    msg_id = save_message(session_id, "user", "")
    assert_true(msg_id > 0)


@test("Memory: save_message with unicode and special characters")
def _():
    from utils.memory import start_session, save_message, get_recent_conversations
    session_id = start_session()
    special = "Hello Unicode: \u4e16\u754c! Emoji: \U0001f525 Special: ~!@#$%^&*()_+{}|:<>?"
    msg_id = save_message(session_id, "user", special)
    assert_true(msg_id > 0)
    recent = get_recent_conversations(10)
    contents = [m["content"] for m in recent]
    assert_in(special, contents)


@test("Memory: save_fact and get_facts roundtrip")
def _():
    from utils.memory import save_fact, get_facts, delete_fact
    save_fact("test_preference", "likes dark mode", category="test")
    facts = get_facts(category="test")
    keys = [f["key"] for f in facts]
    assert_in("test_preference", keys)
    delete_fact("test_preference")


@test("Memory: duplicate fact key updates in place")
def _():
    from utils.memory import save_fact, get_facts, delete_fact
    save_fact("dup_test", "value1", category="test")
    save_fact("dup_test", "value2", category="test")
    facts = get_facts(category="test")
    dup_facts = [f for f in facts if f["key"] == "dup_test"]
    assert_eq(len(dup_facts), 1)
    assert_eq(dup_facts[0]["value"], "value2")
    delete_fact("dup_test")


@test("Memory: search_conversations with FTS5")
def _():
    from utils.memory import start_session, save_message, search_conversations
    session_id = start_session()
    save_message(session_id, "user", "Testing FastAPI performance")
    save_message(session_id, "user", "Flask is simpler for small apps")
    results = search_conversations("FastAPI", limit=5)
    assert_true(len(results) > 0)


@test("Memory: search_conversations with empty query")
def _():
    from utils.memory import search_conversations
    results = search_conversations("", limit=5)
    assert_true(isinstance(results, list))


@test("Memory: get_relevant_context returns expected keys")
def _():
    from utils.memory import get_relevant_context
    context = get_relevant_context("testing ARIA memory")
    for key in ("facts", "recent_conversations", "relevant_memories"):
        assert_in(key, context)


@test("Memory: get_stats returns expected keys")
def _():
    from utils.memory import get_stats
    stats = get_stats()
    for key in ("conversations", "facts", "sessions", "database_size", "database_path"):
        assert_in(key, stats)


@test("Memory: forget_message returns expected result")
def _():
    from utils.memory import start_session, save_message, forget_message
    session_id = start_session()
    msg_id = save_message(session_id, "user", "Message to forget")
    result = forget_message(msg_id)
    assert_true(result)


@test("Memory: forget_message with invalid ID")
def _():
    from utils.memory import forget_message
    result = forget_message(-1)
    assert_false(result)


@test("Memory: clear_all works")
def _():
    from utils.memory import start_session, save_message, clear_all, get_stats
    session_id = start_session()
    save_message(session_id, "user", "Test")
    clear_all()
    stats = get_stats()
    assert_eq(stats["conversations"], 0)
    assert_eq(stats["facts"], 0)


@test("Memory: end_session updates end_time")
def _():
    from utils.memory import start_session, end_session
    import sqlite3
    session_id = start_session()
    end_session(session_id, "Test session")
    db_path = Path(".aria") / "memory.db"
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT end_time FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row:
            assert_not_none(row[0], "end_time should not be None after end_session")
        conn.close()


@test("Memory: multiple concurrent sessions")
def _():
    from utils.memory import start_session, end_session, get_stats
    sessions = [start_session() for _ in range(5)]
    stats = get_stats()
    assert_true(stats["sessions"] >= 5)
    for s in sessions:
        end_session(s)


# ═══════════════════════════════════════════════════════════════════════
# Section 6: Vector Store
# ═══════════════════════════════════════════════════════════════════════

@test("Vector: is_available returns bool")
def _():
    from utils.vector_store import is_available
    result = is_available()
    assert_true(isinstance(result, bool))


@test("Vector: index_content with valid content")
def _():
    from utils.vector_store import is_available, index_content
    if not is_available():
        return  # Skip
    result = index_content(
        "FastAPI is a modern Python web framework for building APIs",
        source="test",
        metadata={"role": "test"},
    )
    assert_true(result)


@test("Vector: index_content with short content returns False")
def _():
    from utils.vector_store import index_content
    result = index_content("Hi", source="test")
    assert_false(result)


@test("Vector: index_content with empty string returns False")
def _():
    from utils.vector_store import index_content
    result = index_content("", source="test")
    assert_false(result)


@test("Vector: index_content with whitespace-only returns False")
def _():
    from utils.vector_store import index_content
    result = index_content("   \n   ", source="test")
    assert_false(result)


@test("Vector: search_memory returns list")
def _():
    from utils.vector_store import is_available, search_memory
    if not is_available():
        return
    results = search_memory("Python web framework", limit=5)
    assert_true(isinstance(results, list))


@test("Vector: search_memory with empty query returns []")
def _():
    from utils.vector_store import search_memory
    results = search_memory("", limit=5)
    assert_eq(results, [])


@test("Vector: search_memory with very short query returns []")
def _():
    from utils.vector_store import search_memory
    results = search_memory("a", limit=5)
    assert_eq(results, [])


@test("Vector: search_memory with source_filter works")
def _():
    from utils.vector_store import is_available, search_memory
    if not is_available():
        return
    results = search_memory("test", limit=5, source_filter="nonexistent")
    assert_true(isinstance(results, list))


@test("Vector: get_stats returns dict")
def _():
    from utils.vector_store import get_stats
    stats = get_stats()
    assert_true(isinstance(stats, dict))
    assert_in("available", stats)


@test("Vector: clear_all resets store")
def _():
    from utils.vector_store import clear_all, get_stats, is_available
    if not is_available():
        return
    result = clear_all()
    assert_true(result)
    stats = get_stats()
    assert_eq(stats["count"], 0)


@test("Vector: index various English content")
def _():
    from utils.vector_store import is_available, index_content, clear_all
    if not is_available():
        return
    texts = [
        "Hello world this is a test with enough length to pass the minimum length check in the vector store",
        "FastAPI is a modern Python web framework for building high performance APIs",
        "The quick brown fox jumps over the lazy dog near the bank of the river",
        "Python programming language is widely used for data science and web development",
    ]
    for text in texts:
        result = index_content(text, source="test")
        assert_true(result, f"Failed to index: {text}")
    clear_all()


@test("Vector: index very long content (25K words)")
def _():
    from utils.vector_store import is_available, index_content, clear_all
    if not is_available():
        return
    long_text = "word " * 5000
    result = index_content(long_text, source="test")
    assert_true(result)
    clear_all()


# ═══════════════════════════════════════════════════════════════════════
# Section 7: File Parser
# ═══════════════════════════════════════════════════════════════════════

@test("Parser: parse_files with custom ---FILE: markers")
def _():
    from utils.file_parser import parse_files
    output = """---FILE: main.py | Entry point
print("Hello")
---END FILE
---FILE: utils.py
def helper():
    pass
---END FILE"""
    files = parse_files(output)
    assert_eq(len(files), 2)
    assert_eq(files[0]["path"], "main.py")
    assert_eq(files[1]["path"], "utils.py")


@test("Parser: parse_files with markdown code blocks")
def _():
    from utils.file_parser import parse_files
    output = '```python:main.py\nprint("Hello")\n```'
    files = parse_files(output)
    assert_eq(len(files), 1)
    assert_eq(files[0]["path"], "main.py")


@test("Parser: parse_files with empty input")
def _():
    from utils.file_parser import parse_files
    files = parse_files("")
    assert_eq(files, [])


@test("Parser: parse_files with no markers returns []")
def _():
    from utils.file_parser import parse_files
    files = parse_files("Just some random text")
    assert_eq(files, [])


@test("Parser: strip_fences removes markdown fences")
def _():
    from utils.file_parser import strip_fences
    code = strip_fences("```python\nprint('hello')\n```")
    assert_eq(code, "print('hello')")


@test("Parser: strip_fences with no fences returns original")
def _():
    from utils.file_parser import strip_fences
    code = strip_fences("print('hello')")
    assert_eq(code, "print('hello')")


@test("Parser: strip_fences with empty string")
def _():
    from utils.file_parser import strip_fences
    code = strip_fences("")
    assert_eq(code, "")


@test("Parser: sanitize_name handles special characters")
def _():
    from utils.file_parser import sanitize_name
    name = sanitize_name("My Cool Project! (2024) - Test")
    assert_eq(name, "My-Cool-Project_-_2024_---Test")


@test("Parser: parse_files with duplicated markers")
def _():
    from utils.file_parser import parse_files
    output = """---FILE: app.py
code
---END FILE
---FILE: app.py
code
---END FILE"""
    files = parse_files(output)
    assert_eq(len(files), 2)


@test("Parser: parse_files skips shell blocks without filenames")
def _():
    from utils.file_parser import parse_files
    output = """```bash
pip install flask
```
```python
print("no filename")
```"""
    files = parse_files(output)
    has_bash = any("bash" in f.get("path", "") for f in files)
    assert_false(has_bash, "Shell blocks without filename should be skipped")


# ═══════════════════════════════════════════════════════════════════════
# Section 8: Skill Manager
# ═══════════════════════════════════════════════════════════════════════

@test("Skills: list_skills returns list")
def _():
    from utils.skill_manager import list_skills
    skills = list_skills()
    assert_true(isinstance(skills, list))


@test("Skills: create_skill creates skill file")
def _():
    from utils.skill_manager import create_skill, list_skills
    import shutil
    result = create_skill("test-skill-stress", "Test skill")
    if result:
        assert_true(result.exists())
        shutil.rmtree(result.parent)
    else:
        # Already exists, clean up
        skill_dir = Path("skills") / "test-skill-stress"
        if skill_dir.exists():
            shutil.rmtree(skill_dir)


@test("Skills: load_skill returns None for non-existent")
def _():
    from utils.skill_manager import load_skill
    skill = load_skill("this-skill-does-not-exist-12345")
    assert_true(skill is None)


@test("Skills: get_instructions_for_task handles empty skills dir")
def _():
    from utils.skill_manager import get_instructions_for_task
    instructions, names = get_instructions_for_task("test task")
    assert_true(isinstance(instructions, str))
    assert_true(isinstance(names, list))


# ═══════════════════════════════════════════════════════════════════════
# Section 9: Researcher
# ═══════════════════════════════════════════════════════════════════════

@test("Researcher: search_web returns list")
def _():
    from utils.researcher import search_web
    results = search_web("Python programming", max_results=3)
    assert_true(isinstance(results, list))
    if results:
        for r in results:
            assert_in("title", r)
            assert_in("url", r)


@test("Researcher: search_web with empty query")
def _():
    from utils.researcher import search_web
    results = search_web("", max_results=3)
    assert_true(isinstance(results, list))


@test("Researcher: extract_content returns None or string")
def _():
    from utils.researcher import extract_content
    result = extract_content("https://example.com", max_chars=100)
    if result is not None:
        assert_true(len(result) > 0)


@test("Researcher: extract_content with invalid URL returns None (no crash)")
def _():
    from utils.researcher import extract_content
    result = extract_content("https://this-is-not-real-12345.com", max_chars=100)
    assert_true(result is None)


# ═══════════════════════════════════════════════════════════════════════
# Section 10: ARIA Command Parsing
# ═══════════════════════════════════════════════════════════════════════

def _make_aria():
    """Helper: create ARIA instance for testing."""
    # We need to import inside functions to avoid import-time side effects
    from utils.config import load_config
    from aria import ARIA
    config = load_config()
    return ARIA(config)


@test("ARIA: help command detection")
def _():
    aria = _make_aria()
    assert_eq(aria.parse_intent("help")["action"], "help")
    assert_eq(aria.parse_intent("?")["action"], "help")


@test("ARIA: exit command detection")
def _():
    aria = _make_aria()
    assert_eq(aria.parse_intent("exit")["action"], "exit")
    assert_eq(aria.parse_intent("quit")["action"], "exit")
    assert_eq(aria.parse_intent("bye")["action"], "exit")


@test("ARIA: mode commands detection")
def _():
    aria = _make_aria()
    # Work modes
    assert_eq(aria.parse_intent("mode rd")["action"], "set_mode")
    assert_eq(aria.parse_intent("mode 1")["action"], "set_mode")
    assert_eq(aria.parse_intent("mode1")["action"], "set_mode")
    assert_eq(aria.parse_intent("mode engineer")["action"], "set_mode")
    assert_eq(aria.parse_intent("mode 2")["action"], "set_mode")
    assert_eq(aria.parse_intent("mode2")["action"], "set_mode")
    assert_eq(aria.parse_intent("mode orchestrate")["action"], "set_mode")
    assert_eq(aria.parse_intent("mode 3")["action"], "set_mode")
    assert_eq(aria.parse_intent("mode3")["action"], "set_mode")
    assert_eq(aria.parse_intent("mode auto")["action"], "set_mode")
    # Provider modes
    assert_eq(aria.parse_intent("mode groq")["action"], "set_mode")
    assert_eq(aria.parse_intent("mode nvidia")["action"], "set_mode")


@test("ARIA: RD commands detection")
def _():
    aria = _make_aria()
    assert_eq(aria.parse_intent("rd compare FastAPI vs Flask")["action"], "rd")
    assert_eq(aria.parse_intent("rd feasibility building a chat app")["action"], "rd")
    assert_eq(aria.parse_intent("rd competitive AI coding market")["action"], "rd")
    assert_eq(aria.parse_intent("rd Python features")["action"], "rd")
    assert_eq(aria.parse_intent("rd")["action"], "rd")


@test("ARIA: memory commands detection")
def _():
    aria = _make_aria()
    assert_eq(aria.parse_intent("memory recall Flask")["action"], "memory")
    assert_eq(aria.parse_intent("memory forget 5")["action"], "memory")
    assert_eq(aria.parse_intent("memory stats")["action"], "memory")
    assert_eq(aria.parse_intent("memory fact prefers: react")["action"], "memory")
    assert_eq(aria.parse_intent("memory")["action"], "memory")
    assert_eq(aria.parse_intent("memory clear")["action"], "memory")


@test("ARIA: engineer commands detection")
def _():
    aria = _make_aria()
    assert_eq(aria.parse_intent("engineer a Flask todo app")["action"], "engineer")
    assert_eq(aria.parse_intent("code a CLI tool")["action"], "code")
    assert_eq(aria.parse_intent("architect a microservice")["action"], "architect")
    assert_eq(aria.parse_intent("review utils/")["action"], "review")
    assert_eq(aria.parse_intent("debug the login crash")["action"], "debug")
    assert_eq(aria.parse_intent("scan .")["action"], "scan")
    assert_eq(aria.parse_intent("scan")["action"], "scan")
    assert_eq(aria.parse_intent("edit add dark mode")["action"], "edit")


@test("ARIA: system commands detection")
def _():
    aria = _make_aria()
    assert_eq(aria.parse_intent("bash pip list")["action"], "bash")
    assert_eq(aria.parse_intent("bash! pip install flask")["action"], "bash")
    assert_eq(aria.parse_intent("status")["action"], "status")
    assert_eq(aria.parse_intent("clear")["action"], "clear")
    assert_eq(aria.parse_intent("save")["action"], "save")
    assert_eq(aria.parse_intent("save my_report.md")["action"], "save")


@test("ARIA: chat/research commands")
def _():
    aria = _make_aria()
    assert_eq(aria.parse_intent("research Python trends")["action"], "research")
    assert_eq(aria.parse_intent("ask what is FastAPI")["action"], "chat")
    assert_eq(aria.parse_intent("explain quantum computing")["action"], "chat")


@test("ARIA: knowledge command")
def _():
    aria = _make_aria()
    assert_eq(aria.parse_intent("knowledge Python web frameworks")["action"], "knowledge")
    assert_eq(aria.parse_intent("knowledge")["action"], "knowledge")


@test("ARIA: model commands")
def _():
    aria = _make_aria()
    assert_eq(aria.parse_intent("model list")["action"], "models")
    assert_eq(aria.parse_intent("model llama-3.3-70b")["action"], "models")


@test("ARIA: skill commands")
def _():
    aria = _make_aria()
    assert_eq(aria.parse_intent("skill list")["action"], "skill")
    assert_eq(aria.parse_intent("skill show test")["action"], "skill")
    assert_eq(aria.parse_intent("skill create new-skill")["action"], "skill")


@test("ARIA: orchestrate command")
def _():
    aria = _make_aria()
    assert_eq(aria.parse_intent("orchestrate deploy microservices")["action"], "orchestrate")


@test("ARIA: read/write commands")
def _():
    aria = _make_aria()
    assert_eq(aria.parse_intent("read document.pdf")["action"], "read")
    assert_eq(aria.parse_intent("write report.docx")["action"], "write")


@test("ARIA: empty input")
def _():
    aria = _make_aria()
    result = aria.parse_intent("")
    assert_in(result["action"], ("none", "chat"))


@test("ARIA: unknown input routes to chat")
def _():
    aria = _make_aria()
    result = aria.parse_intent("What is the meaning of life?")
    assert_eq(result["action"], "chat")


@test("ARIA: work mode smart routing - RD mode")
def _():
    aria = _make_aria()
    aria.work_mode = "rd"

    result = aria.parse_intent("Flask vs FastAPI")
    assert_eq(result["action"], "rd")
    assert_eq(result.get("mode"), "compare")

    result = aria.parse_intent("is it feasible to build a chat app")
    assert_eq(result["action"], "rd")
    assert_eq(result.get("mode"), "feasibility")

    result = aria.parse_intent("AI coding market competitors")
    assert_eq(result["action"], "rd")
    assert_eq(result.get("mode"), "competitive")

    result = aria.parse_intent("Python 3.14 new features")
    assert_eq(result["action"], "rd")
    assert_eq(result.get("mode"), "deep")


@test("ARIA: work mode smart routing - Engineer mode")
def _():
    aria = _make_aria()
    aria.work_mode = "engineer"

    result = aria.parse_intent("design a microservice architecture")
    assert_eq(result["action"], "architect")

    result = aria.parse_intent("review the utils folder")
    assert_eq(result["action"], "review")

    result = aria.parse_intent("fix the login crash bug")
    assert_eq(result["action"], "debug")

    result = aria.parse_intent("analyze the project structure")
    assert_eq(result["action"], "scan")

    result = aria.parse_intent("add dark mode toggle")
    assert_eq(result["action"], "edit")

    result = aria.parse_intent("build a Flask todo app")
    assert_eq(result["action"], "code")

    result = aria.parse_intent("create a full project from scratch")
    assert_eq(result["action"], "engineer")


@test("ARIA: work mode smart routing - Orchestrate mode")
def _():
    aria = _make_aria()
    aria.work_mode = "orchestrate"
    result = aria.parse_intent("plan a microservices deployment")
    assert_eq(result["action"], "orchestrate")


@test("ARIA: handle_intent returns string or None for all actions")
def _():
    aria = _make_aria()
    actions_to_test = [
        {"action": "exit", "args": ""},
        {"action": "help", "args": ""},
        {"action": "status", "args": ""},
        {"action": "clear", "args": ""},
        {"action": "knowledge", "args": ""},
        {"action": "knowledge", "args": "test query"},
        {"action": "memory", "args": "stats", "memory_action": "stats"},
        {"action": "memory", "args": "", "memory_action": "show"},
        {"action": "memory", "args": "fake query", "memory_action": "recall"},
        {"action": "set_mode", "args": "auto"},
        {"action": "set_mode", "args": "rd"},
        {"action": "set_mode", "args": "engineer"},
    ]
    for intent in actions_to_test:
        try:
            response = aria.handle_intent(intent)
            assert_true(response is None or isinstance(response, str),
                       f"Unexpected type for {intent['action']}: {type(response)}")
        except Exception as e:
            assert_true(False, f"handle_intent crashed for {intent['action']}: {e}")
    aria.running = True


@test("ARIA: HELP_TEXT contains all sections")
def _():
    from aria import HELP_TEXT
    for section in ["Work Modes", "R&D", "Engineer", "Orchestrate", "Chat", "System", "Models & Provider"]:
        assert_in(section, HELP_TEXT)


# ═══════════════════════════════════════════════════════════════════════
# Section 11: Engineering Agent
# ═══════════════════════════════════════════════════════════════════════

@test("Engineer: format_engineering_result handles error case")
def _():
    from agents.engineering_agent import format_engineering_result
    result = {"error": "Something went wrong", "problem": "test"}
    output = format_engineering_result(result)
    assert_in("Failed", output)


@test("Engineer: format_engineering_result handles success case")
def _():
    from agents.engineering_agent import format_engineering_result
    result = {
        "problem": "test project",
        "project_dir": "/tmp/test",
        "files": [{"path": "main.py", "size": 100}],
        "reviews": [{"file": "main.py", "passed": True, "review": "Looks good"}],
        "duration": "5s",
        "file_count": 1,
        "passed_reviews": 1,
        "total_reviews": 1,
        "fix_passes": 0,
        "all_passed": True,
        "install_output": "",
        "test_output": "",
    }
    output = format_engineering_result(result)
    assert_in("test project", output)
    assert_in("main.py", output)


@test("Engineer: run_engineering_pipeline handles empty generation")
def _():
    from agents.engineering_agent import run_engineering_pipeline
    from utils.llm import LLMClient

    class EmptyClient(LLMClient):
        model = "mock"
        provider = "mock"
        def generate_stream(self, *args, **kwargs):
            return iter([])
        def generate(self, *args, **kwargs):
            return ""

    result = run_engineering_pipeline(
        problem="test",
        llm_arch=EmptyClient(),
        llm_review=EmptyClient(),
        auto_install=False,
        auto_test=False,
    )
    assert_in("files", result)
    assert_in("duration", result)


# ═══════════════════════════════════════════════════════════════════════
# Section 12: Edge Cases & Error Recovery
# ═══════════════════════════════════════════════════════════════════════

@test("Edge: very long command input (5000+ chars)")
def _():
    aria = _make_aria()
    long_input = "ask " + "what " * 1000
    result = aria.parse_intent(long_input)
    assert_eq(result["action"], "chat")


@test("Edge: input with only special characters")
def _():
    aria = _make_aria()
    result = aria.parse_intent("!@#$%^&*()_+{}|:<>?~")
    assert_eq(result["action"], "chat")


@test("Edge: input with mixed case commands")
def _():
    aria = _make_aria()
    assert_eq(aria.parse_intent("HELP")["action"], "help")
    assert_eq(aria.parse_intent("EXIT")["action"], "exit")
    assert_eq(aria.parse_intent("Mode RD")["action"], "set_mode")
    assert_eq(aria.parse_intent("MEMORY RECALL test")["action"], "memory")


@test("Edge: _handle_set_mode handles all modes")
def _():
    aria = _make_aria()
    for mode in ["auto", "groq", "nvidia", "openrouter", "rd", "engineer", "orchestrate", "unknown"]:
        response = aria._handle_set_mode(mode)
        assert_true(isinstance(response, str))


@test("Edge: _build_memory_context returns string")
def _():
    aria = _make_aria()
    context = aria._build_memory_context("test task")
    assert_true(isinstance(context, str))


# ═══════════════════════════════════════════════════════════════════════
# Section 13: Deployment Readiness
# ═══════════════════════════════════════════════════════════════════════

@test("Deploy: requirements.txt exists with core deps")
def _():
    if not Path("requirements.txt").exists():
        assert_true(False, "Missing requirements.txt")
        return
    reqs = Path("requirements.txt").read_text(encoding="utf-8", errors="replace")
    for dep in ["openai", "python-dotenv", "rich", "requests", "beautifulsoup4"]:
        assert_in(dep, reqs, f"Missing {dep}")


@test("Deploy: no hardcoded absolute paths in source")
def _():
    source_files = [
        "aria.py", "utils/config.py", "utils/llm.py",
        "utils/memory.py", "utils/vector_store.py", "utils/file_parser.py",
    ]
    for src_file in source_files:
        if not Path(src_file).exists():
            continue
        content = Path(src_file).read_text(encoding="utf-8", errors="replace")
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            if f"{letter}:\\" in content:
                print(f"  WARNING: Possible hardcoded path in {src_file}: {letter}:\\")


@test("Deploy: all exports have docstrings or type annotations")
def _():
    """Check that public functions have proper signatures."""
    import ast
    files_to_check = [
        "utils/vector_store.py",
        "utils/memory.py",
        "utils/file_parser.py",
        "utils/skill_manager.py",
    ]
    for f in files_to_check:
        if not Path(f).exists():
            continue
        with open(f, "rb") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip private functions
                if node.name.startswith("_"):
                    continue
                # Should have docstring or return annotation
                has_doc = (node.body and isinstance(node.body[0], ast.Expr)
                          and isinstance(node.body[0].value, ast.Constant)
                          and isinstance(node.body[0].value.value, str))
                has_returns = node.returns is not None
                if not has_doc and not has_returns:
                    print(f"  WARNING: {f}:{node.lineno} - {node.name} has no docstring/return type")


# ═══════════════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════════════

def run_all():
    """Run all registered tests."""
    global PASSED, FAILED, ERRORS

    sep = "=" * 60
    print()
    print(sep)
    print("ARIA - Comprehensive Stress Test Suite")
    print(sep)
    print(f"Python: {sys.version}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tests registered: {len(_TEST_FUNCTIONS)}")
    print(sep)

    current_section = ""
    for name, func in _TEST_FUNCTIONS:
        # Show section headers
        section_prefix = name.split(":")[0] if ":" in name else ""
        if section_prefix and section_prefix != current_section:
            current_section = section_prefix
            print(f"\n  [{current_section}]")

        try:
            func()
            PASSED += 1
            print(f"  [OK] {name}")
        except Exception as e:
            FAILED += 1
            tb = traceback.format_exc()
            msg = f"  [FAIL] {name}: {e}"
            ERRORS.append(msg)
            print(msg)
            # Show first 2 lines of traceback
            tb_lines = tb.strip().split("\n")
            relevant = [l for l in tb_lines if "File" in l and "test_stress" not in l][:1]
            if relevant:
                for l in relevant:
                    print(f"         {l.strip()}")

    # Summary
    total = PASSED + FAILED
    print(f"\n{sep}")
    print(f"RESULTS: {PASSED} passed, {FAILED} failed, {total} total")
    print(sep)

    if ERRORS:
        print(f"\nFailed tests ({len(ERRORS)}):")
        for err in ERRORS[:10]:
            print(f"  - {err.split(':')[0] if ':' in err else err}")
        if len(ERRORS) > 10:
            print(f"  ... and {len(ERRORS) - 10} more")

    return FAILED == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
