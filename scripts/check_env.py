"""
check_env.py — Phase 0 environment verification for the Wikipedia RAG project.

Checks:
  1. Python version (>= 3.11)
  2. All required packages importable (fails fast with install hint)
  3. Ollama daemon reachable at localhost:11434
  4. Required models available in the Ollama registry: llama3.2:3b, nomic-embed-text
  5. Metal / GPU acceleration status for each loaded model
  6. ChromaDB functional (in-memory smoke test)
  7. Memory headroom estimate via /proc or psutil fallback

Run with:
    python scripts/check_env.py
"""

from __future__ import annotations

import importlib
import json
import platform
import subprocess
import sys
import urllib.error
import urllib.request
from typing import NamedTuple

# ── ANSI colours (macOS terminal always supports these) ──────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✔{RESET}  {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET}  {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✘{RESET}  {msg}")


def section(title: str) -> None:
    print(f"\n{BOLD}── {title} {'─' * (55 - len(title))}{RESET}")


# ── 1. Python version ─────────────────────────────────────────────────────────
section("1 · Python version")
major, minor = sys.version_info.major, sys.version_info.minor
version_str = f"{major}.{minor}.{sys.version_info.micro}"
if (major, minor) >= (3, 11):
    ok(f"Python {version_str} — meets requirement (≥ 3.11)")
else:
    fail(f"Python {version_str} — need ≥ 3.11.  Install with: brew install python@3.11")
    sys.exit(1)

arch = platform.machine()
ok(f"Architecture: {arch} ({'Apple Silicon ✓' if arch == 'arm64' else 'Intel — Metal not available'})")


# ── 2. Required packages ──────────────────────────────────────────────────────
section("2 · Required packages")

REQUIRED: list[tuple[str, str]] = [
    ("wikipediaapi", "wikipedia-api"),
    ("chromadb", "chromadb"),
    ("ollama", "ollama"),
    ("yaml", "PyYAML"),
]
OPTIONAL: list[tuple[str, str]] = [
    ("streamlit", "streamlit"),
    ("pytest", "pytest"),
]

missing_required: list[str] = []

for module_name, pip_name in REQUIRED:
    try:
        importlib.import_module(module_name)
        ok(f"{pip_name}")
    except ImportError:
        fail(f"{pip_name} — not found.  Run: pip install {pip_name}")
        missing_required.append(pip_name)

for module_name, pip_name in OPTIONAL:
    try:
        importlib.import_module(module_name)
        ok(f"{pip_name} (optional)")
    except ImportError:
        warn(f"{pip_name} (optional) — not installed.  Run: pip install {pip_name}")

if missing_required:
    print(f"\n{RED}Required packages missing — run:{RESET}")
    print(f"  pip install {' '.join(missing_required)}")
    sys.exit(1)


# ── 3. Ollama daemon reachability ─────────────────────────────────────────────
section("3 · Ollama daemon")
OLLAMA_BASE = "http://localhost:11434"

ollama_ok = False
try:
    with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=5) as resp:
        tags_data = json.loads(resp.read().decode())
    ollama_ok = True
    ok(f"Ollama daemon reachable at {OLLAMA_BASE}")
except urllib.error.URLError as exc:
    fail(f"Ollama daemon not running — {exc}")
    print(f"\n  {YELLOW}Fix:{RESET} Open a new terminal and run:  ollama serve")
    print("  Then re-run this script.\n")


# ── 4. Required models ────────────────────────────────────────────────────────
section("4 · Ollama model registry")

REQUIRED_MODELS = ["llama3.2:3b", "nomic-embed-text"]


class ModelInfo(NamedTuple):
    name: str
    size_gb: float
    present: bool


available_models: dict[str, ModelInfo] = {}

if ollama_ok:
    for m in tags_data.get("models", []):
        full_name: str = m.get("name", "")
        size_bytes: int = m.get("size", 0)
        size_gb = size_bytes / 1e9
        available_models[full_name] = ModelInfo(full_name, size_gb, True)

    for required in REQUIRED_MODELS:
        found = any(key.startswith(required.split(":")[0]) for key in available_models)
        if found:
            match = next(k for k in available_models if k.startswith(required.split(":")[0]))
            info = available_models[match]
            ok(f"{required} — present ({info.size_gb:.1f} GB on disk)")
        else:
            fail(f"{required} — not found")
            print(f"\n  {YELLOW}Fix:{RESET} Run:  ollama pull {required}\n")
else:
    warn("Skipping model checks — daemon unreachable")


# ── 5. Metal / GPU acceleration ───────────────────────────────────────────────
section("5 · Metal / GPU acceleration")

if platform.machine() != "arm64":
    warn("Not running on Apple Silicon — Metal not available. ChromaDB and Ollama will use CPU.")
else:
    ok("Apple Silicon (arm64) detected — Ollama will use Metal automatically")

if ollama_ok:
    try:
        result = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        ps_output = result.stdout.strip()
        if ps_output and "NAME" in ps_output:
            lines = ps_output.splitlines()
            print()
            for line in lines:
                print(f"  {line}")
            if "100% GPU" in ps_output:
                ok("At least one model is using 100% GPU (Metal)")
            elif "GPU" in ps_output:
                ok("GPU column present — models loaded into Metal")
            else:
                warn("No model currently loaded. Load a model and re-run to verify GPU usage.")
        else:
            warn("No models currently running. Start a model to verify Metal: ollama run llama3.2:3b 'ping'")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        warn("Could not run 'ollama ps' — verify GPU usage manually after starting a model")
else:
    warn("Skipping GPU check — daemon unreachable")


# ── 6. ChromaDB smoke test ────────────────────────────────────────────────────
section("6 · ChromaDB smoke test (in-memory)")

try:
    import chromadb  # noqa: PLC0415

    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection("_env_check")
    col.add(
        ids=["smoke-1"],
        embeddings=[[0.1] * 768],
        documents=["env check document"],
        metadatas=[{"entity_type": "place", "entity_name": "TestPlace"}],
    )
    results = col.query(query_embeddings=[[0.1] * 768], n_results=1)
    assert results["ids"][0][0] == "smoke-1", "ID mismatch in smoke test"
    client.delete_collection("_env_check")
    ok(f"ChromaDB {chromadb.__version__} — upsert + query + metadata filter OK")
except Exception as exc:  # noqa: BLE001
    fail(f"ChromaDB smoke test failed: {exc}")


# ── 7. Available memory estimate ──────────────────────────────────────────────
section("7 · System memory")

try:
    result = subprocess.run(
        ["sysctl", "-n", "hw.memsize"],
        capture_output=True,
        text=True,
        timeout=3,
    )
    total_bytes = int(result.stdout.strip())
    total_gb = total_bytes / 1e9
    if total_gb >= 8:
        ok(f"Total unified memory: {total_gb:.1f} GB — meets 8 GB requirement")
    else:
        warn(f"Total unified memory: {total_gb:.1f} GB — may be tight; monitor Activity Monitor")

    vm_result = subprocess.run(
        ["vm_stat"],
        capture_output=True,
        text=True,
        timeout=3,
    )
    free_pages = 0
    page_size = 16384
    for line in vm_result.stdout.splitlines():
        if "Pages free" in line or "Pages inactive" in line:
            parts = line.split(":")
            if len(parts) == 2:
                free_pages += int(parts[1].strip().rstrip("."))
    approx_free_gb = (free_pages * page_size) / 1e9
    msg = f"Approx. available: ~{approx_free_gb:.1f} GB (free + inactive pages)"
    if approx_free_gb >= 4:
        ok(msg)
    else:
        warn(msg + " — close other apps before running ingest")

except Exception as exc:  # noqa: BLE001
    warn(f"Could not measure memory: {exc}")


# ── Summary ───────────────────────────────────────────────────────────────────
section("Summary")

if not ollama_ok:
    print(f"\n{RED}{BOLD}ACTION REQUIRED:{RESET} Start the Ollama daemon with:  ollama serve")
    print("Then re-run this script to confirm all checks pass before moving to Phase 1.\n")
else:
    print(f"\n{GREEN}{BOLD}Environment looks ready.{RESET}")
    print("Next steps (TODO.md Phase 0):")
    print("  0.3  ollama pull llama3.2:3b  &&  ollama pull nomic-embed-text")
    print("  0.4  ollama run llama3.2:3b 'ping'  →  verify '100% GPU' in ollama ps")
    print("  Then begin Phase 1 — Ingest & Store.\n")
