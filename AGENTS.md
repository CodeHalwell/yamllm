# Repository Guidelines

## Project Structure & Module Organization
- `yamllm/`: Core library
  - `core/`: LLM orchestration, config parser, unified provider layer (`providers/`), and main `LLM` class
  - `tools/`: Built‑in tools and `ToolManager`
  - `ui/`: Optional Rich‑based CLI renderers
  - `memory/`: Conversation + vector store utilities
- `examples/`: Small runnable demos (OpenAI, Google, Mistral, OpenRouter, etc.)
- `.config_examples/`: Example YAML configs (edit and export API keys via `.env`)
- `tests/`: Unit tests
- `docs/`: How‑to, API, and integration notes

## Build, Test, and Development Commands
- Install (editable): `uv pip install -e .`
- Run examples: `uv run examples/openai_example.py`
- Tests (quiet): `uv run -m pytest -q`
- Lint: `uv run ruff check .`
- Format check: `uv run black --check .`
- Types: `uv run mypy yamllm`

## Coding Style & Naming Conventions
- Python 3.10+; PEP 8; 4‑space indentation
- Names: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- Keep modules focused and small; avoid cross‑layer imports (UI → core only)
- Use `ruff` and `black`; keep diffs minimal and scoped

## Testing Guidelines
- Framework: `pytest`
- Location: `tests/` with files named `test_*.py`
- Prefer unit tests close to changed functionality; mock external I/O and network
- Validate tool schemas and provider behaviors with minimal tokens/fixtures

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject; include “why” and scope in body when needed
- PRs: clear description, linked issues, screenshots/CLI output for UX changes
- Include tests/docs for user‑facing changes; note any follow‑ups or known limitations

## Security & Configuration Tips
- Never commit API keys; use `.env` and environment variables (e.g., `OPENAI_API_KEY`, `WEATHER_API_KEY`, `OPENROUTER_API_KEY`)
- Example configs live in `.config_examples/`; copy and customize locally

## Architecture Overview (for Contributors)
- Single provider abstraction under `yamllm/providers/*` with a `ProviderFactory`
- `LLM` coordinates config, memory, tools, providers; UI is optional and separate
- Tools are registered via `ToolManager`; prefer JSON‑schema signatures and safe I/O

## Manifesto Alignment (Working Agreement)
- Purpose: “A human terminal for serious work and playful intelligence.” Build a fast, joyful terminal chat with tools, memory, and streaming by default.
- Philosophy: practical defaults, beautiful terminal output, agentic workflows, interoperability (MCP), async everywhere, user respect (privacy, clear permissions).
- Non‑goals: not an IDE, no lock‑in, avoid sprawling configuration; clarity over cleverness.

## Product Promises (What “Good” Looks Like)
- Effortless onboarding: install → configure → chat in 10–20 lines or a single CLI command; YAML config with sensible defaults.
- Terminal worth talking to: themes (colours, banners, bubbles), streaming with tidy chunking, handy commands (`/save`, `/clear`, `/theme`).
- Thinking in the open: `off|on|auto` with adaptive depth; fast replies for simple prompts; brief streamed plans for complex ones; redact internal reasoning in logs.
- Agentic tools: developer‑centric tool library; smart routing, progress streaming, guardrails (allowlists, timeouts, confirmations).
- MCP first‑class: client to external MCP servers; host mode to expose yamllm’s tools; namespacing, auth, health, reconnection.
- Async architecture: everything streams; overlap planning, tools, and tokens; cancellation and backpressure in the renderer.
- Memory & logging: SQLite conversation store; optional local vector memory; telemetry off by default; secrets masked and never printed.
- Reliability: clear compact errors, backoff with jitter, session snapshots to preserve chats on crash.
- Extensibility: tools/themes/providers as small plugins; schema‑first contracts with streaming events; simple YAML themes.
- Quality gates: UI snapshots, latency harness, tool conformance, MCP contract tests, no‑regression prompt pack.

## Acceptance Criteria (User‑Visible Contract)
- Typing “hello” streams a reply in < 400 ms with no visible thinking.
- “Refactor this…” shows thinking for ≤ 3.5 s, then streams highlighted code.
- Browse task (`web_search` → `web_scrape`) streams progress and a compact summary with collapsible details.
- `/mcp list` shows at least one server; invoking `mcp:*` streams like local tools.
- Switching theme at runtime updates bubbles, colours, and banner without restart.
- 10–20 lines of code yields full streaming chat with tools and memory.

## Performance Targets
- First token < 350 ms (tools off) / < 600 ms (tools on).
- Tool first byte < 0.5–0.9 s.
- Thinking panel appears < 120 ms after input.
- No blocking operations on async paths; HTTP/2 + pooled connections for lower latency.

## Contributor Priorities (Day‑to‑Day)
- Prefer async‑first implementations with streaming everywhere; add sync wrappers only when necessary.
- Keep tool IO safe: allowlists, timeouts, confirmations; log succinctly with secret masking.
- Favor provider‑agnostic interfaces; implement providers via the minimal streaming contract.
- Keep UI polish in the library (themes, formatting) rather than in examples; avoid bespoke Rich/Textual code in app code.
- Tests before complex refactors; prioritize latency and streaming regressions; mock external I/O and providers.

## Roadmap Slices (Deliver Value Early)
- v0.1: CLI `yamllm run`, OpenAI streaming, two themes, core tools (file_search, http, web_search, web_scrape, python), thinking `off|on|auto`, basic MCP client (stdio).
- v0.2: git, code_search, fs, code_run, SQL; MCP host mode + multi‑client; vector memory; guarded shell.
- v0.3: perf polish (HTTP/2 pooling, smarter routing); accessibility theme; plugin registry; Gemini, Mistral, DeepSeek adapters.

## Quality Gates (PR Ready Checklist)
- Meets acceptance criteria for the touched area; no UI regressions in snapshots.
- Streaming preserved; no added blocking calls on async paths.
- Tool schemas validated; timeouts and cancellation covered by tests where applicable.
- MCP changes verified against reference servers where relevant.
- Secrets masked; telemetry unchanged (off by default) unless explicitly opted-in.

## Current Architecture Snapshot
- `yamllm/core/llm.py` owns orchestration: config load + validation, provider selection, tool orchestration, thinking manager, streaming callbacks, usage tracking, and memory writes.
- Provider layer lives under `yamllm/providers/*` with a `ProviderFactory` that normalises sync and async providers (OpenAI, Azure, Anthropic, Google, Mistral, DeepSeek, OpenRouter) and exposes capability flags via `providers/capabilities.py`.
- Tools flow through `core/tool_orchestrator.py`, `tools/thread_safe_manager.py`, and `tools/security.py`, giving JSON-schema signatures, per-call sandboxing, and thread-safe execution with concurrency limits.
- Memory is handled by `core/memory_manager.py` and `yamllm/memory/conversation_store.py`, combining SQLite transcript storage with a FAISS-backed vector store plus CLI migration support.
- Terminal UX is provided by `yamllm/ui/chat.py` and `ui/themes.py`, delivering Rich streaming, theme switching, and tool/thinking event panels; CLI entry points are centralised in `yamllm/cli.py`.

## Capabilities Implemented
- Streaming chat with tool usage, thinking traces, and UI events is wired end-to-end for OpenAI-compatible providers; tool execution emits structured callbacks for renderers.
- Built-in tool library covers web search/scrape, weather, conversions, file IO, UUID/random generators, etc., backed by shared networking and security guards.
- MCP connectors (`yamllm/mcp/*.py`) support HTTP, WebSocket, and stdio transports, convert remote schemas into local tool definitions, and stream results back through the orchestrator.
- Memory subsystem captures both text history and embeddings with caching and OpenAI fallback, including dimension checks and migration helpers in the CLI.
- Extensive pytest suite (`tests/`) exercises config parsing, provider adapters, tool security, thinking heuristics, streaming callbacks, and thread safety; most integrations are mocked for offline runs.

## Identified Gaps and Risks
- Async story is partial: `core/async_llm.py` only supports OpenAI today and is not integrated into the CLI or provider-agnostic pathways; broader async streaming remains on the roadmap.
- Tool metadata manager in `core/tool_management.py` overlaps with `tools.manager.ToolManager`, leading to dual abstractions and potential confusion for contributors.
- Tool return payloads are not fully uniform (some dict, some stringified JSON), increasing formatting work in UIs and risking inconsistent downstream behaviour.
- Embedding fallback always leans on OpenAI (`core/llm.py`), so non-OpenAI installs still need an OpenAI key unless provider embeddings succeed; consider documenting or introducing pluggable backends.
- Thinking manager currently reuses the primary model and streams via provider completions; there is limited guard for providers without streaming thinking support, so failure handling relies on debug logging.
- Performance hooks exist (usage tracking, retry/backoff) but there is no telemetry or latency measurement aligned with manifesto targets, and several paths remain blocking (memory searches, tool execution dispatch).

## Testing and Tooling Notes
- Test suite relies on `tests/mock_config.yaml` and heavy use of `unittest.mock` to isolate provider calls; `test_mcp.py.disabled` indicates pending work to stabilise full MCP coverage.
- CLI utilities (`yamllm/cli.py`) expose `yamllm run`, tool registry inspection, and FAISS index migration; docs (`README.md`, `improvement_plan.md`) give onboarding guidance but should stay in sync with evolving defaults.
- Repository already includes safety guards (secret masking, network allowlists); contributors should expand tests when touching these areas to keep regressions out.
