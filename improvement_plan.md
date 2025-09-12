# YAMLLM Comprehensive Improvement Plan

## Executive Summary

This comprehensive improvement plan aligns yamllm with its manifesto vision of becoming "a human terminal for serious work and playful intelligence." The review identified critical bugs blocking core functionality, significant gaps in user experience features, and architectural improvements needed to achieve the manifesto's goals of effortless onboarding, beautiful terminal output, low-latency async operations, and first-class MCP support.

### Current State vs. Vision
- **Current**: Complex configuration, blocking operations, broken MCP, limited tools, no UI polish
- **Vision**: 10-20 lines to chat, beautiful themes, <350ms latency, full MCP ecosystem, rich tools
- **Gap**: ~70% feature completion needed, critical bugs blocking core functionality

### Investment Required
- **Timeline**: 12 weeks to manifesto-ready v1.0
- **Effort**: 2-3 developers full-time or 4-5 part-time
- **Priority**: Fix critical bugs first (2 weeks), then parallel streams for UI/async/tools

### Expected Outcomes
- **Week 2**: All critical bugs fixed, basic functionality restored
- **Week 6**: Async architecture, Rich UI, streaming chat working
- **Week 10**: Full tool suite, MCP working, CLI polished
- **Week 12**: Performance targets met, >80% test coverage, v1.0 ready

## Quick Wins (Can implement immediately)

### Day 1-3 Fixes
1. **Fix streaming+tools method name** (1 hour)
   - Change `process_tool_calls_stream` to `process_streaming_tool_calls`
2. **Fix Anthropic tool parsing** (2 hours)
   - Parse content parts for `tool_use` type
3. **Fix MCP async/await** (4 hours)
   - Add proper await calls, convert to sync or use httpx
4. **Add API key masking** (2 hours)
   - Mask keys in logs and error messages
5. **Fix path traversal** (1 hour)
   - Add `expanduser` before `realpath`

### Week 1 Improvements
1. **Add basic Rich integration** (1 day)
   - Console output with colors and formatting
2. **Create `yamllm run` command** (1 day)
   - Quick-start CLI for instant chat
3. **Fix vector store dimensions** (4 hours)
   - Better error messages and migration path
4. **Add 3 example configs** (2 hours)
   - Minimal working examples for each provider
5. **Clean up duplicate code** (1 day)
   - Remove llm_legacy.py, llm_old.py

## Manifesto Gap Analysis

### Critical Gaps vs. Manifesto Promises

1. **"Effortless onboarding" (10-20 lines to chat)** - Currently requires complex configuration
2. **"Beautiful terminal output"** - No Rich/Textual integration, no themes, no streaming UI
3. **"Async everywhere"** - Most operations are blocking; no HTTP/2, no connection pooling
4. **"First token < 350ms"** - No performance optimization or monitoring
5. **"MCP as first-class citizen"** - MCP implementation is broken (async misuse)
6. **"Developer-centric tools"** - Limited tools, no shell/git/SQL integration
7. **"Thinking in the open"** - No adaptive thinking display
8. **"Memory and logging that help"** - Vector store has critical bugs, no export features

## P0 — Critical bugs and correctness issues (fix first)

- **Streaming tool pipeline not used due to wrong method name**
  - In `yamllm/core/llm.py` the check uses `process_tool_calls_stream` but the provider exposes `process_streaming_tool_calls`.
  - Impact: forces a slow fallback and may break expected streaming+tools UX.
  - Fix: call `process_streaming_tool_calls` when available.

- **Anthropic tool-call extraction likely wrong**
  - `_extract_tool_calls` in `yamllm/core/llm.py` expects `response["tool_calls"]` for Anthropic, but Claude returns tool calls as content parts (`{"type":"tool_use","name":...,"input":...}`) inside the assistant message. Provider `format_tool_calls` already knows how to convert.
  - Impact: tools may never be executed for Anthropic.
  - Fix: parse `response["content"]` and pass parts with `tool_use` to `format_tool_calls` (or let provider surface a normalized structure consistently).

- **MCP client/connector async misuse**
  - `yamllm/mcp/client.py` calls `MCPConnector.discover_tools()` without `await`, but the connector function is `async def`.
  - Several connector methods are `async def` but perform blocking `requests.*` calls.
  - Impact: conversion will receive coroutines instead of tool arrays; MCP tools won’t work and may raise at runtime; async methods block the event loop.
  - Fix: make connector methods synchronous (HTTP only) or use `aiohttp`/`httpx` async clients; ensure calls are properly `await`ed or refactor the client to be async.

- **Async provider initialization in `LLM` hard-coded to OpenAI**
  - `yamllm/core/llm.py::_initialize_async_provider` only supports OpenAI, despite async providers existing for others and being mapped by `ProviderFactory.create_async_provider`.
  - Impact: `LLM.aquery` unusable for non-OpenAI providers.
  - Fix: delegate to `ProviderFactory.create_async_provider` with configured provider, mirroring sync.

- **Retry decorator catches the wrong exceptions**
  - `ErrorHandler.with_retry` defaults to custom `NetworkError, TimeoutError`, but `_initialize_provider` passes built-in `ConnectionError, TimeoutError` and does not import the custom `TimeoutError`. Also provider creation rarely does I/O.
  - Impact: retries likely never trigger where intended; misleading semantics.
  - Fix: use the custom exceptions where thrown, or remove retry from provider construction.

- **Anthropic embeddings are fabricated**
  - `yamllm/providers/anthropic.py::create_embedding` returns pseudo-random vectors.
  - Impact: corrupts similarity search and FAISS indices; inconsistent with OpenAI fallback logic in `LLM.create_embedding`.
  - Fix: raise `NotImplementedError` (or `ProviderError`) so `LLM.create_embedding` reliably falls back to OpenAI embeddings.

## P1 — Security and safety

- **Filesystem path handling could misinterpret `~` and tolerate oddities**
  - `yamllm/tools/security.py::validate_file_access` does not `expanduser`, and checks `Path.parts` for `"~"`. Also consider explicitly blocking `0.0.0.0` and `169.254.0.0/16` in network checks.
  - Fix: `file_path = os.path.expanduser(file_path)` before `realpath`; expand blocked networks set.

- **Network tools allow large timeouts or implicit defaults scattered**
  - While `NetworkTool` sets a timeout default, some providers/tools still create `requests.Session()` directly (ok) but keep per-provider retry logic.
  - Fix: centralize timeouts/retries in `NetworkTool` and re-use it; ensure all network calls respect max timeout, retries, and backoff uniformly.

- **Localhost and internal IP blocking is good — add tests**
  - Add unit tests for `SecurityManager.check_network_permission` over IP, IPv6, and domains with subdomain matches.

## P1 — API/UX inconsistencies and design issues

- **Two ToolManager concepts with the same name**
  - `yamllm/tools/manager.py` and `yamllm/core/tool_management.py` both define a `ToolManager` but with different responsibilities.
  - Impact: confusing mental model and imports; increases future maintenance friction.
  - Fix: rename the CLI/metadata manager to `ToolRegistryManager` (or similar), or merge the metadata/registry surface into the thread-safe manager as read-only views.

- **Provider base interface signature mismatch**
  - `BaseProvider.__init__` requires `(api_key, model, base_url, **kwargs)` but actual providers don’t accept `model` in `__init__`.
  - Impact: misleading abstraction and potential future TypeErrors if instantiation follows base signature.
  - Fix: remove `model` from the abstract signature; ensure per-call `model` param is standard across methods.

- **Inconsistent tool result shapes**
  - Some tools return `Dict`, some return JSON `str` (e.g., `TimezoneTool`).
  - Impact: downstream formatters must guess; formatting functions already expect content strings.
  - Fix: standardize: tools should return `Dict` consistently; providers’ `format_tool_results` can stringify.

- **`LLM.__exit__` event-loop shutdown approach may fail on 3.12+**
  - Uses `asyncio.get_event_loop()` and `run_until_complete` which can raise in newer Python.
  - Fix: prefer `asyncio.run` in a fresh loop when needed, or use `anyio` for compatibility; guard against already running loop.

- **`parse_yaml_config` prints to stdout on error**
  - Library code should avoid `print`; it already re-raises.
  - Fix: log via the project logger or just raise.

## P2 — Performance and resilience

- **Memory similarity gating can be more robust**
  - `LLM._prepare_messages` fetches embeddings twice per request and searches FAISS synchronously.
  - Improve by caching last N embeddings, avoid duplicate embed calls, and move vector search to a small thread.

- **Tool selection heuristics are string-match only**
  - Consider lightweight intent classification or a ruleset that looks at structured content to avoid unnecessary enablement.

- **Vector-store migration UX**
  - `yamllm/cli.py::migrate_index` prints and returns—consider offering an interactive confirm and dry-run diff of dims/paths.

## P2 — Developer experience and testing

- **Adopt ProviderFactory for async in `LLM`**
  - Once the async provider path is unified, add tests for `LLM.aquery` across providers (skip via env when keys missing) mirroring `tests/integration/test_providers_integration.py`.

- **Add tests for streaming+tools**
  - Verify OpenAI path uses provider-native streaming tool pipeline once the method name is fixed.

- **Harmonize exceptions**
  - Providers raise `ProviderError` already; wire retry logic to these where it makes sense (e.g., on 429/5xx) rather than during object construction.

- **Trim unused heavy deps**
  - `matplotlib`, `seaborn`, `scikit-learn` appear unused in code. They substantially increase install time and footprint.
  - Fix: move to an optional extras group or remove.

## Suggested concrete fixes (by file)

- `yamllm/core/llm.py`
  - Use `ProviderFactory.create_async_provider` in `_initialize_async_provider` and await its context manager.
  - In `_handle_streaming_with_tools`, switch attribute check to `process_streaming_tool_calls`.
  - Update Anthropic tool-call extraction path to parse content parts.
  - Swap retry exceptions to custom ones if used; otherwise drop retry from provider init.
  - Replace stdout prints in `print_settings` with logger output or keep as-is but note it’s a UI helper.

- `yamllm/mcp/client.py` and `yamllm/mcp/connector.py`
  - Make client async (rename `discover_all_tools` to `async def` and `await` connector calls) OR convert connector methods to sync if you keep the client sync.
  - Replace `requests` with `httpx.AsyncClient` or `aiohttp` inside async functions.

- `yamllm/providers/anthropic.py`
  - Raise for `create_embedding` to trigger OpenAI fallback; or implement a real embedding path if available.
  - Ensure `format_tool_calls` accepts Claude content parts when passed directly from `LLM`.

- `yamllm/tools/security.py`
  - `expanduser` before `realpath`. Block `0.0.0.0` and link-local ranges. Add tests.

- `yamllm/core/tool_management.py`
  - Rename class to avoid confusion with `yamllm/tools/manager.py`. Expose read-only views for CLI.

- `yamllm/providers/base.py`
  - Remove `model` from `__init__` abstract signature; ensure method docs emphasize per-call `model` arg.

## DX/UX polish

- Add a `yamllm diagnose` CLI command to run quick checks (env vars, model availability, vector-store dims) and suggest remedies.
- Emit a clear warning when tool gating disables all tools for a prompt (toggle-able with a verbose flag).
- Improve error surfaces: e.g., when provider fallback to OpenAI occurs, print a one-liner with the original reason and how to fix config.
- Provide first-run `yamllm init` with optional creation of a `.env` template and minimal config in `~/.yamllm`.

## Manifesto-Aligned Feature Additions

### Terminal UI & Themes (Manifesto Goals 1, 2)
- **Implement Rich/Textual integration**
  - Create base `TerminalUI` class with Rich console
  - Add streaming message renderer with sentence chunking
  - Implement chat bubbles, timestamps, copy shortcuts
  - Create theme system (YAML-based) with 5+ starter themes
  - Add `/theme`, `/save`, `/clear` commands
  - ASCII banners and customizable chrome

### Async Architecture Overhaul (Manifesto Goal 6)
- **Full async conversion**
  - Convert all providers to async-first with sync wrappers
  - Implement HTTP/2 with connection pooling (httpx)
  - Add cancellation on keypress
  - Implement backpressure handling
  - Target: First token < 350ms, tool first byte < 0.9s

### Thinking Display System (Manifesto Goal 3)
- **Adaptive reasoning UI**
  - Implement `off|on|auto` modes
  - Add complexity detection for auto mode
  - Stream thinking panel with < 120ms appearance
  - Redact internal reasoning from logs
  - Show brief summaries for complex tasks

### Enhanced Tool Library (Manifesto Goal 4)
- **Developer tools expansion**
  - Add: `shell` (with guardrails), `git`, `sql`, `code_search`
  - Add: `archive`, `process`, `table_csv`, `image_info`
  - Add: `notebook_snippets`, `code_run`
  - Implement smart routing based on schema fit
  - Add progress streaming with spinners
  - Permission system with allowlists and confirmations

### MCP Implementation Fix & Enhancement (Manifesto Goal 5)
- **Complete MCP support**
  - Fix async/await issues in client and connector
  - Implement stdio, websocket, and HTTP transports
  - Add MCP host mode to expose yamllm tools
  - Implement namespacing, auth, health checks
  - Add reconnection logic
  - Create `/mcp list` command

### CLI & Quick Start (Manifesto Goal 1)
- **Streamlined onboarding**
  - Create `yamllm init` wizard for config generation
  - Add `yamllm run` for instant chat
  - Implement `yamllm diagnose` for troubleshooting
  - Create minimal working examples (10-20 lines)
  - Add `.env` template generation

### Memory & Export Features (Manifesto Goal 7)
- **Enhanced memory system**
  - Fix vector store dimension issues
  - Add conversation export (markdown, JSON)
  - Implement session snapshots for crash recovery
  - Add telemetry opt-in (anonymous only)
  - Mask secrets in all outputs

### Quality & Testing (Manifesto Goals 9, 10)
- **Comprehensive test suite**
  - Add UI snapshot tests for themes
  - Create latency test harness
  - Implement tool conformance tests
  - Add MCP contract tests
  - Create no-regression prompt pack
  - Target: >80% code coverage

## Roadmap and priorities

### Sprint 1 (Week 1-2): Critical Fixes & Foundation
- **P0**: Fix all critical bugs (streaming+tools, Anthropic, MCP async, provider init)
- **P0**: Fix vector store dimension handling and connection leaks
- **P0**: Security fixes (path traversal, API key masking)
- **P1**: Standardize error handling and exceptions

### Sprint 2 (Week 3-4): Async & Performance
- **P0**: Convert core to async-first architecture
- **P0**: Implement httpx with HTTP/2 and connection pooling
- **P1**: Add cancellation and backpressure handling
- **P1**: Optimize vector search and embedding caching

### Sprint 3 (Week 5-6): Terminal UI & UX
- **P0**: Integrate Rich/Textual for beautiful output
- **P0**: Implement streaming UI with chat bubbles
- **P1**: Create theme system with 5 starter themes
- **P1**: Add thinking display system (off|on|auto)

### Sprint 4 (Week 7-8): Tools & MCP
- **P0**: Fix MCP implementation completely
- **P0**: Add core developer tools (shell, git, sql)
- **P1**: Implement tool routing and progress streaming
- **P1**: Add MCP host mode

### Sprint 5 (Week 9-10): CLI & Onboarding
- **P0**: Create `yamllm init` and `yamllm run` commands
- **P0**: Write 10-20 line quickstart examples
- **P1**: Add `yamllm diagnose` command
- **P1**: Create comprehensive documentation

### Sprint 6 (Week 11-12): Polish & Testing
- **P0**: Add comprehensive test coverage
- **P1**: Performance optimization to meet latency targets
- **P1**: Add telemetry and monitoring
- **P2**: Final polish and bug fixes

## Acceptance Criteria (from Manifesto)

### User-Visible Contract
- [ ] Typing "hello" yields streamed reply in **< 400 ms** with no visible thinking
- [ ] "Refactor this..." prompt shows thinking for **≤ 3.5 s**, then streams highlighted code
- [ ] Browse task (web_search → web_scrape) shows progress and compact summary
- [ ] `/mcp list` shows at least one server; `mcp:*` streams results like local tools
- [ ] Theme switching at runtime updates bubbles, colors, banner without restart
- [ ] 10-20 lines of code gets user to full streaming chat with tools and memory

### Performance Targets
- [ ] First token < **350 ms** (tools off) / < **600 ms** (tools on)
- [ ] Tool first byte < **0.5-0.9 s**
- [ ] Thinking panel appears < **120 ms** after input
- [ ] Zero blocking operations in async code paths
- [ ] Connection pooling reduces latency by >30%

### Quality Gates
- [ ] UI snapshot tests for all themes
- [ ] Latency harness with deterministic tests
- [ ] Tool conformance: schema validation, timeout & cancellation
- [ ] MCP contract tests against reference servers
- [ ] No-regression prompt pack (greetings, coding, browsing, multi-tool)
- [ ] >80% test coverage
- [ ] Zero memory leaks in 24-hour stress test

## Success Metrics

### Developer Experience
- Time from install to first chat: < 5 minutes
- Lines of code for basic chat: 10-20
- Provider addition complexity: < 100 lines
- Tool plugin creation: < 50 lines
- Theme contribution: < 30 lines YAML

### User Experience  
- Perceived responsiveness: "instant" for simple, "working" indicator < 120ms
- Error clarity: actionable messages with recovery suggestions
- Memory usefulness: relevant context retrieval >85% accuracy
- Tool success rate: >95% for well-formed requests

### Technical Excellence
- API consistency: 100% providers follow base interface
- Async coverage: 100% I/O operations non-blocking
- Security: Zero path traversal, zero API key leaks
- Reliability: <0.1% crash rate, graceful degradation

## Implementation Notes

### Dependency Management
- **Core**: pyyaml, pydantic, httpx, rich, textual
- **Providers**: openai, anthropic[bedrock], google-generativeai, mistralai
- **Memory**: faiss-cpu, sqlite3 (built-in)
- **Tools**: duckduckgo-search, beautifulsoup4, requests (migrate to httpx)
- **Optional**: matplotlib, seaborn, scikit-learn (move to extras)

### Architecture Decisions
- **Async-first**: All new code async, sync wrappers for compatibility
- **Plugin system**: Entry points for tools, themes, providers
- **Streaming everywhere**: Server-sent events, chunked responses
- **Progressive enhancement**: Core works without optional deps

## Appendix — References

- Wrong method name (provider vs core):
  - Provider: `yamllm/providers/openai.py::process_streaming_tool_calls`
  - Core: `yamllm/core/llm.py::_handle_streaming_with_tools` checks `process_tool_calls_stream` (does not exist)
- Anthropic tool calls live in message content parts (`tool_use`) not at top-level `tool_calls`.
- MCP async issues: `yamllm/mcp/client.py::discover_all_tools` calls an async method without awaiting; connector uses `requests` in `async def` methods.
