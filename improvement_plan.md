# YAMLLM Comprehensive Improvement Plan

## Executive Summary

This comprehensive improvement plan consolidates findings from three major code reviews (Claude, Codex, and Gemini) to align YAMLLM with its manifesto vision. The reviews identified critical architectural issues, performance bottlenecks, security vulnerabilities, and user experience gaps that must be addressed to achieve the project's goals.

### Combined Review Findings
- **Claude Review**: Focus on code complexity, error handling, performance optimization, and security
- **Codex Review**: Tool routing reliability, thinking pipeline integration, documentation gaps
- **Gemini Review**: Code duplication, unified async architecture, provider abstraction consistency

### Current State vs. Vision
- **Current**: Complex configuration, blocking operations, broken MCP, inconsistent tool routing, 1,548-line main class
- **Vision**: 10-20 lines to chat, beautiful themes, <350ms latency, reliable tool execution, clean architecture
- **Gap**: ~70% feature completion needed, critical architectural refactoring required

### Investment Required
- **Timeline**: 12 weeks to manifesto-ready v1.0
- **Effort**: 2-3 developers full-time or 4-5 part-time
- **Priority**: Architecture refactoring and critical bugs first (3 weeks), then parallel feature streams

### Expected Outcomes
- **Week 3**: Architecture refactored, critical bugs fixed, tool routing reliable
- **Week 6**: Async-first core, Rich UI, streaming working properly
- **Week 10**: Full tool suite, MCP working, performance targets met
- **Week 12**: >80% test coverage, documentation complete, v1.0 ready

## Quick Wins (Can implement immediately)

### Day 1-3 Fixes
- [x] Fix streaming+tools method name (1 hour)
  - Change `process_tool_calls_stream` to `process_streaming_tool_calls`
- [x] Fix Anthropic tool parsing (2 hours)
  - Parse content parts for `tool_use` type (confirmed in core/provider path)
- [x] Fix MCP async/await (4 hours)
  - Add proper await calls, convert to sync or use httpx
- [x] Add API key masking (2 hours)
  - Baseline masking in core logs; audit remaining surfaces
- [x] Fix path traversal (1 hour)
  - Add `expanduser` before `realpath`; null-byte checks and internal IP/domain blocks
### Week 1 Improvements
- [x] Add basic Rich integration (1 day)
  - Console output with colors and formatting
- [x] Create `yamllm run` command (1 day)
  - Quick-start CLI for instant chat
- [x] Fix vector store dimensions (4 hours)
  - Better error messages and migration path
- [x] Add 3 example configs (2 hours)
  - Minimal working examples for each provider
- [x] Clean up duplicate code (1 day)
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

## P0 — Critical Architecture & Bugs (Fix First)

### Core Architecture Refactoring (Claude + Gemini Reviews) ✅ VERIFIED

- [ ] **Refactor monolithic LLM class** (yamllm/core/llm.py:1547 lines - CONFIRMED)
  - [ ] Split into ResponseOrchestrator, StreamingManager, ToolSelector classes
  - [ ] Target: Reduce main class to <500 lines
  - [ ] Extract tool filtering logic (lines 616-752)
  - [ ] Separate thinking mode processing (49 methods total in class)
  - [ ] Timeline: 2-3 sprints

- [ ] **Unify sync/async architecture** (Gemini Review) ✅ VERIFIED
  - [ ] Merge LLM (1547 lines) and AsyncLLM (270 lines) into single async-first class
  - [ ] Provide sync wrapper methods (query_sync)
  - [ ] Remove code duplication (AsyncLLM lacks memory, tool orchestration, thinking)
  - [ ] Create single BaseProvider with both sync/async abstract methods
  - [ ] Timeline: 1-2 sprints

- [ ] **Consolidate dual ToolManager concepts** (All Reviews) ✅ VERIFIED
  - [ ] Rename tools/manager.py:ToolManager to ToolExecutor (execution-focused)
  - [ ] Keep core/tool_management.py:ToolRegistryManager for CLI/metadata
  - [ ] Unify tool registration, execution, and metadata systems
  - [ ] Timeline: 1 sprint

### Critical Bug Fixes (Codex + Previous Reviews)

- [x] Streaming tool pipeline not used due to wrong method name
  - In `yamllm/core/llm.py` the check used `process_tool_calls_stream` but the provider exposes `process_streaming_tool_calls`. Fixed and added `tool_executor` wiring.

- [x] Anthropic tool-call extraction likely wrong
  - `_extract_tool_calls` handles `tool_use` parts; provider errors and streaming normalized.

- [x] MCP client/connector async misuse
  - `yamllm/mcp/client.py` should consistently await connector methods; connector methods should avoid blocking I/O.

- [x] Async provider initialization in `LLM` hard-coded to OpenAI
  - Refactored `_initialize_async_provider` to use `ProviderFactory.create_async_provider`.

- [x] Retry decorator catches the wrong exceptions
  - Align retry surfaces and exceptions where appropriate.

- [x] Anthropic embeddings are fabricated
  - `yamllm/providers/anthropic.py::create_embedding` now raises `ProviderError` to trigger fallback to OpenAI embeddings.

### Tool Routing Reliability (Codex Review) ✅ PARTIALLY VERIFIED

- [ ] **Fix inconsistent tool selection** (yamllm/core/llm.py:616-709) ⚠️ CONFIRMED ISSUE
  - [ ] When user explicitly requests a tool, return ONLY that tool definition
  - [ ] Set tool_choice="required" for explicit tool requests
  - [ ] Add unit tests for _filter_tools_for_prompt and _determine_tool_choice (MISSING - no tests found)
  - [ ] Fix OpenAI ignoring forced tool_choice parameter
  - [ ] Timeline: 1 week

- [ ] **Connect thinking pipeline to UI** (yamllm/core/llm.py:753-820) ✅ CONFIRMED ISSUE
  - [ ] Wire up event_callback in CLI examples
  - [ ] Show thinking plans and tool activity to users
  - [ ] Implement "thinking in the open" manifesto promise
  - [ ] Timeline: 1 week

## P1 — Security and Error Handling (All Reviews)

### Security Enhancements (Claude + Previous Reviews) ✅ VERIFIED

- [x] **Filesystem path handling hardened** ✅ CONFIRMED IMPLEMENTED
  - [x] `yamllm/tools/security.py::validate_file_access` uses `expanduser` before `realpath` (lines 55-56)
  - [x] Added null-byte checks (line 49-50) and internal IP blocking (lines 98-99)
  - [ ] Add comprehensive tests for SecurityManager.check_network_permission
  - [ ] Block additional networks: 0.0.0.0, 169.254.0.0/16, .local domains
  - [ ] Timeline: 1 week

- [ ] **Network security improvements**
  - [ ] Centralize timeouts/retries in NetworkTool across all tools
  - [ ] Add rate limiting for API calls (max_requests_per_minute)
  - [ ] Implement input validation for all tool parameters
  - [ ] Add audit logging for security events
  - [ ] Timeline: 2 weeks

- [x] **API key masking implemented**
  - [x] Baseline masking in core logs
  - [ ] Audit remaining surfaces for potential key leaks
  - [ ] Ensure no secrets in memory manager or conversation storage
  - [ ] Timeline: 1 week

### Error Handling Standardization (Claude Review)

- [ ] **Create unified exception hierarchy**
  - [ ] Standardize ProviderError across all providers
  - [ ] Implement consistent retry logic for transient failures
  - [ ] Add structured error messages with recovery suggestions
  - [ ] Remove inconsistent error swallowing (utility_tools.py:149)
  - [ ] Timeline: 1-2 sprints

- [ ] **Improve error surfaces**
  - [ ] Provider fallback warnings with actionable messages
  - [ ] Configuration validation error details
  - [ ] Tool execution failure recovery strategies
  - [ ] Timeline: 1 sprint

## P1 — Performance and Code Quality (Claude + All Reviews)

### Performance Optimization (Claude Review) ✅ VERIFIED

- [ ] **Enhance caching strategies** ✅ CONFIRMED BOTTLENECKS
  - [ ] Increase embedding cache from 64 to 1000 entries with TTL (confirmed lines 1342, 1360)
  - [ ] Implement connection pooling for HTTP-based providers
  - [ ] Cache tool definitions (currently regenerated 5x per request - confirmed lines 341, 346, 502, 509, 1306)
  - [ ] Optimize vector store searches (avoid on every query when not relevant)
  - [ ] Timeline: 2-3 sprints

- [ ] **Reduce computation overhead**
  - [ ] Move vector search to background thread
  - [ ] Avoid duplicate embedding calls in _prepare_messages
  - [ ] Implement smarter tool selection heuristics beyond string matching
  - [ ] Add instrumentation for tool execution queue pressure
  - [ ] Timeline: 2 sprints

### Code Quality Improvements (All Reviews)

- [x] **Provider base interface signature corrected**
  - [x] Removed `model` from `BaseProvider.__init__` abstract signature
  - [x] Ensured per-call `model` parameter is standard across methods

- [x] **Tool result shapes standardized**
  - [x] Tools return `Dict` consistently; providers stringify via `format_tool_results`

- [x] **Event-loop shutdown robustness**
  - [x] Uses `asyncio.run` with anyio compatibility for Python 3.12+

- [x] **Library code print statements removed**
  - [x] `parse_yaml_config` uses logging instead of stdout

### Documentation and UX (Codex Review)

- [ ] **Update README and documentation**
  - [ ] Fix install instructions (remove non-existent `yamllm-core` package reference)
  - [ ] List current 20+ built-in tools (not just 4)
  - [ ] Add up-to-date quick-start examples
  - [ ] Document CLI debugging features (logging, tool output handling)
  - [ ] Timeline: 1 week

- [ ] **Improve web scraper UX** (yamllm/tools/utility_tools.py:516)
  - [ ] Summarize scrape results instead of dumping raw HTML
  - [ ] Add pagination/truncation utilities
  - [ ] Pair with web_headlines for better UX
  - [ ] Timeline: 1 week

- [ ] **CLI logging improvements**
  - [ ] Enable console logging in example configs
  - [ ] Add CLI flags for debugging
  - [ ] Show graceful messaging for empty tool output
  - [ ] Timeline: 1 week

## P2 — Testing and Quality Assurance (All Reviews)

### Test Coverage Expansion (Claude + Codex Reviews) ✅ VERIFIED

- [ ] **Add missing critical test coverage** ⚠️ CONFIRMED GAPS
  - [ ] Unit tests for _filter_tools_for_prompt and _determine_tool_choice (MISSING - no tests found)
  - [ ] Integration tests for CLI flows with mock provider responses
  - [ ] MCP connector failure scenarios and edge cases (basic test_mcp_async.py exists)
  - [ ] Tool circular dependency detection
  - [ ] Provider fallback mechanisms under load
  - [ ] Concurrent memory access and thread safety
  - [ ] Timeline: 2 sprints

- [ ] **Add specialized test suites**
  - [ ] UI snapshot tests for all themes
  - [ ] Latency test harness for performance targets
  - [ ] Tool conformance tests (schema validation, timeout, cancellation)
  - [ ] MCP contract tests against reference servers
  - [ ] No-regression prompt pack (greetings, coding, browsing, multi-tool)
  - [ ] Security tests for path traversal and API key handling
  - [ ] Timeline: 2-3 sprints

### Dependency and Build Quality (Gemini Review)

- [x] **Cleaned up heavy dependencies**
  - [x] Moved matplotlib, seaborn, scikit-learn to optional extras
  - [x] Reduced install footprint and complexity

- [ ] **CI/CD improvements**
  - [ ] Enforce lint/test/mypy in GitHub Actions
  - [ ] Add automated performance regression tests
  - [ ] Implement security scanning for dependencies
  - [ ] Timeline: 1 sprint

## P2 — Developer experience and testing

- [x] Adopt ProviderFactory for async in `LLM`
  - Async provider path unified via `ProviderFactory.create_async_provider`; tests can be added for `LLM.aquery` across providers.

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

- [x] Add a `yamllm diagnose` CLI command to run quick checks (env vars, vector-store dims) and suggest remedies.
- [ ] Emit a clear warning when tool gating disables all tools for a prompt (toggle-able with a verbose flag).
- [ ] Improve error surfaces: e.g., when provider fallback to OpenAI occurs, print a one-liner with the original reason and how to fix config.
- [ ] Provide first-run `yamllm init` with optional creation of a `.env` template and minimal config in `~/.yamllm`.

## Manifesto-Aligned Feature Additions

### Terminal UI & Themes (Manifesto Goals 1, 2)
- [x] Implement Rich/Textual integration
  - Create base `TerminalUI` class with Rich console
  - Add streaming message renderer with sentence chunking
  - Implement chat bubbles, timestamps, copy shortcuts
  - Create theme system (YAML-based) with starter themes and CLI `theme` commands
  - `clear` command in chat; `/save` TBD
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

-### MCP Implementation Fix & Enhancement (Manifesto Goal 5)
- **Complete MCP support**
  - [ ] Fix async/await issues in client and connector
  - [ ] Implement stdio, websocket, and HTTP transports
  - [ ] Add MCP host mode to expose yamllm tools
  - [ ] Implement namespacing, auth, health checks
  - [ ] Add reconnection logic
  - [x] Create `/mcp list` command

### CLI & Quick Start (Manifesto Goal 1)
- **Streamlined onboarding**
  - [x] Create `yamllm init` wizard for config generation
  - [x] Add `yamllm run` for instant chat
  - [x] Implement `yamllm diagnose` for troubleshooting
  - [ ] Create minimal working examples (10-20 lines)
  - [ ] Add `.env` template generation

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

## Roadmap and Priorities (Updated with Combined Reviews)

### Sprint 1 (Week 1-3): Architecture Refactoring & Critical Fixes
- **P0**: Refactor monolithic LLM class (1,548 lines → <500 lines)
- **P0**: Unify sync/async architecture (merge LLM + AsyncLLM)
- **P0**: Consolidate dual ToolManager concepts
- **P0**: Fix tool routing reliability (explicit tool selection)
- **P0**: Connect thinking pipeline to UI (event callbacks)
- **P0**: Standardize error handling and exceptions

### Sprint 2 (Week 4-5): Security & Performance Foundation
- **P0**: Complete security hardening (network blocking, input validation)
- **P0**: Implement unified exception hierarchy
- **P0**: Enhance caching strategies (embeddings, tool definitions)
- **P1**: Add rate limiting and audit logging
- **P1**: Optimize vector search and embedding calls

### Sprint 3 (Week 6-7): Terminal UI & Streaming
- **P0**: Integrate Rich/Textual for beautiful output
- **P0**: Implement streaming UI with chat bubbles
- **P1**: Create theme system with 5 starter themes
- **P1**: Add thinking display system (off|on|auto)

### Sprint 4 (Week 8-9): Tools & MCP Enhancement
- **P0**: Fix MCP implementation completely
- **P0**: Improve web scraper UX (summarization, pagination)
- **P0**: Add core developer tools (shell, git, sql)
- **P1**: Implement tool routing and progress streaming
- **P1**: Add MCP host mode

### Sprint 5 (Week 10-11): CLI & Documentation
- **P0**: Update README and documentation (fix install instructions)
- **P0**: Create comprehensive CLI debugging features
- **P0**: Write 10-20 line quickstart examples
- **P1**: Add telemetry and monitoring features
- **P1**: Create comprehensive user guides

### Sprint 6 (Week 12): Testing & Polish
- **P0**: Add comprehensive test coverage (>80%)
- **P0**: Performance optimization to meet latency targets
- **P1**: CI/CD improvements (GitHub Actions, security scanning)
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

## Success Metrics (Updated with Combined Review Findings)

### Architecture Quality (Gemini + Claude Reviews)
- Main LLM class: < 500 lines (from current 1,548)
- Code duplication: 0 between sync/async implementations
- Tool management: Single unified system (not dual concepts)
- Provider abstraction: 100% consistency across all providers
- Error handling: Unified exception hierarchy across all components

### Performance Targets (Claude Review)
- First token: < 350 ms (tools off) / < 600 ms (tools on)
- Tool first byte: < 0.5-0.9 s
- Thinking panel appears: < 120 ms after input
- Embedding cache: 1000 entries with TTL (from current 64)
- Connection pooling: >30% latency reduction

### User Experience (Codex Review)
- Tool routing reliability: >99% for explicit tool requests
- Thinking visibility: Event callbacks wired in all CLI examples
- Documentation accuracy: Install instructions work, all tools listed
- Web scraper UX: Summarized results, not raw HTML dumps
- Error clarity: Actionable messages with recovery suggestions

### Developer Experience
- Time from install to first chat: < 5 minutes
- Lines of code for basic chat: 10-20
- Provider addition complexity: < 100 lines
- Tool plugin creation: < 50 lines
- Theme contribution: < 30 lines YAML

### Security & Quality (Claude + Previous Reviews)
- Security: Zero path traversal, zero API key leaks
- Input validation: 100% of tool parameters validated
- Test coverage: >80% with edge case coverage
- API consistency: 100% providers follow base interface
- Async coverage: 100% I/O operations non-blocking
- Reliability: <0.1% crash rate, graceful degradation

## Implementation Notes (Updated with Review Findings)

### Architectural Decisions (Based on Combined Reviews)
- **Async-first core**: Single LLM class with sync wrappers (Gemini recommendation)
- **Unified tool management**: Single ToolRegistryManager with ToolExecutor (All reviews)
- **Consistent provider interface**: Single BaseProvider with sync/async methods (Gemini)
- **Streaming everywhere**: Server-sent events, chunked responses
- **Progressive enhancement**: Core works without optional deps

### Dependency Management (Cleaned per Gemini Review)
- **Core**: pyyaml, pydantic, httpx, rich, textual
- **Providers**: openai, anthropic[bedrock], google-generativeai, mistralai
- **Memory**: faiss-cpu, sqlite3 (built-in)
- **Tools**: duckduckgo-search, beautifulsoup4, requests (migrate to httpx)
- **Optional extras**: matplotlib, seaborn, scikit-learn (moved to extras group)

### Security Architecture (Enhanced per Claude Review)
- **Input validation**: Schema validation for all tool parameters
- **Network controls**: IP blocking, domain allowlists, timeout enforcement
- **Filesystem protection**: Path normalization, traversal prevention
- **API key protection**: Masking in logs, no storage in conversation history
- **Rate limiting**: Per-provider and per-tool request throttling

### Testing Strategy (Enhanced per All Reviews)
- **Unit tests**: Core logic, tool routing, provider interfaces
- **Integration tests**: CLI flows, provider fallback, MCP connectors
- **Security tests**: Path traversal, injection attempts, key leakage
- **Performance tests**: Latency benchmarks, memory usage, stress testing
- **Contract tests**: Provider compatibility, MCP compliance

## Appendix — Combined Review References

### Claude Review Key Findings
- **File**: `claude_review.md`
- **Main Issues**: 1,548-line LLM class, inconsistent error handling, performance bottlenecks
- **Key Metrics**: ~98 Python files, 345K lines of code, extensive provider support
- **Rating**: B+ (Very Good) - production-ready with optimization needed

### Codex Review Key Findings
- **File**: `codex_updates.md`
- **Main Issues**: Tool routing unreliability, thinking pipeline disconnected, documentation outdated
- **Key Problems**: Wrong method names, UI callbacks not wired, README references non-existent packages
- **Focus**: Restore manifesto promises of reliable tool usage and visible thinking

### Gemini Review Key Findings
- **File**: `gemini_review.md`
- **Main Issues**: Code duplication (sync/async), dual ToolManager concepts, monolithic CLI
- **Key Recommendations**: Unified async-first architecture, consolidated tool management
- **Focus**: Structural refactoring for maintainability and consistency

### Technical References
- **Tool method mismatch**:
  - Provider: `yamllm/providers/openai.py::process_streaming_tool_calls`
  - Core: `yamllm/core/llm.py::_handle_streaming_with_tools` checks `process_tool_calls_stream` (wrong name)
- **Anthropic tool parsing**: Tool calls in message content parts (`tool_use`) not top-level `tool_calls`
- **MCP async issues**: `yamllm/mcp/client.py::discover_all_tools` calls async without await
- **Architecture duplication**: `yamllm/core/llm.py` (1,548 lines) vs `yamllm/core/async_llm.py` (subset)
- **Tool management split**: `yamllm/tools/manager.py` vs `yamllm/core/tool_management.py`

---

## Full Progress Checklist

Quick Wins — Day 1–3
- [x] Fix streaming+tools method name (core → process_streaming_tool_calls)
- [x] Fix Anthropic tool parsing (tool_use parts)
- [x] Fix MCP async/await (await client calls; async connectors; add MCPClient.close)
- [x] Add API key masking (baseline masking in core logs)
- [x] Fix path traversal (expanduser + realpath; null-byte checks; internal IP/.local blocking)

Week 1 Improvements
- [x] Add basic Rich integration (Terminal UI components; streaming renderer; theming)
- [x] Create `yamllm run` command (quick-start chat)
- [x] Fix vector store dimension handling (enforced dims; migrate-index helper)
- [x] Add 3 example configs (.config_examples/openai_minimal.yaml, anthropic_minimal.yaml, google_minimal.yaml)
- [x] Clean up duplicate code (remove llm_old/llm_legacy)

P0 — Critical bugs and correctness
- [x] Streaming+tools pipeline mismatch fixed
- [x] Anthropic tool-call extraction corrected
- [x] Async provider init via `ProviderFactory.create_async_provider`
- [x] MCP client/connector async misuse (broader refactor and tests)
- [x] Retry decorator surfaces aligned to custom exceptions
- [x] Anthropic embeddings: remove pseudo-random fallback and use ProviderError/NotImplemented

P1 — Security and safety
- [x] Filesystem path handling hardened (expanduser + realpath)
- [x] Network blocking expanded (unspecified/multicast/reserved/.local)
- [ ] Add granular tests for `SecurityManager.check_network_permission`
- [ ] Centralize network tool timeouts/retries across all tools

P1 — API/UX consistency
- [x] Rename/merge dual ToolManager concepts (core vs tools.manager)
- [x] Provider base signature cleanup (remove `model` from `__init__`)
- [x] Standardize tool result shapes (prefer dict; stringify in providers)
- [x] Event-loop shutdown robustness on 3.12+ (`asyncio.run`/anyio)
- [x] parse_yaml_config: remove prints and re-raise errors

P2 — Developer experience and testing
- [x] Adopt ProviderFactory for async in LLM
- [x] Add tests for streaming+tools path (stub provider)
- [x] Harmonize exceptions and retry surfaces across providers
- [x] Trim unused heavy deps to optional extras

P2 — Performance and resilience
- [x] Cache embeddings per request; reduce duplicate calls
- [x] Improve tool selection heuristics beyond string matching
- [x] Interactive vector-store migration UX

DX/UX polish
- [x] Add `yamllm diagnose` command (env/config/tools/vector dims)
- [ ] Emit warning when tool gating disables all tools for a prompt
- [ ] Improve fallback error surfaces (actionable summaries)
- [x] Create `yamllm init` wizard
- [ ] `.env` template generation

MCP Implementation (first-class citizen)
- [x] `/mcp list` CLI to enumerate tools by connector
- [x] Fully fix async/await across client/connector (broader audit/tests)
- [x] Implement WS/HTTP/stdio transports and reconnection logic
- [ ] Add MCP host mode to expose local tools
- [x] Namespacing, auth, health checks

CLI & Quick Start
- [x] `yamllm run` alias for streamlined chat
- [x] `yamllm diagnose` command
- [x] `yamllm init` wizard
- [ ] Minimal 10–20 LOC Python examples in `examples/`
- [ ] `.env` template and docs

Terminal UI & Themes
- [x] Rich/Textual integration and theme system (YAML themes; theme list/preview/set)
- [x] Streaming renderer with sentence chunking; chat bubbles, timestamps
- [x] ASCII banners and customizable chrome
- [ ] Copy shortcuts and additional UX niceties

Async Architecture Overhaul
- [x] Async-first across providers (sync wrappers retained)
- [x] HTTP/2 connection pooling with reuse
- [x] Cancellation on keypress (propagate through model/tool calls)
- [ ] Backpressure handling in renderer (baseline via sentence chunking/live refresh)

Thinking Display System
- [x] `off|on|auto` modes with complexity detection
- [x] Thinking panel streaming in <120ms; redact internal logs
- [x] Summarized plans for complex tasks
