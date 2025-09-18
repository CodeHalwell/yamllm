# Project Review & Recommendations

## Executive Summary
YAMLLM has grown into a feature-rich orchestration layer, but several critical paths break the manifesto promises: tool selection is unreliable, examples bypass the streaming UX, docs lag behind the current code, and the test suite does not cover new behaviours. The notes below prioritise fixes that restore predictable tool usage, surface thinking/tool events, and tighten documentation and developer ergonomics.

## Architecture & Core Library
- **Tool routing is inconsistent** (`yamllm/core/llm.py:616-709`)
  - The current heuristic leaves calculator enabled even when the user explicitly requests web scraping, so OpenAI repeatedly chooses the wrong tool.
  - When `_determine_tool_choice` returns a specific tool, the provider silently ignores it if the tool list still contains other entries.
  - Recommendation: when `_extract_explicit_tool` resolves a tool name, return *only* that tool definition and set `tool_choice="required"`. Add unit tests around `_filter_tools_for_prompt` and `_determine_tool_choice` to confirm behaviour.
- **Thinking pipeline is partially disconnected** (`yamllm/core/llm.py:753-820`)
  - The code emits thinking/tool events, but the example CLI never registers an `event_callback`, so users see neither the plan nor tool activity.
  - Recommendation: in higher-level entry points (CLI, API adapters), wire up both stream and event callbacks so the manifesto’s “thinking in the open” promise is visible.
- **Parser back-compat still fragile** (`yamllm/core/parser.py:82-99`)
  - Accepting both `tools` and `tool_list` avoids runtime errors, but downstream code consumes `tools` only, so a config that mixes keys can diverge silently.
  - Recommendation: normalise in the parser (merge, dedupe, then expose one list) and remove the extra attribute from the Pydantic model to prevent future drift.
- **Concurrency & streaming**
  - Tool execution runs on a `ThreadPoolExecutor`, but there is no instrumentation for queue pressure or latency.
  - Recommendation: log start/end timings within `ToolOrchestrator.execute_tool` and expose metrics hooks for future observability.

## Tools & Providers
- **Web scraper UX** (`yamllm/tools/utility_tools.py:516`)
  - The tool returns raw text; the example prints 1,000 characters, overwhelming the terminal.
  - Recommendation: pair scraper results with summarisation (either call `web_headlines` first or prompt the LLM with the scrape result) and add pagination/truncation utilities.
- **Provider alignment**
  - OpenAI-specific behaviour (ignoring forced `tool_choice`) suggests we need provider-specific wrappers or fallback strategies.
  - Recommendation: add wrapper logic in `yamllm/providers/openai.py` that converts `{type: function}` specs into the expected `{"function": {"name": ...}}` or clamps the tool list before calling the API.

## CLI & Examples
- **Example bypasses tool orchestration** (`examples/full_cli_chat.py`)
  - Previous iterations short-circuited tool calls; after removal, the CLI still lacks visibility into tool planning and thinking.
  - Recommendation: set `llm.set_event_callback(ui.print_event)` and show graceful messaging when tool output is empty. Consider adding subcommands (`/scrape`) that orchestrate a scrape + summarise flow deterministically.
- **Logging for developers**
  - With `logging.console` disabled, INFO diagnostics never appear, making debugging difficult.
  - Recommendation: enable console logging in the example config or document how to toggle it via CLI flags.

## Documentation & Guides
- **README is outdated**
  - References `pip install yamllm-core`, a package that does not exist, and only lists four tools.
  - Recommendation: refresh the README with the current install path (`uv pip install -e .`), list of built-in tools, and updated quick-start reflecting the CLI experience.
- **Manifesto alignment**
  - Manifesto promises streamed thinking and reliable tool usage; current demos do not meet these guarantees.
  - Recommendation: add a “Status vs Manifesto” section to docs or improvement_plan.md noting work still outstanding (tool reliability, MCP host/client, etc.).

## Testing & CI
- **Coverage gaps**
  - No tests cover `_filter_tools_for_prompt`, `_determine_tool_choice`, or explicit tool commands. MCP connectors, vector store migrations, and CLI flows also lack coverage.
  - Recommendation: add unit tests for tool gating, integration tests for CLI flows (mock provider responses), and regression tests for MCP registration.
- **Automation**
  - CI (not shown) should enforce lint/test/mypy; confirm this pipeline exists and add GitHub Actions if missing.

## Prioritised Action Plan
1. **P0 (immediate) – Tool routing reliability**
   - Restrict tool list when instructions are explicit and confirm provider compliance.
   - Expose thinking/tool events in CLI, ensuring users see plans and results.
2. **P1 – Documentation & UX**
   - Update README and example instructions, add CLI documentation for debugging (logging, tool output handling).
   - Improve scraper UX: summarise results, avoid dumping raw HTML.
3. **P2 – Testing & Observability**
   - Add unit/integration tests covering tool gating and CLI behaviours.
   - Introduce structured logging around tool invocation and thinking pipeline.
4. **P3 – Future improvements**
   - Simplify dual tool managers (`yamllm/core/tool_management.py` vs `yamllm/tools/manager.py`).
   - Evaluate provider-specific adapters to reconcile tool-choice semantics across vendors.

Addressing these items will bring the demo experience back in line with the manifesto and give contributors confidence that tools and streaming behave deterministically.
