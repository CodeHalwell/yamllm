# Architecture Overview

This document explains the separation of concerns in YAMLLM.

- LLM (yamllm.core.llm):
  - Loads config, owns request settings, memory, and tool orchestration
  - Selects a provider via `yamllm.core.providers.ProviderFactory`
  - Exposes streaming via `set_stream_callback` so callers control rendering
  - Delegates tool definitions and execution to `ToolManager`

- Providers (yamllm/core/providers/*):
  - Implement a single `BaseProvider` interface (completion, streaming, embeddings, tool formatting)
  - Contain no UI or printing; return SDK-native objects or iterators

- Tooling (yamllm/tools/*):
  - `Tool` base class provides schema exposure via `get_signature()`
  - `ToolManager` registers tools, exposes provider-friendly definitions, and executes with timeouts

- UI (yamllm/ui/rich_renderer.py):
  - Optional helper that attaches a streaming renderer to an `LLM` instance
  - Keeps all rendering out of core logic

