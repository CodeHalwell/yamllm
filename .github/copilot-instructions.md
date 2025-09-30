# Copilot Instructions for yamllm

This guide is for AI coding agents working in the `yamllm` repository. It summarizes essential architecture, workflows, and conventions to maximize productivity and code quality.

## Architecture Overview
- **Core library:** `yamllm/`
  - `core/`: LLM orchestration, config parsing, provider abstraction (`providers/`), main `LLM` class
  - `tools/`: Built-in tools and `ToolManager` (function calling, tool packs)
  - `ui/`: Optional Rich-based CLI renderers
  - `memory/`: Conversation and vector store utilities (SQLite, FAISS)
- **Providers:** Unified via `ProviderFactory` (`yamllm/providers/`). All providers implement `BaseProvider` (completion, streaming, embedding, tool formatting). Supported: OpenAI, Anthropic, Google, Mistral, DeepSeek, Azure, OpenRouter.
- **Tools:** Registered in `tools/manager.py`, support JSON-schema signatures, safe I/O, and MCP connectors. Tool execution is thread-safe with timeouts.
- **Memory:** Managed in `memory/` (SQLite for chat, FAISS for vectors). Configurable via YAML.
- **CLI:** Entry points in `yamllm/cli.py`. Example scripts in `examples/`.

## Key Workflows
- **Install (editable):** `uv pip install -e .`
- **Run example:** `uv run examples/openai_example.py`
- **Test:** `uv run -m pytest -q` (unit tests in `tests/`)
- **Lint:** `uv run ruff check .`
- **Format:** `uv run black --check .`
- **Type check:** `uv run mypy yamllm`

## Configuration
- YAML config files (see `.config_examples/`).
- Never commit API keys; use `.env` and environment variables (e.g., `OPENAI_API_KEY`).
- Config sections: `provider`, `model_settings`, `request`, `context`, `output`, `logging`, `tools`, `safety`.
- Supports `${VAR_NAME}` env substitution.

## Coding & Testing Conventions
- Python 3.10+, PEP 8, 4-space indent.
- `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Keep modules focused; avoid cross-layer imports (UI → core only).
- Use type hints everywhere; validate config with Pydantic.
- Tests: `pytest`, files as `test_*.py`, mock external I/O.
- Tool schemas and provider behaviors must be validated with minimal tokens/fixtures.

## Integration & Extensibility
- Tools and providers are plugin-like; add new ones via `tools/` or `providers/`.
- MCP (Model Context Protocol) connectors in `yamllm/mcp/` for remote tool use.
- UI themes and CLI commands are extensible via `ui/` and `cli.py`.

## Examples
- See `examples/` for runnable demos (OpenAI, Google, Mistral, etc.).
- Example config: `.config_examples/`.

## Security & Safety
- Secrets must be masked in logs and never printed.
- Tool I/O is sandboxed and allowlisted; timeouts and confirmations enforced.
- Telemetry is off by default.

## References
- See `AGENTS.md` for contributor guidelines and architecture details.
- See `README.md` for user-facing quickstart and API usage.
- See `CLAUDE.md` for additional dev commands and provider interface notes.

---

If a pattern or workflow is unclear, check `AGENTS.md` and `README.md` first, or ask for clarification before proceeding.
