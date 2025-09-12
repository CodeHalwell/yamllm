# Gemini Code Assistant Context

## Project Overview

**Vision:** `yamllm` is a Python library aspiring to be "a human terminal for serious work and playful intelligence." Its goal is to make interacting with Large Language Models (LLMs) from the terminal effortless, fast, and enjoyable.

**Philosophy:** The project prioritizes practicality, a beautiful and customizable terminal UI, agentic tool use, interoperability via the Model Context Protocol (MCP), and a low-latency, asynchronous-first architecture.

**Core Functionality:**
The library uses YAML-based configurations to provide a unified interface to multiple LLM providers (OpenAI, Gemini, Anthropic, etc.). Key features are being developed to align with the project's manifesto:

*   **Effortless Onboarding:** Aims for a "chat-ready" state in minutes with a single CLI command or ~10-20 lines of code.
*   **Rich Terminal Experience:** Built-in support for themes, streaming output, and a polished UI using Rich/Textual.
*   **Agentic, Developer-Centric Tools:** A growing library of tools like `shell`, `git`, `sql`, and `code_search` with smart routing and security guardrails.
*   **Async Everywhere:** A focus on performance with a target of <350ms for the first token, achieved through a fully asynchronous pipeline.
*   **First-Class MCP Support:** Integration with the Model Context Protocol to connect with and expose tools to other AI agents.
*   **Helpful Memory:** Local-first conversation history (SQLite) and optional vector memory for RAG.
*   **Extensibility:** Designed to be easily extended with new providers, tools, and UI themes.

## Building and Running

The project is a Python library and can be installed using `pip` or `uv`.

### Installation

```bash
# Using pip
pip install yamllm-core

# Using uv
uv add yamllm-core
```

### Running Tests

The project uses `pytest` for testing. To run the tests, you will need to install the development dependencies:

```bash
uv pip install -r requirements.txt
pytest
```

## Development Conventions

*   **Manifesto-Driven:** Development is guided by the `yamllm_manifesto.md`, which outlines the project's goals and quality standards. The `improvement_plan.md` details the roadmap to achieve this vision.
*   **Async-First:** New I/O-bound code should be asynchronous, with synchronous wrappers provided for compatibility.
*   **Modular Architecture:** The project is organized into modules for core logic, providers, tools, memory, and UI.
*   **Provider Factory:** A `ProviderFactory` is used to create instances of LLM providers.
*   **Type Hinting:** The codebase uses Python's type hinting to improve code clarity and maintainability.
*   **Testing:** The project uses `pytest` and aims for high coverage, including UI snapshot tests, latency benchmarks, and tool conformance tests.
