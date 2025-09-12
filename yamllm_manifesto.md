# The **yamllm** Manifesto

*A human terminal for serious work and playful intelligence.*

## Purpose

yamllm exists to make working with large language models effortless, fast, and enjoyable—straight from the terminal. No boilerplate, no UI wrangling, no yak-shaving. Install it, write a handful of lines, and you’re in a rich, streaming conversation where tools, memory, and reasoning are handled for you.

## Philosophy

* **Practical first.** Defaults that work; power when you want it.
* **Beauty matters.** Terminal output should be clean, legible, and fun without becoming twee.
* **Agentic by design.** Conversations become workflows; tools are part of the language.
* **Interoperable.** We embrace open protocols (MCP) and a pluggable tool system.
* **Asynchronous everywhere.** Low latency isn’t a nicety; it’s the point.
* **Respect the user.** Privacy by default, clear permissions, easy exits.

---

## What “good” looks like

### 1) Effortless onboarding

After installation, a user should reach a full, themed, streaming chat with tools and memory in **10–20 lines of code** or a single CLI command. yamllm hides Rich/Textual plumbing, provider quirks, and logging setup. YAML captures configuration; the library supplies sensible defaults.

**Promise:** “Install → configure → chat” in minutes, not hours.

### 2) A terminal worth talking to

The interface is vibrant by default and widely customisable:

* **Themes** define colours, ASCII banners, emojis, chat bubbles, borders, and chrome.
* **Catalogue of styles** from minimalist to maximalist; switchable at runtime.
* **Streaming by default** with tidy sentence chunking to avoid flicker.
* **Useful affordances**: timestamps, copy shortcuts, collapsible blocks, `/save`, `/clear`, `/theme`.

**Promise:** Beautiful output with zero user code. The fun bits are a theme, not a chore.

### 3) Thinking in the open (without faff)

Reasoning is visible **when it helps** and invisible when it doesn’t.

* **Modes:** `off`, `on`, `auto` (default).
* **Adaptive depth:** greetings get near-instant replies; complex prompts show a brief, streamed “thinking” panel.
* **Redaction by default:** logs store short plan summaries, not raw internal reasoning.

**Promise:** Transparency for problems, speed for pleasantries.

### 4) Agentic tools, sensibly done

yamllm ships with a **developer-centric tool library** (file and code search, shell with guardrails, Python/runner, HTTP, web search & scrape, git, SQL, archive, process, table/CSV, image info, notebook snippets, and more).

* **Smart routing** chooses tools by schema fit and past success.
* **Progress streaming** shows spinners, steps, and partial results.
* **Permissions & safety**: allowlists, timeouts, confirmations for destructive ops.

**Promise:** Tools feel native to conversation. You ask; the right thing happens—fast, visible, reversible.

### 5) MCP as a first-class citizen

We adopt the **Model Context Protocol** to connect with the wider ecosystem.

* **Client role:** connect to external MCP servers (stdio/websocket), list their tools, and invoke them as if local.
* **Host role:** expose yamllm’s tools as an MCP server for other agents.
* **Namespacing, auth, health, and reconnection** are all built-in.

**Promise:** Plug into other agents and services without bespoke glue.

### 6) Low-latency, fully async architecture

Everything that can stream, does stream. Planning, tool calls, and model tokens overlap. HTTP/2 keep-alive, pooled connections, cancellation on keypress, and backpressure when the renderer is busy.

**Targets:**

* First token < **350 ms** (tools off) / < **600 ms** (tools on).
* Tool first byte < **0.5–0.9 s**.
* Thinking panel appears < **120 ms** after input.

**Promise:** Crisp interaction as standard, not as a future optimisation.

### 7) Memory and logging that help (and don’t haunt)

* **Conversation store** in SQLite; exportable transcripts.
* **Vector memory** opt-in for RAG; local by default.
* **Telemetry off by default;** if enabled, only anonymous performance counters.
* **Secrets** come from env/OS keychain, never printed, always masked.

**Promise:** Useful recall without surveillance vibes.

### 8) Reliability when things go sideways

* **Exponential backoff** with jitter for flaky networks.
* **Clear, compact errors** in the UI; verbose details behind a disclosure.
* **Session snapshots** so a crash mid-stream doesn’t cost the chat.

**Promise:** Fail politely, recover quickly.

### 9) Extensible by normal humans

* **Tools as plugins** via entry points or drop-in modules.
* **Schema-first contract** (JSON Schema for inputs/outputs) with streaming events.
* **Themes** are simple YAML; contribute one in minutes.
* **Providers** implement a tiny streaming interface; add your favourite model cleanly.

**Promise:** Contributions are measured in lines, not weekends.

### 10) Quality gates that matter

* Golden **UI snapshots** for themes.
* **Latency harness** with fake providers for deterministic tests.
* **Tool conformance**: schema validation, timeout & cancellation tests.
* **MCP contract tests** against reference servers.
* A **no-regression prompt pack** covering greetings, coding, browsing, multi-tool, and MCP flows.

**Promise:** We don’t break your muscle memory on update.

---

## Non-goals (so we stay honest)

* We do **not** become a kitchen-sink IDE. yamllm complements editors; it doesn’t replace them.
* We avoid lock-in. Protocols over proprietary magic.
* We resist sprawling configuration: defaults first, clarity over cleverness.

---

## Roadmap slices (deliver value early)

* **v0.1 (MVP):** CLI `yamllm run`; OpenAI streaming; two themes; tools: file\_search, http, web\_search, web\_scrape, python; thinking `off|on|auto`; basic MCP client (stdio).
* **v0.2:** git, code\_search, fs, code\_run, SQL; MCP host mode & multi-client; vector memory; guarded shell.
* **v0.3:** perf polish (HTTP/2 pooling, smarter routing); accessibility theme; plugin registry; adapters for Gemini, Mistral, DeepSeek.

---

## Acceptance tests (the user-visible contract)

* Typing “hello” yields a streamed reply in **< 400 ms** with **no visible** thinking.
* A “refactor this…” prompt shows thinking for **≤ 3.5 s**, then streams highlighted code.
* A browse task (`web_search` → `web_scrape`) shows progress and a compact summary, with raw details collapsible.
* `/mcp list` shows at least one server; invoking `mcp:*` streams results like local tools.
* Switching theme at runtime updates bubbles, colours, and banner without restart.

---

## Ethos

We value craft, speed, and a touch of whimsy. The terminal is a stage; the model is a performer; tools are the orchestra. yamllm’s job is to conduct without getting in the way—and to make sure the audience (you) leaves with useful work done.

---

