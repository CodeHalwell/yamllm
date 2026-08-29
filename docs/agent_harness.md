# Agent Harness (2026)

The agent subsystem was modernised around a single event-driven engine,
`yamllm.agent.harness.AgentHarness`, with a Textual TUI on top. The original
`Agent` class remains as a thin backwards-compatible wrapper.

## Architecture

```
                 ┌────────────────────────────────────────┐
                 │              AgentHarness              │
   goal ───────► │  plan → reason → (approve?) → act →    │ ───► AgentState
                 │  observe → checkpoint → repeat         │
                 └───────┬──────────────────┬─────────────┘
                         │ AgentEvent       │ SteeringDecision
                         ▼                  ▲
              ┌──────────────────┐  ┌───────┴────────┐
              │  subscribers     │  │ decision       │
              │  (TUI, console,  │  │ provider       │
              │   JSONL sinks)   │  │ (modal / rich) │
              └──────────────────┘  └────────────────┘
```

The engine never touches a console. It emits typed `AgentEvent` objects
(`yamllm.agent.events`) for every phase — `run_started`, `plan_created`,
`thought`, `action_started`, `action_finished`, `observation`,
`approval_requested`, `checkpoint_saved`, `budget_exceeded`, `run_finished`,
and more. Any number of subscribers can render or persist the stream.

## Quick start

```python
from yamllm import LLM
from yamllm.agent import AgentHarness, ApprovalPolicy

llm = LLM("config.yaml")

harness = AgentHarness(
    llm,
    max_iterations=15,
    max_wall_time=600,  # seconds
    checkpoint_dir=".agent-checkpoints",
    approval_policy=ApprovalPolicy.NEVER,
)
harness.add_listener(lambda event: print(event.to_json()))

state = harness.run("Add retry logic to the HTTP client")
print(state.success, state.error)
```

### Budgets

- `max_iterations` — ReAct loop budget (as before).
- `max_wall_time` — wall-clock budget in seconds.
- `max_consecutive_failures` — abort after N failed actions in a row.

When a budget trips, the harness emits `budget_exceeded` and finishes the
run with `success=False` and a descriptive `error`.

### Cancellation

`harness.request_stop()` is thread-safe and stops the loop at the next
iteration boundary — this is what the TUI's `x` binding calls.

### Human-in-the-loop approvals

```python
from yamllm.agent import ApprovalPolicy

harness = AgentHarness(
    llm,
    approval_policy=ApprovalPolicy.ON_WATCHPOINT,
    decision_provider=my_blocking_prompt,  # SteeringPoint -> SteeringDecision
)
harness.add_watchpoint(lambda sp: "delete" in sp.thought.lower())
```

- `NEVER` — fully autonomous (default).
- `ALWAYS` — gate every action.
- `ON_WATCHPOINT` — gate only when a watchpoint matches.

Decisions reuse the existing `SteeringAction` vocabulary: approve, reject,
skip, modify (with feedback), auto-approve the rest, or stop the run.

### Checkpoint / resume

With `checkpoint_dir` set, the harness serialises the full `AgentState` to
JSON after every iteration. Resume with:

```python
from yamllm.agent import AgentHarness, load_checkpoint

state = load_checkpoint(".agent-checkpoints/ab12cd34.json")
harness.run(state.goal, initial_state=state)
```

## Textual TUI

`yamllm.ui.agent_tui.AgentTUI` is a full-screen Textual app: live task
board on the left, streaming transcript (thoughts, tool calls, results,
observations) on the right, and a modal approval dialog when the harness
pauses for a decision.

Key bindings: `a` approve · `r` reject · `s` skip · `m` modify (type
guidance, Enter) · `o` auto-approve the rest · `x` stop the run ·
`q` quit.

The harness runs on a worker thread; events cross to the UI thread via
`call_from_thread`, and approval requests block the worker on a queue until
the operator decides.

## CLI

```bash
# Autonomous run with plain console output
yamllm agent run "Fix the failing tests" --config config.yaml

# Live dashboard (no gating)
yamllm agent run "Fix the failing tests" --config config.yaml --tui

# Review every action in the TUI before it runs
yamllm agent run "Refactor parser.py" --config config.yaml --interactive

# Legacy prompt-based approvals without the TUI
yamllm agent run "Refactor parser.py" --config config.yaml --interactive --plain

# Budgets, checkpoints, recording, and an event log
yamllm agent run "Ship the feature" --config config.yaml \
    --max-iterations 20 --max-wall-time 900 \
    --checkpoint-dir .agent-checkpoints --record \
    --events-jsonl run-events.jsonl

# Resume an interrupted run
yamllm agent run "unused" --config config.yaml --resume .agent-checkpoints/ab12cd34.json
```

## Backwards compatibility

- `Agent` / `SimpleAgent` keep their public API; `Agent.execute` now
  delegates to `AgentHarness` internally.
- `InteractiveAgent` / `InteractiveSteering` still work; the prompt-based
  flow is available via `--interactive --plain`, and
  `InteractiveSteering.request_decision` plugs directly into the harness as
  a `decision_provider`.
- Session recordings, workflows, and the `AgentState`/`Task` models are
  unchanged (the models gained `to_dict`/`from_dict` for checkpoints).
