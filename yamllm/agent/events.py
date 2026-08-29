"""Typed events emitted by the agent harness.

Every phase of a run produces an :class:`AgentEvent`. UIs (the Textual app,
the plain CLI renderer, log sinks) subscribe to the harness and render the
stream however they like — the engine never talks to a console directly.
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class EventKind(str, Enum):
    """Kinds of events emitted during an agent run."""

    RUN_STARTED = "run_started"
    PLAN_CREATED = "plan_created"
    ITERATION_STARTED = "iteration_started"
    THOUGHT = "thought"
    ACTION_STARTED = "action_started"
    ACTION_FINISHED = "action_finished"
    OBSERVATION = "observation"
    TASK_UPDATED = "task_updated"
    APPROVAL_REQUESTED = "approval_requested"
    DECISION = "decision"
    CHECKPOINT_SAVED = "checkpoint_saved"
    BUDGET_EXCEEDED = "budget_exceeded"
    RUN_FINISHED = "run_finished"
    ERROR = "error"


@dataclass
class AgentEvent:
    """A single event in the run's event stream.

    Attributes:
        kind: What happened.
        payload: Kind-specific data (JSON-serialisable where possible).
        timestamp: Unix timestamp of when the event was emitted.
    """

    kind: EventKind
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary (e.g. for JSONL sinks)."""
        return {
            "kind": self.kind.value,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }

    def to_json(self) -> str:
        """Serialise to a single JSON line, tolerating non-serialisable values."""
        return json.dumps(self.to_dict(), default=str)
