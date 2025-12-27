"""Session recording and replay for agent executions."""

import yaml
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from pathlib import Path

from .models import AgentState


class SessionRecorder:
    """Record agent sessions for replay and analysis."""

    def __init__(self, agent_state: AgentState):
        """
        Initialize recorder with agent state.

        Args:
            agent_state: AgentState to record
        """
        self.state = agent_state
        self.recording: Dict[str, Any] = {
            "version": "1.0",
            "session_id": self._generate_session_id(),
            "goal": agent_state.goal,
            "start_time": datetime.now().isoformat(),
            "iterations": [],
            "metadata": agent_state.metadata.copy()
        }

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        from uuid import uuid4
        return str(uuid4())[:8]

    def record_iteration(
        self,
        iteration: int,
        thought: str,
        action: Dict[str, Any],
        observation: Optional[str] = None
    ) -> None:
        """
        Record an iteration.

        Args:
            iteration: Iteration number
            thought: Agent's thought/reasoning
            action: Action taken (task execution result)
            observation: Observation from result
        """
        iteration_data = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "thought": thought,
            "action": action,
            "observation": observation,
            "state_snapshot": {
                "completed_tasks": len(self.state.get_completed_tasks()),
                "pending_tasks": len(self.state.get_pending_tasks()),
                "progress": self.state.get_progress()
            }
        }

        self.recording["iterations"].append(iteration_data)

    def finalize(self, success: bool, error: Optional[str] = None) -> None:
        """
        Finalize recording.

        Args:
            success: Whether session completed successfully
            error: Error message if failed
        """
        self.recording["end_time"] = datetime.now().isoformat()
        self.recording["success"] = success
        self.recording["error"] = error
        self.recording["final_state"] = {
            "completed": self.state.completed,
            "total_iterations": self.state.iteration,
            "tasks": [
                {
                    "id": t.id,
                    "description": t.description,
                    "status": t.status.value,
                    "result": t.result,
                    "error": t.error
                }
                for t in self.state.tasks
            ]
        }

    def save(self, filepath: str, format: str = "yaml") -> None:
        """
        Save recording to file.

        Args:
            filepath: Path to save to
            format: Format (yaml or json)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if format == "yaml":
            with open(filepath, 'w') as f:
                yaml.dump(self.recording, f, default_flow_style=False, sort_keys=False)
        elif format == "json":
            with open(filepath, 'w') as f:
                json.dump(self.recording, f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")

    def get_recording(self) -> Dict[str, Any]:
        """Get the recording data."""
        return self.recording


class SessionPlayer:
    """Replay recorded agent sessions."""

    def __init__(self, recording: Dict[str, Any]):
        """
        Initialize player with recording.

        Args:
            recording: Recording data
        """
        self.recording = recording
        self.current_iteration = 0

    @classmethod
    def load(cls, filepath: str) -> "SessionPlayer":
        """
        Load recording from file.

        Args:
            filepath: Path to recording file

        Returns:
            SessionPlayer instance
        """
        filepath = Path(filepath)

        if filepath.suffix in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
                recording = yaml.safe_load(f)
        elif filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                recording = json.load(f)
        else:
            raise ValueError(f"Unknown file format: {filepath.suffix}")

        return cls(recording)

    def replay(
        self,
        speed: float = 1.0,
        until_iteration: Optional[int] = None,
        callback: Optional[Callable[[Dict], None]] = None
    ) -> None:
        """
        Replay the session.

        Args:
            speed: Playback speed multiplier (2.0 = 2x speed)
            until_iteration: Stop at this iteration
            callback: Optional callback for each iteration
        """
        import time
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console()

        # Show session header
        console.print(Panel(
            f"[bold]Session: {self.recording['session_id']}[/bold]\n"
            f"Goal: {self.recording['goal']}\n"
            f"Start: {self.recording['start_time']}",
            title="📼 Replay",
            border_style="cyan"
        ))

        iterations = self.recording.get("iterations", [])

        for iter_data in iterations:
            iteration = iter_data["iteration"]

            if until_iteration and iteration > until_iteration:
                break

            self.current_iteration = iteration

            # Display iteration
            console.print(f"\n[bold cyan]Iteration {iteration}[/bold cyan]")

            # Show thought
            if iter_data.get("thought"):
                console.print(Panel(
                    f"[italic]{iter_data['thought']}[/italic]",
                    title="💭 Thought",
                    border_style="magenta"
                ))

            # Show action
            action = iter_data.get("action", {})
            if action:
                task_id = action.get("task_id", "unknown")
                success = action.get("success", False)
                status = "✓" if success else "✗"
                console.print(f"\n{status} Action: Task {task_id}")

                if action.get("tool_calls"):
                    console.print(f"  Tools used: {len(action['tool_calls'])}")

            # Show observation
            if iter_data.get("observation"):
                console.print(f"\n[dim]→ {iter_data['observation']}[/dim]")

            # Show progress
            snapshot = iter_data.get("state_snapshot", {})
            if snapshot:
                progress = snapshot.get("progress", 0)
                console.print(f"  Progress: {progress:.0f}%")

            # Callback
            if callback:
                callback(iter_data)

            # Delay (adjusted by speed)
            time.sleep(0.5 / speed)

        # Show final state
        if "final_state" in self.recording:
            final = self.recording["final_state"]
            success = self.recording.get("success", False)

            status_text = "[bold green]✓ Success[/bold green]" if success else "[bold red]✗ Failed[/bold red]"
            console.print(f"\n\n{status_text}")
            console.print(f"Total iterations: {final.get('total_iterations', 0)}")

            if self.recording.get("error"):
                console.print(f"[red]Error: {self.recording['error']}[/red]")

    def get_iteration(self, n: int) -> Optional[Dict]:
        """Get specific iteration data."""
        iterations = self.recording.get("iterations", [])
        for iter_data in iterations:
            if iter_data["iteration"] == n:
                return iter_data
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of the session."""
        iterations = self.recording.get("iterations", [])

        return {
            "session_id": self.recording["session_id"],
            "goal": self.recording["goal"],
            "start_time": self.recording["start_time"],
            "end_time": self.recording.get("end_time"),
            "success": self.recording.get("success"),
            "total_iterations": len(iterations),
            "error": self.recording.get("error")
        }

    def compare_with(self, other: "SessionPlayer") -> Dict[str, Any]:
        """
        Compare this session with another.

        Args:
            other: Another SessionPlayer

        Returns:
            Comparison data
        """
        my_summary = self.get_summary()
        other_summary = other.get_summary()

        return {
            "session_1": {
                "id": my_summary["session_id"],
                "iterations": my_summary["total_iterations"],
                "success": my_summary["success"]
            },
            "session_2": {
                "id": other_summary["session_id"],
                "iterations": other_summary["total_iterations"],
                "success": other_summary["success"]
            },
            "comparison": {
                "iteration_diff": my_summary["total_iterations"] - other_summary["total_iterations"],
                "both_succeeded": my_summary["success"] and other_summary["success"],
                "same_goal": my_summary["goal"] == other_summary["goal"]
            }
        }


class RecordingManager:
    """Manage multiple session recordings."""

    def __init__(self, recordings_dir: str = ".yamllm/recordings"):
        """
        Initialize manager.

        Args:
            recordings_dir: Directory to store recordings
        """
        self.recordings_dir = Path(recordings_dir)
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

    def save_recording(
        self,
        recording: Dict[str, Any],
        name: Optional[str] = None
    ) -> Path:
        """
        Save a recording.

        Args:
            recording: Recording data
            name: Optional name (uses session_id if not provided)

        Returns:
            Path to saved recording
        """
        if name is None:
            name = recording["session_id"]

        filepath = self.recordings_dir / f"{name}.yaml"

        with open(filepath, 'w') as f:
            yaml.dump(recording, f, default_flow_style=False, sort_keys=False)

        return filepath

    def list_recordings(self) -> List[Dict[str, Any]]:
        """List all recordings."""
        recordings = []

        for filepath in self.recordings_dir.glob("*.yaml"):
            try:
                with open(filepath, 'r') as f:
                    recording = yaml.safe_load(f)

                recordings.append({
                    "filename": filepath.name,
                    "session_id": recording.get("session_id"),
                    "goal": recording.get("goal"),
                    "start_time": recording.get("start_time"),
                    "success": recording.get("success")
                })
            except Exception:
                # Skip invalid recording files
                pass

        return sorted(recordings, key=lambda x: x.get("start_time", ""), reverse=True)

    def load_recording(self, name: str) -> SessionPlayer:
        """
        Load a recording.

        Args:
            name: Recording name (with or without .yaml extension)

        Returns:
            SessionPlayer instance
        """
        if not name.endswith('.yaml'):
            name = f"{name}.yaml"

        filepath = self.recordings_dir / name

        return SessionPlayer.load(str(filepath))

    def delete_recording(self, name: str) -> bool:
        """
        Delete a recording.

        Args:
            name: Recording name

        Returns:
            True if deleted, False if not found
        """
        if not name.endswith('.yaml'):
            name = f"{name}.yaml"

        filepath = self.recordings_dir / name

        if filepath.exists():
            filepath.unlink()
            return True

        return False
