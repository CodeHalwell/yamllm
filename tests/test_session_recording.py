"""Tests for session recording and replay."""

import pytest
import tempfile
import os
from pathlib import Path
import yaml
import json
from yamllm.agent.recording import (
    SessionRecorder,
    SessionPlayer,
    RecordingManager
)
from yamllm.agent.models import AgentState


def test_recorder_initialization():
    """Test session recorder initialization."""
    state = AgentState(goal="Test goal")
    recorder = SessionRecorder(state)

    assert recorder.recording["session_id"] is not None
    assert recorder.recording["goal"] == "Test goal"
    assert len(recorder.recording["iterations"]) == 0


def test_record_iteration():
    """Test recording an iteration."""
    state = AgentState(goal="Test goal")
    recorder = SessionRecorder(state)

    recorder.record_iteration(
        iteration=1,
        thought="I should do X",
        action={"task": "do_something"},
        observation={"result": "success"}
    )

    assert len(recorder.recording["iterations"]) == 1

    iter_data = recorder.recording["iterations"][0]
    assert iter_data["iteration"] == 1
    assert iter_data["thought"] == "I should do X"
    assert iter_data["action"] == {"task": "do_something"}
    assert iter_data["observation"] == {"result": "success"}
    assert "timestamp" in iter_data


def test_record_multiple_iterations():
    """Test recording multiple iterations."""
    state = AgentState(goal="Test goal")
    recorder = SessionRecorder(state)

    for i in range(1, 4):
        recorder.record_iteration(
            iteration=i,
            thought=f"Thought {i}",
            action={"task": f"action_{i}"},
            observation={"progress": i/3}
        )

    assert len(recorder.recording["iterations"]) == 3


def test_save_yaml(tmp_path):
    """Test saving recording to YAML."""
    state = AgentState(goal="Test goal")
    recorder = SessionRecorder(state)

    recorder.record_iteration(1, "thought", {"action": 1}, {"obs": 1})

    filepath = tmp_path / "recording.yaml"
    recorder.save(str(filepath), format="yaml")

    assert filepath.exists()

    # Load and verify
    with open(filepath) as f:
        data = yaml.safe_load(f)

    assert data["session_id"] == recorder.recording["session_id"]
    assert data["goal"] == "Test goal"
    assert len(data["iterations"]) == 1


def test_save_json(tmp_path):
    """Test saving recording to JSON."""
    state = AgentState(goal="Test goal")
    recorder = SessionRecorder(state)

    recorder.record_iteration(1, "thought", {"action": 1}, {"obs": 1})

    filepath = tmp_path / "recording.json"
    recorder.save(str(filepath), format="json")

    assert filepath.exists()

    # Load and verify
    with open(filepath) as f:
        data = json.load(f)

    assert data["session_id"] == recorder.recording["session_id"]
    assert data["goal"] == "Test goal"
    assert len(data["iterations"]) == 1


def test_load_recording(tmp_path):
    """Test loading a recording."""
    # Create and save a recording
    state = AgentState(goal="Test goal")
    recorder = SessionRecorder(state)
    recorder.record_iteration(1, "thought", {"action": 1}, {"obs": 1})

    filepath = tmp_path / "recording.yaml"
    recorder.save(str(filepath))

    # Load with player
    player = SessionPlayer(str(filepath))

    assert player.session_id == recorder.recording["session_id"]
    assert player.goal == "Test goal"
    assert len(player.iterations) == 1


def test_player_get_iteration():
    """Test getting specific iterations."""
    state = AgentState(goal="Test")
    recorder = SessionRecorder(state)
    recorder.record_iteration(1, "thought1", {"a": 1}, {"b": 1})
    recorder.record_iteration(2, "thought2", {"a": 2}, {"b": 2})

    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(recorder.recording, f)
        temp_file = f.name

    try:
        player = SessionPlayer(temp_file)

        iter1 = player.get_iteration(1)
        assert iter1["thought"] == "thought1"

        iter2 = player.get_iteration(2)
        assert iter2["thought"] == "thought2"
    finally:
        os.unlink(temp_file)


def test_player_get_summary():
    """Test getting recording summary."""
    state = AgentState(goal="Test goal", metadata={"key": "value"})
    recorder = SessionRecorder(state)
    recorder.record_iteration(1, "thought1", {"a": 1}, {"b": 1})
    recorder.record_iteration(2, "thought2", {"a": 2}, {"b": 2})

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(recorder.recording, f)
        temp_file = f.name

    try:
        player = SessionPlayer(temp_file)
        summary = player.get_summary()

        assert summary["session_id"] == recorder.recording["session_id"]
        assert summary["goal"] == "Test goal"
        assert summary["total_iterations"] == 2
        assert "context" in summary
    finally:
        os.unlink(temp_file)


def test_compare_recordings(tmp_path):
    """Test comparing two recordings."""
    # Create two recordings
    state1 = AgentState(goal="Goal 1")
    recorder1 = SessionRecorder(state1)
    recorder1.record_iteration(1, "thought1", {}, {})
    file1 = tmp_path / "rec1.yaml"
    recorder1.save(str(file1))

    state2 = AgentState(goal="Goal 2")
    recorder2 = SessionRecorder(state2)
    recorder2.record_iteration(1, "thought1", {}, {})
    recorder2.record_iteration(2, "thought2", {}, {})
    file2 = tmp_path / "rec2.yaml"
    recorder2.save(str(file2))

    # Compare
    player1 = SessionPlayer(str(file1))
    comparison = player1.compare_with(SessionPlayer(str(file2)))

    assert comparison["session1"]["session_id"] == recorder1.recording["session_id"]
    assert comparison["session2"]["session_id"] == recorder2.recording["session_id"]
    assert comparison["session1"]["iterations"] == 1
    assert comparison["session2"]["iterations"] == 2


def test_recording_manager(tmp_path):
    """Test recording manager."""
    manager = RecordingManager(str(tmp_path))

    # Create some recordings
    state1 = AgentState(goal="Goal 1")
    recorder1 = SessionRecorder(state1)
    recorder1.save(str(tmp_path / "test1.yaml"))

    state2 = AgentState(goal="Goal 2")
    recorder2 = SessionRecorder(state2)
    recorder2.save(str(tmp_path / "test2.yaml"))

    # List recordings
    recordings = manager.list_recordings()

    assert len(recordings) == 2


def test_recording_manager_load():
    """Test loading recording via manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create recording
        state = AgentState(goal="Test")
        recorder = SessionRecorder(state)
        filepath = Path(tmpdir) / "test.yaml"
        recorder.save(str(filepath))

        # Load via manager
        manager = RecordingManager(tmpdir)
        session_id = recorder.recording["session_id"]
        player = manager.load_recording("test")

        assert player.session_id == session_id


def test_recording_manager_delete():
    """Test deleting recording via manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create recording
        state = AgentState(goal="Test")
        recorder = SessionRecorder(state)
        filepath = Path(tmpdir) / "test.yaml"
        recorder.save(str(filepath))

        # Delete via manager
        manager = RecordingManager(tmpdir)
        manager.delete_recording("test")

        # Should be gone
        recordings = manager.list_recordings()
        assert len(recordings) == 0


def test_recording_with_context():
    """Test recording with context."""
    context = {
        "repo": "/path/to/repo",
        "files": ["file1.py", "file2.py"]
    }

    state = AgentState(goal="Test", metadata=context)
    recorder = SessionRecorder(state)

    assert recorder.recording["metadata"] == context


def test_recording_timestamps():
    """Test that timestamps are recorded."""
    state = AgentState(goal="Test")
    recorder = SessionRecorder(state)

    recorder.record_iteration(1, "thought", {"action": 1}, {"obs": 1})

    assert "start_time" in recorder.recording
    iter_data = recorder.recording["iterations"][0]
    assert "timestamp" in iter_data


def test_invalid_recording_file():
    """Test handling invalid recording file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: content:")
        temp_file = f.name

    try:
        with pytest.raises(Exception):
            SessionPlayer(temp_file)
    finally:
        os.unlink(temp_file)


def test_nonexistent_recording_file():
    """Test handling nonexistent recording file."""
    with pytest.raises(FileNotFoundError):
        SessionPlayer("/nonexistent/file.yaml")


def test_recording_auto_format_detection(tmp_path):
    """Test auto-detection of file format."""
    state = AgentState(goal="Test")
    recorder = SessionRecorder(state)

    # YAML file
    yaml_file = tmp_path / "recording.yaml"
    recorder.save(str(yaml_file))
    assert yaml_file.exists()

    # JSON file
    json_file = tmp_path / "recording.json"
    recorder.save(str(json_file))
    assert json_file.exists()

    # Both should be loadable
    player_yaml = SessionPlayer(str(yaml_file))
    player_json = SessionPlayer(str(json_file))

    assert player_yaml.goal == player_json.goal


def test_recording_manager_search():
    """Test searching recordings by goal."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create recordings with different goals
        state1 = AgentState(goal="Fix bug in auth")
        recorder1 = SessionRecorder(state1)
        recorder1.save(str(Path(tmpdir) / "r1.yaml"))

        state2 = AgentState(goal="Add feature X")
        recorder2 = SessionRecorder(state2)
        recorder2.save(str(Path(tmpdir) / "r2.yaml"))

        state3 = AgentState(goal="Fix bug in database")
        recorder3 = SessionRecorder(state3)
        recorder3.save(str(Path(tmpdir) / "r3.yaml"))

        # Search
        manager = RecordingManager(tmpdir)
        results = manager.search_by_goal("bug")

        assert len(results) == 2
        assert all("bug" in r["goal"].lower() for r in results)
