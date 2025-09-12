import os
from pathlib import Path
import tempfile
import pytest

from yamllm.tools.security import SecurityManager, ToolExecutionError
from yamllm.tools.utility_tools import FileReadTool, WebScraper


def test_file_read_denies_outside_workspace(tmp_path: Path):
    # Create a file outside the workspace (simulate by absolute path under tmp)
    external = tmp_path / "secret.txt"
    external.write_text("top secret")

    sm = SecurityManager(allowed_paths=[os.getcwd()])
    tool = FileReadTool(security_manager=sm)

    out = tool.execute(str(external))
    assert "error" in out
    assert "Access denied" in out["error"] or "disabled" in out["error"].lower()


def test_network_denied_in_safe_mode():
    sm = SecurityManager(safe_mode=True)
    tool = WebScraper(security_manager=sm)
    out = tool.execute("https://example.com")
    assert "error" in out
    assert "Network" in out["error"] or "disabled" in out["error"].capitalize()


def test_path_traversal_prevention():
    """Test that path traversal attacks are prevented."""
    with tempfile.TemporaryDirectory() as tmpdir:
        allowed_dir = os.path.join(tmpdir, "allowed")
        os.makedirs(allowed_dir)
        
        # Create a file outside allowed directory
        secret_file = os.path.join(tmpdir, "secret.txt")
        with open(secret_file, "w") as f:
            f.write("secret content")
        
        sm = SecurityManager(allowed_paths=[allowed_dir])
        
        # Test various path traversal attempts
        traversal_attempts = [
            os.path.join(allowed_dir, "..", "secret.txt"),
            os.path.join(allowed_dir, "..", "..", "etc", "passwd"),
            allowed_dir + "/../secret.txt",
            os.path.join(allowed_dir, "subdir", "..", "..", "secret.txt"),
        ]
        
        for path in traversal_attempts:
            with pytest.raises(ToolExecutionError) as exc_info:
                sm.validate_file_access(path)
            assert "Access denied" in str(exc_info.value)


def test_symlink_prevention():
    """Test that symlink attacks are prevented."""
    with tempfile.TemporaryDirectory() as tmpdir:
        allowed_dir = os.path.join(tmpdir, "allowed")
        os.makedirs(allowed_dir)
        
        # Create a file outside allowed directory
        secret_file = os.path.join(tmpdir, "secret.txt")
        with open(secret_file, "w") as f:
            f.write("secret content")
        
        # Create symlink inside allowed directory pointing outside
        symlink_path = os.path.join(allowed_dir, "link_to_secret")
        try:
            os.symlink(secret_file, symlink_path)
        except OSError:
            # Skip test on systems that don't support symlinks
            pytest.skip("Symlinks not supported on this system")
        
        sm = SecurityManager(allowed_paths=[allowed_dir])
        
        # Symlink should be rejected as it points outside allowed directory
        with pytest.raises(ToolExecutionError) as exc_info:
            sm.validate_file_access(symlink_path)
        assert "Access denied" in str(exc_info.value)


def test_null_byte_prevention():
    """Test that null byte injection is prevented."""
    sm = SecurityManager(allowed_paths=[os.getcwd()])
    
    # Test null byte in path
    with pytest.raises(ToolExecutionError) as exc_info:
        sm.validate_file_access("/tmp/file\x00.txt")
    assert "Null byte" in str(exc_info.value)

