"""Git integration tools for repository operations."""

from .base import Tool
from typing import Dict, Any, Optional
import subprocess
import os
from pathlib import Path


class GitError(Exception):
    """Exception raised for git command errors."""
    pass


class GitTool(Tool):
    """Base class for git operations with common functionality."""

    def __init__(self, name: str, description: str, cwd: Optional[str] = None):
        super().__init__(name=name, description=description)
        self.cwd = cwd or os.getcwd()

    def _run_git_command(self, args: list, capture_output: bool = True, check: bool = True) -> subprocess.CompletedProcess:
        """Execute a git command and return the result."""
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=self.cwd,
                capture_output=capture_output,
                text=True,
                check=check,
                timeout=30
            )
            return result
        except subprocess.CalledProcessError as e:
            raise GitError(f"Git command failed: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise GitError(f"Git command timed out after 30 seconds")
        except FileNotFoundError:
            raise GitError("Git is not installed or not in PATH")

    def _check_git_repo(self) -> bool:
        """Check if current directory is a git repository."""
        try:
            result = self._run_git_command(["rev-parse", "--git-dir"], check=False)
            return result.returncode == 0
        except GitError:
            return False


class GitStatusTool(GitTool):
    """Get the status of the git repository."""

    def __init__(self, cwd: Optional[str] = None):
        super().__init__(
            name="git_status",
            description="Get the current status of the git repository including staged, unstaged, and untracked files",
            cwd=cwd
        )

    def execute(self) -> Dict[str, Any]:
        """Execute git status command."""
        if not self._check_git_repo():
            return {"error": "Not a git repository"}

        try:
            result = self._run_git_command(["status", "--porcelain", "-b"])
            lines = result.stdout.strip().split("\n")

            branch = ""
            if lines and lines[0].startswith("##"):
                branch = lines[0][3:].split("...")[0].strip()
                lines = lines[1:]

            staged = []
            unstaged = []
            untracked = []

            for line in lines:
                if not line:
                    continue
                status = line[:2]
                filename = line[3:]

                if status[0] != ' ' and status[0] != '?':
                    staged.append({"file": filename, "status": status[0]})
                if status[1] != ' ' and status[1] != '?':
                    unstaged.append({"file": filename, "status": status[1]})
                if status == '??':
                    untracked.append(filename)

            return {
                "branch": branch,
                "staged": staged,
                "unstaged": unstaged,
                "untracked": untracked,
                "clean": len(staged) == 0 and len(unstaged) == 0 and len(untracked) == 0
            }
        except GitError as e:
            return {"error": str(e)}

    def _get_parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {},
            "required": []
        }


class GitDiffTool(GitTool):
    """Show differences in the repository."""

    def __init__(self, cwd: Optional[str] = None):
        super().__init__(
            name="git_diff",
            description="Show changes in the working directory or staged changes. Can diff specific files or all changes.",
            cwd=cwd
        )

    def execute(self, staged: bool = False, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute git diff command.

        Args:
            staged: If True, show staged changes (--cached). If False, show unstaged changes.
            file_path: Optional specific file to diff.
        """
        if not self._check_git_repo():
            return {"error": "Not a git repository"}

        try:
            args = ["diff"]
            if staged:
                args.append("--cached")
            if file_path:
                args.append(file_path)

            result = self._run_git_command(args)

            return {
                "diff": result.stdout,
                "staged": staged,
                "file": file_path,
                "has_changes": bool(result.stdout.strip())
            }
        except GitError as e:
            return {"error": str(e)}

    def _get_parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "staged": {
                    "type": "boolean",
                    "description": "Show staged changes instead of unstaged",
                    "default": False
                },
                "file_path": {
                    "type": "string",
                    "description": "Optional path to specific file to diff"
                }
            },
            "required": []
        }


class GitLogTool(GitTool):
    """Show commit history."""

    def __init__(self, cwd: Optional[str] = None):
        super().__init__(
            name="git_log",
            description="Show commit history with author, date, and message",
            cwd=cwd
        )

    def execute(self, max_count: int = 10, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute git log command.

        Args:
            max_count: Maximum number of commits to show (default: 10)
            file_path: Optional file path to show history for specific file
        """
        if not self._check_git_repo():
            return {"error": "Not a git repository"}

        try:
            args = [
                "log",
                f"-{max_count}",
                "--pretty=format:%H|%an|%ae|%ad|%s",
                "--date=iso"
            ]
            if file_path:
                args.extend(["--", file_path])

            result = self._run_git_command(args)

            commits = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("|", 4)
                if len(parts) == 5:
                    commits.append({
                        "hash": parts[0][:8],  # Short hash
                        "author": parts[1],
                        "email": parts[2],
                        "date": parts[3],
                        "message": parts[4]
                    })

            return {
                "commits": commits,
                "count": len(commits),
                "file": file_path
            }
        except GitError as e:
            return {"error": str(e)}

    def _get_parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "max_count": {
                    "type": "integer",
                    "description": "Maximum number of commits to show",
                    "default": 10
                },
                "file_path": {
                    "type": "string",
                    "description": "Optional file path to show history for"
                }
            },
            "required": []
        }


class GitBranchTool(GitTool):
    """Manage git branches."""

    def __init__(self, cwd: Optional[str] = None):
        super().__init__(
            name="git_branch",
            description="List, create, or switch git branches",
            cwd=cwd
        )

    def execute(self, action: str = "list", branch_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute git branch operations.

        Args:
            action: Operation to perform - 'list', 'create', or 'switch'
            branch_name: Branch name for create/switch operations
        """
        if not self._check_git_repo():
            return {"error": "Not a git repository"}

        try:
            if action == "list":
                result = self._run_git_command(["branch", "-a"])
                branches = []
                current = None

                for line in result.stdout.strip().split("\n"):
                    if not line:
                        continue
                    is_current = line.startswith("*")
                    branch = line[2:].strip()
                    branches.append(branch)
                    if is_current:
                        current = branch

                return {
                    "branches": branches,
                    "current": current
                }

            elif action == "create":
                if not branch_name:
                    return {"error": "branch_name required for create action"}

                result = self._run_git_command(["branch", branch_name])
                return {
                    "action": "created",
                    "branch": branch_name,
                    "message": f"Branch '{branch_name}' created"
                }

            elif action == "switch":
                if not branch_name:
                    return {"error": "branch_name required for switch action"}

                result = self._run_git_command(["checkout", branch_name])
                return {
                    "action": "switched",
                    "branch": branch_name,
                    "message": f"Switched to branch '{branch_name}'"
                }

            else:
                return {"error": f"Unknown action: {action}. Use 'list', 'create', or 'switch'"}

        except GitError as e:
            return {"error": str(e)}

    def _get_parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "create", "switch"],
                    "description": "Operation to perform",
                    "default": "list"
                },
                "branch_name": {
                    "type": "string",
                    "description": "Branch name for create/switch operations"
                }
            },
            "required": []
        }


class GitCommitTool(GitTool):
    """Create a git commit."""

    def __init__(self, cwd: Optional[str] = None):
        super().__init__(
            name="git_commit",
            description="Create a git commit with staged changes",
            cwd=cwd
        )

    def execute(self, message: str, add_all: bool = False) -> Dict[str, Any]:
        """
        Execute git commit command.

        Args:
            message: Commit message
            add_all: If True, stage all changes before committing
        """
        if not self._check_git_repo():
            return {"error": "Not a git repository"}

        if not message:
            return {"error": "Commit message is required"}

        try:
            # Optionally stage all changes
            if add_all:
                self._run_git_command(["add", "-A"])

            # Create commit
            result = self._run_git_command(["commit", "-m", message])

            # Get commit hash
            hash_result = self._run_git_command(["rev-parse", "HEAD"])
            commit_hash = hash_result.stdout.strip()[:8]

            return {
                "success": True,
                "commit_hash": commit_hash,
                "message": message,
                "output": result.stdout
            }
        except GitError as e:
            return {"error": str(e)}

    def _get_parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Commit message describing the changes"
                },
                "add_all": {
                    "type": "boolean",
                    "description": "Stage all changes before committing",
                    "default": False
                }
            },
            "required": ["message"]
        }


class GitPushTool(GitTool):
    """Push commits to remote repository."""

    def __init__(self, cwd: Optional[str] = None):
        super().__init__(
            name="git_push",
            description="Push local commits to remote repository",
            cwd=cwd
        )

    def execute(self, remote: str = "origin", branch: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
        """
        Execute git push command.

        Args:
            remote: Remote name (default: origin)
            branch: Branch name to push (default: current branch)
            force: Force push (use with caution!)
        """
        if not self._check_git_repo():
            return {"error": "Not a git repository"}

        try:
            args = ["push", remote]

            if branch:
                args.append(branch)

            if force:
                args.append("--force")

            result = self._run_git_command(args)

            return {
                "success": True,
                "remote": remote,
                "branch": branch or "current",
                "output": result.stderr  # Git push writes to stderr
            }
        except GitError as e:
            return {"error": str(e)}

    def _get_parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "remote": {
                    "type": "string",
                    "description": "Remote repository name",
                    "default": "origin"
                },
                "branch": {
                    "type": "string",
                    "description": "Branch to push (default: current branch)"
                },
                "force": {
                    "type": "boolean",
                    "description": "Force push (WARNING: can overwrite remote history)",
                    "default": False
                }
            },
            "required": []
        }


class GitPullTool(GitTool):
    """Pull changes from remote repository."""

    def __init__(self, cwd: Optional[str] = None):
        super().__init__(
            name="git_pull",
            description="Pull and merge changes from remote repository",
            cwd=cwd
        )

    def execute(self, remote: str = "origin", branch: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute git pull command.

        Args:
            remote: Remote name (default: origin)
            branch: Branch to pull from (default: current branch tracking)
        """
        if not self._check_git_repo():
            return {"error": "Not a git repository"}

        try:
            args = ["pull", remote]
            if branch:
                args.append(branch)

            result = self._run_git_command(args)

            return {
                "success": True,
                "remote": remote,
                "branch": branch or "current",
                "output": result.stdout
            }
        except GitError as e:
            return {"error": str(e)}

    def _get_parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "remote": {
                    "type": "string",
                    "description": "Remote repository name",
                    "default": "origin"
                },
                "branch": {
                    "type": "string",
                    "description": "Branch to pull from"
                }
            },
            "required": []
        }


# Export all git tools
__all__ = [
    "GitStatusTool",
    "GitDiffTool",
    "GitLogTool",
    "GitBranchTool",
    "GitCommitTool",
    "GitPushTool",
    "GitPullTool",
    "GitError"
]
