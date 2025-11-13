"""Advanced git workflow automation with intelligent operations."""

import subprocess
import os
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ConflictResolutionStrategy(Enum):
    """Strategy for resolving merge conflicts."""
    OURS = "ours"
    THEIRS = "theirs"
    MANUAL = "manual"
    SMART = "smart"  # LLM-assisted


class BranchStrategy(Enum):
    """Git branching strategy."""
    GITFLOW = "gitflow"
    TRUNK_BASED = "trunk_based"
    GITHUB_FLOW = "github_flow"
    GITLAB_FLOW = "gitlab_flow"


@dataclass
class GitStatus:
    """Git repository status."""
    branch: str
    is_dirty: bool
    staged_files: List[str]
    unstaged_files: List[str]
    untracked_files: List[str]
    ahead: int
    behind: int
    stashed: int


@dataclass
class ConflictInfo:
    """Information about a merge conflict."""
    file_path: str
    conflict_markers: List[Tuple[int, int]]  # (start_line, end_line)
    ours_content: str
    theirs_content: str
    base_content: Optional[str]


@dataclass
class CommitAnalysis:
    """Analysis of commit changes."""
    files_changed: int
    lines_added: int
    lines_deleted: int
    affected_components: List[str]
    suggested_message: str
    breaking_changes: bool


class AdvancedGitWorkflow:
    """Advanced git workflow automation."""

    def __init__(
        self,
        repo_path: str,
        llm=None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize advanced git workflow.

        Args:
            repo_path: Path to git repository
            llm: Optional LLM for intelligent operations
            logger: Optional logger
        """
        self.repo_path = Path(repo_path)
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)

        if not self._is_git_repo():
            raise ValueError(f"Not a git repository: {repo_path}")

    def _is_git_repo(self) -> bool:
        """Check if directory is a git repository."""
        return (self.repo_path / '.git').exists()

    def _run_git(self, *args, check: bool = True) -> subprocess.CompletedProcess:
        """Run git command."""
        cmd = ['git', '-C', str(self.repo_path)] + list(args)
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check
        )

    def get_status(self) -> GitStatus:
        """Get comprehensive git status."""
        # Current branch
        result = self._run_git('branch', '--show-current')
        branch = result.stdout.strip()

        # Status
        result = self._run_git('status', '--porcelain')
        lines = result.stdout.strip().split('\n') if result.stdout.strip() else []

        staged = []
        unstaged = []
        untracked = []

        for line in lines:
            if not line:
                continue

            status = line[:2]
            filepath = line[3:]

            if status[0] in ['M', 'A', 'D', 'R']:
                staged.append(filepath)
            if status[1] in ['M', 'D']:
                unstaged.append(filepath)
            if status == '??':
                untracked.append(filepath)

        # Ahead/behind
        result = self._run_git('rev-list', '--left-right', '--count', f'HEAD...@{{u}}', check=False)
        ahead, behind = 0, 0
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            if len(parts) == 2:
                ahead, behind = int(parts[0]), int(parts[1])

        # Stashed
        result = self._run_git('stash', 'list')
        stashed = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0

        is_dirty = bool(staged or unstaged or untracked)

        return GitStatus(
            branch=branch,
            is_dirty=is_dirty,
            staged_files=staged,
            unstaged_files=unstaged,
            untracked_files=untracked,
            ahead=ahead,
            behind=behind,
            stashed=stashed
        )

    def smart_commit(
        self,
        files: Optional[List[str]] = None,
        message: Optional[str] = None,
        auto_stage: bool = True
    ) -> str:
        """
        Create intelligent commit with auto-generated message.

        Args:
            files: Specific files to commit (None = all)
            message: Commit message (None = auto-generate)
            auto_stage: Whether to auto-stage files

        Returns:
            Commit hash
        """
        # Stage files
        if auto_stage:
            if files:
                for f in files:
                    self._run_git('add', f)
            else:
                self._run_git('add', '-A')

        # Analyze changes
        analysis = self.analyze_changes()

        # Generate message if not provided
        if not message:
            message = analysis.suggested_message
            self.logger.info(f"Generated commit message: {message}")

        # Create commit
        result = self._run_git('commit', '-m', message)

        # Get commit hash
        result = self._run_git('rev-parse', 'HEAD')
        commit_hash = result.stdout.strip()

        return commit_hash

    def analyze_changes(self) -> CommitAnalysis:
        """Analyze staged changes and suggest commit message."""
        # Get diff stats
        result = self._run_git('diff', '--cached', '--numstat')
        lines = result.stdout.strip().split('\n') if result.stdout.strip() else []

        files_changed = 0
        lines_added = 0
        lines_deleted = 0
        changed_files = []

        for line in lines:
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) == 3:
                added, deleted, filepath = parts
                files_changed += 1
                lines_added += int(added) if added != '-' else 0
                lines_deleted += int(deleted) if deleted != '-' else 0
                changed_files.append(filepath)

        # Detect affected components
        components = self._detect_components(changed_files)

        # Check for breaking changes
        result = self._run_git('diff', '--cached')
        diff_content = result.stdout
        breaking = self._detect_breaking_changes(diff_content)

        # Generate suggested message
        suggested_message = self._generate_commit_message(
            files_changed, lines_added, lines_deleted, components, breaking
        )

        return CommitAnalysis(
            files_changed=files_changed,
            lines_added=lines_added,
            lines_deleted=lines_deleted,
            affected_components=components,
            suggested_message=suggested_message,
            breaking_changes=breaking
        )

    def _detect_components(self, files: List[str]) -> List[str]:
        """Detect affected components from file paths."""
        components = set()

        for filepath in files:
            parts = filepath.split('/')
            if len(parts) > 1:
                # Add directory as component
                components.add(parts[0])

            # Detect component from filename
            filename = os.path.basename(filepath)
            if '_' in filename:
                components.add(filename.split('_')[0])

        return list(components)[:3]  # Top 3 components

    def _detect_breaking_changes(self, diff: str) -> bool:
        """Detect potential breaking changes in diff."""
        breaking_patterns = [
            r'BREAKING CHANGE:',
            r'!:',
            r'def\s+\w+\([^)]*\)\s*->\s*\w+:',  # Function signature change
            r'class\s+\w+\([^)]*\):.*removed',
        ]

        for pattern in breaking_patterns:
            if re.search(pattern, diff, re.IGNORECASE):
                return True

        return False

    def _generate_commit_message(
        self,
        files_changed: int,
        lines_added: int,
        lines_deleted: int,
        components: List[str],
        breaking: bool
    ) -> str:
        """Generate commit message from analysis."""
        if not self.llm:
            # Simple template-based message
            scope = components[0] if components else "core"
            return f"Update {scope}: {files_changed} files changed"

        # Use LLM for better message
        result = self._run_git('diff', '--cached')
        diff_content = result.stdout[:2000]  # Limit diff size

        prompt = f"""Generate a conventional commit message for these changes:

Files changed: {files_changed}
Lines added: {lines_added}
Lines deleted: {lines_deleted}
Components: {', '.join(components)}
Breaking: {breaking}

Diff (partial):
{diff_content}

Generate a commit message in conventional commit format:
<type>(<scope>): <description>

Where type is: feat, fix, docs, style, refactor, test, chore
Keep description under 72 characters and focus on WHY, not WHAT.
{"Add BREAKING CHANGE: footer if breaking" if breaking else ""}

Commit message:"""

        try:
            message = self.llm.query(prompt).strip()
            # Extract just the commit message if LLM adds extra text
            lines = message.split('\n')
            return lines[0]
        except Exception as e:
            self.logger.warning(f"LLM commit message generation failed: {e}")
            scope = components[0] if components else "core"
            return f"update({scope}): {files_changed} files changed"

    def smart_branch(
        self,
        task: str,
        strategy: BranchStrategy = BranchStrategy.GITHUB_FLOW
    ) -> str:
        """
        Create intelligently named branch for task.

        Args:
            task: Task description
            strategy: Branching strategy to use

        Returns:
            Branch name
        """
        # Generate branch name
        branch_name = self._generate_branch_name(task, strategy)

        # Create and checkout branch
        self._run_git('checkout', '-b', branch_name)

        self.logger.info(f"Created branch: {branch_name}")
        return branch_name

    def _generate_branch_name(self, task: str, strategy: BranchStrategy) -> str:
        """Generate branch name from task description."""
        # Clean task description
        clean_task = re.sub(r'[^\w\s-]', '', task.lower())
        clean_task = re.sub(r'\s+', '-', clean_task.strip())
        clean_task = clean_task[:50]  # Limit length

        # Apply strategy prefix
        if strategy == BranchStrategy.GITFLOW:
            # Detect type
            if any(word in task.lower() for word in ['fix', 'bug', 'error']):
                prefix = 'hotfix'
            elif any(word in task.lower() for word in ['feature', 'add', 'new']):
                prefix = 'feature'
            else:
                prefix = 'develop'
            return f"{prefix}/{clean_task}"

        elif strategy == BranchStrategy.GITHUB_FLOW:
            # Simple descriptive names
            return clean_task

        elif strategy == BranchStrategy.TRUNK_BASED:
            # Short-lived branches
            return f"task/{clean_task}"

        else:
            return clean_task

    def smart_merge(
        self,
        branch: str,
        strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.SMART,
        squash: bool = False
    ) -> bool:
        """
        Intelligently merge branch with conflict resolution.

        Args:
            branch: Branch to merge
            strategy: Conflict resolution strategy
            squash: Whether to squash commits

        Returns:
            True if successful
        """
        try:
            # Attempt merge
            args = ['merge', branch]
            if squash:
                args.append('--squash')

            self._run_git(*args)
            return True

        except subprocess.CalledProcessError:
            # Merge conflict occurred
            self.logger.warning(f"Merge conflict detected with branch: {branch}")

            if strategy == ConflictResolutionStrategy.SMART and self.llm:
                return self._smart_resolve_conflicts()
            elif strategy == ConflictResolutionStrategy.OURS:
                self._run_git('checkout', '--ours', '.')
                self._run_git('add', '-A')
                return True
            elif strategy == ConflictResolutionStrategy.THEIRS:
                self._run_git('checkout', '--theirs', '.')
                self._run_git('add', '-A')
                return True
            else:
                # Manual resolution required
                return False

    def _smart_resolve_conflicts(self) -> bool:
        """Use LLM to intelligently resolve merge conflicts."""
        # Get conflicted files
        result = self._run_git('diff', '--name-only', '--diff-filter=U')
        conflicted_files = result.stdout.strip().split('\n')

        self.logger.info(f"Resolving {len(conflicted_files)} conflicted files")

        for filepath in conflicted_files:
            if not filepath:
                continue

            try:
                # Read conflict
                full_path = self.repo_path / filepath
                with open(full_path, 'r') as f:
                    content = f.read()

                # Extract conflicts
                conflict_info = self._parse_conflict(content)

                # Resolve with LLM
                resolved = self._llm_resolve_conflict(filepath, conflict_info)

                # Write resolved content
                with open(full_path, 'w') as f:
                    f.write(resolved)

                # Stage resolved file
                self._run_git('add', filepath)

            except Exception as e:
                self.logger.error(f"Failed to resolve {filepath}: {e}")
                return False

        return True

    def _parse_conflict(self, content: str) -> ConflictInfo:
        """Parse conflict markers in file content."""
        # Find conflict markers
        ours_start = content.find('<<<<<<< HEAD')
        separator = content.find('=======', ours_start)
        theirs_end = content.find('>>>>>>>', separator)

        if ours_start == -1 or separator == -1 or theirs_end == -1:
            raise ValueError("No conflict markers found")

        ours_content = content[ours_start + 12:separator].strip()
        theirs_content = content[separator + 7:theirs_end].strip()

        return ConflictInfo(
            file_path="",
            conflict_markers=[(ours_start, theirs_end)],
            ours_content=ours_content,
            theirs_content=theirs_content,
            base_content=None
        )

    def _llm_resolve_conflict(self, filepath: str, conflict: ConflictInfo) -> str:
        """Use LLM to resolve conflict."""
        prompt = f"""Resolve this merge conflict in {filepath}:

OUR CHANGES:
{conflict.ours_content}

THEIR CHANGES:
{conflict.theirs_content}

Provide the resolved version that integrates both changes intelligently.
Return ONLY the resolved code, no explanations:"""

        try:
            resolved = self.llm.query(prompt).strip()
            # Remove markdown code blocks if present
            if '```' in resolved:
                resolved = re.sub(r'```\w*\n', '', resolved)
                resolved = resolved.replace('```', '')
            return resolved.strip()
        except Exception as e:
            self.logger.error(f"LLM conflict resolution failed: {e}")
            # Fallback to keeping ours
            return conflict.ours_content

    def auto_pr(
        self,
        base: str = "main",
        title: Optional[str] = None,
        body: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Automatically create pull request.

        Args:
            base: Base branch for PR
            title: PR title (auto-generated if None)
            body: PR body (auto-generated if None)

        Returns:
            PR information dict
        """
        status = self.get_status()

        # Generate title from branch or commits
        if not title:
            title = self._generate_pr_title(status.branch)

        # Generate body from commits
        if not body:
            body = self._generate_pr_body(base, status.branch)

        pr_info = {
            "title": title,
            "body": body,
            "base": base,
            "head": status.branch
        }

        self.logger.info(f"PR: {title}")
        return pr_info

    def _generate_pr_title(self, branch: str) -> str:
        """Generate PR title from branch name."""
        # Convert branch name to title
        title = branch.replace('-', ' ').replace('_', ' ')
        title = re.sub(r'(feature|fix|hotfix|task)/', '', title)
        return title.title()

    def _generate_pr_body(self, base: str, head: str) -> str:
        """Generate PR body from commit history."""
        # Get commits in branch
        result = self._run_git('log', f'{base}..{head}', '--pretty=format:%s')
        commits = result.stdout.strip().split('\n') if result.stdout.strip() else []

        body_parts = ["## Changes\n"]
        for commit in commits[:10]:  # Max 10 commits
            body_parts.append(f"- {commit}")

        # Add diff stats
        result = self._run_git('diff', f'{base}...{head}', '--shortstat')
        if result.stdout.strip():
            body_parts.append(f"\n## Stats\n{result.stdout.strip()}")

        return "\n".join(body_parts)
