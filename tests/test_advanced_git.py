"""Tests for advanced git workflow automation."""

import pytest
import tempfile
import subprocess
from pathlib import Path
from yamllm.tools.advanced_git import (
    AdvancedGitWorkflow,
    GitStatus,
    CommitAnalysis,
    BranchStrategy,
    ConflictResolutionStrategy
)


@pytest.fixture
def temp_repo():
    """Create a temporary git repository for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize git repo
        subprocess.run(['git', 'init'], cwd=tmpdir, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.email', 'test@test.com'], cwd=tmpdir, check=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=tmpdir, check=True)

        # Create initial commit
        test_file = Path(tmpdir) / 'README.md'
        test_file.write_text('# Test Repo\n')
        subprocess.run(['git', 'add', '.'], cwd=tmpdir, check=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=tmpdir, check=True)

        yield tmpdir


def test_git_workflow_initialization(temp_repo):
    """Test workflow initialization."""
    git = AdvancedGitWorkflow(temp_repo)

    assert git.repo_path == Path(temp_repo)
    assert git._is_git_repo()


def test_invalid_repo_path():
    """Test initialization with invalid path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="Not a git repository"):
            AdvancedGitWorkflow(tmpdir)


def test_get_status_clean(temp_repo):
    """Test status on clean repo."""
    git = AdvancedGitWorkflow(temp_repo)
    status = git.get_status()

    assert isinstance(status, GitStatus)
    assert status.branch in ["main", "master"]
    assert not status.is_dirty
    assert len(status.staged_files) == 0
    assert len(status.unstaged_files) == 0
    assert len(status.untracked_files) == 0


def test_get_status_with_changes(temp_repo):
    """Test status with changes."""
    # Create untracked file
    test_file = Path(temp_repo) / 'new_file.txt'
    test_file.write_text('New content\n')

    # Modify existing file
    readme = Path(temp_repo) / 'README.md'
    readme.write_text('# Modified\n')

    git = AdvancedGitWorkflow(temp_repo)
    status = git.get_status()

    assert status.is_dirty
    assert 'new_file.txt' in status.untracked_files
    assert 'README.md' in status.unstaged_files


def test_detect_components():
    """Test component detection from file paths."""
    git = AdvancedGitWorkflow.__new__(AdvancedGitWorkflow)

    files = [
        'src/api/handler.py',
        'src/api/routes.py',
        'tests/test_api.py',
        'docs/readme.md'
    ]

    components = git._detect_components(files)

    assert 'src' in components
    assert 'tests' in components or 'docs' in components


def test_detect_breaking_changes():
    """Test detection of breaking changes."""
    git = AdvancedGitWorkflow.__new__(AdvancedGitWorkflow)

    # Breaking change indicator
    diff_with_breaking = """
    +++ b/api.py
    +BREAKING CHANGE: Removed old API
    -def old_function():
    """

    assert git._detect_breaking_changes(diff_with_breaking)

    # No breaking change
    diff_without_breaking = """
    +++ b/api.py
    +def new_function():
    +    return True
    """

    assert not git._detect_breaking_changes(diff_without_breaking)


def test_branch_name_generation():
    """Test branch name generation."""
    git = AdvancedGitWorkflow.__new__(AdvancedGitWorkflow)

    # GitHub Flow
    name = git._generate_branch_name("Fix login bug", BranchStrategy.GITHUB_FLOW)
    assert name == "fix-login-bug"

    # GitFlow
    name = git._generate_branch_name("Add new feature", BranchStrategy.GITFLOW)
    assert name.startswith("feature/")
    assert "add-new-feature" in name

    # Trunk-based
    name = git._generate_branch_name("Update docs", BranchStrategy.TRUNK_BASED)
    assert name.startswith("task/")


def test_generate_pr_title():
    """Test PR title generation."""
    git = AdvancedGitWorkflow.__new__(AdvancedGitWorkflow)

    title = git._generate_pr_title("feature/add-user-auth")
    assert "Add User Auth" in title

    title = git._generate_pr_title("fix/bug-in-login")
    assert "Bug In Login" in title


def test_git_status_dataclass():
    """Test GitStatus dataclass."""
    status = GitStatus(
        branch="main",
        is_dirty=True,
        staged_files=["file1.py"],
        unstaged_files=["file2.py"],
        untracked_files=["file3.py"],
        ahead=2,
        behind=1,
        stashed=0
    )

    assert status.branch == "main"
    assert status.is_dirty
    assert len(status.staged_files) == 1
    assert status.ahead == 2
    assert status.behind == 1


def test_commit_analysis_dataclass():
    """Test CommitAnalysis dataclass."""
    analysis = CommitAnalysis(
        files_changed=5,
        lines_added=50,
        lines_deleted=10,
        affected_components=["api", "frontend"],
        suggested_message="feat(api): add new endpoint",
        breaking_changes=False
    )

    assert analysis.files_changed == 5
    assert analysis.lines_added == 50
    assert "api" in analysis.affected_components
    assert not analysis.breaking_changes


def test_analyze_changes(temp_repo):
    """Test change analysis."""
    # Create and stage changes
    new_file = Path(temp_repo) / 'src/api/handler.py'
    new_file.parent.mkdir(parents=True, exist_ok=True)
    new_file.write_text('''
def new_handler():
    """New API handler."""
    return {"status": "ok"}
''')

    subprocess.run(['git', 'add', '.'], cwd=temp_repo, check=True)

    git = AdvancedGitWorkflow(temp_repo)
    analysis = git.analyze_changes()

    assert analysis.files_changed >= 1
    assert analysis.lines_added > 0
    assert len(analysis.affected_components) > 0
    assert analysis.suggested_message  # Should have generated message


def test_smart_branch(temp_repo):
    """Test smart branch creation."""
    git = AdvancedGitWorkflow(temp_repo)

    branch_name = git.smart_branch(
        "Add user authentication",
        strategy=BranchStrategy.GITHUB_FLOW
    )

    assert "user" in branch_name or "authentication" in branch_name

    # Verify branch was created
    result = subprocess.run(
        ['git', 'branch', '--show-current'],
        cwd=temp_repo,
        capture_output=True,
        text=True,
        check=True
    )

    assert result.stdout.strip() == branch_name


def test_branch_strategies():
    """Test all branch strategies."""
    git = AdvancedGitWorkflow.__new__(AdvancedGitWorkflow)

    task = "Implement payment system"

    # GitHub Flow
    name = git._generate_branch_name(task, BranchStrategy.GITHUB_FLOW)
    assert "payment" in name

    # GitFlow
    name = git._generate_branch_name(task, BranchStrategy.GITFLOW)
    assert name.startswith("feature/")

    # Trunk-based
    name = git._generate_branch_name(task, BranchStrategy.TRUNK_BASED)
    assert name.startswith("task/")

    # GitLab Flow
    name = git._generate_branch_name(task, BranchStrategy.GITLAB_FLOW)
    assert "payment" in name


def test_auto_pr_generation(temp_repo):
    """Test PR info generation."""
    # Create feature branch with commits
    subprocess.run(['git', 'checkout', '-b', 'feature/test'], cwd=temp_repo, check=True)

    test_file = Path(temp_repo) / 'feature.py'
    test_file.write_text('print("feature")\n')
    subprocess.run(['git', 'add', '.'], cwd=temp_repo, check=True)
    subprocess.run(['git', 'commit', '-m', 'Add feature'], cwd=temp_repo, check=True)

    git = AdvancedGitWorkflow(temp_repo)
    pr_info = git.auto_pr(base="main")

    assert "title" in pr_info
    assert "body" in pr_info
    assert "base" in pr_info
    assert "head" in pr_info
    assert pr_info["base"] == "main"
    assert pr_info["head"] == "feature/test"


def test_conflict_resolution_strategies():
    """Test conflict resolution strategy enum."""
    assert ConflictResolutionStrategy.OURS.value == "ours"
    assert ConflictResolutionStrategy.THEIRS.value == "theirs"
    assert ConflictResolutionStrategy.MANUAL.value == "manual"
    assert ConflictResolutionStrategy.SMART.value == "smart"


def test_branch_strategy_enum():
    """Test branch strategy enum."""
    assert BranchStrategy.GITFLOW.value == "gitflow"
    assert BranchStrategy.GITHUB_FLOW.value == "github_flow"
    assert BranchStrategy.TRUNK_BASED.value == "trunk_based"
    assert BranchStrategy.GITLAB_FLOW.value == "gitlab_flow"
