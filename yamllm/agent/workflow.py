"""Workflow manager for common task patterns."""

import logging
from typing import Dict, Any, Optional, List

from .core import Agent
from .models import AgentState


class WorkflowManager:
    """
    Manages pre-defined workflow templates.

    Workflows are reusable patterns for common tasks like:
    - Bug fixing
    - Feature implementation
    - Code refactoring
    - Testing
    """

    WORKFLOWS = {
        "debug_bug": {
            "name": "Debug Bug",
            "description": "Systematically debug and fix a bug",
            "steps": [
                "Reproduce the bug and understand the issue",
                "Read relevant code files to identify the problem area",
                "Identify root cause of the bug",
                "Propose a fix for the bug",
                "Implement the fix",
                "Test the fix to ensure it works",
                "Commit changes with descriptive message"
            ],
            "required_context": ["bug_description"],
            "optional_context": ["file_path", "error_message", "expected_behavior"]
        },
        "implement_feature": {
            "name": "Implement Feature",
            "description": "Design and implement a new feature",
            "steps": [
                "Understand requirements and scope",
                "Design architecture and approach",
                "Identify files that need to be modified",
                "Implement core functionality",
                "Add error handling and edge cases",
                "Write tests for the new feature",
                "Update documentation",
                "Commit changes"
            ],
            "required_context": ["feature_description"],
            "optional_context": ["requirements", "constraints", "files"]
        },
        "refactor_code": {
            "name": "Refactor Code",
            "description": "Improve code quality without changing behavior",
            "steps": [
                "Analyze current code structure",
                "Identify improvement opportunities",
                "Plan refactoring steps",
                "Execute refactoring incrementally",
                "Run tests to verify behavior unchanged",
                "Commit refactored code"
            ],
            "required_context": ["target_file"],
            "optional_context": ["refactoring_goals", "constraints"]
        },
        "write_tests": {
            "name": "Write Tests",
            "description": "Write comprehensive tests for code",
            "steps": [
                "Analyze code to understand functionality",
                "Identify test cases and edge cases",
                "Write unit tests",
                "Write integration tests if needed",
                "Run tests and verify coverage",
                "Commit test code"
            ],
            "required_context": ["target_file"],
            "optional_context": ["test_framework", "coverage_goal"]
        },
        "review_code": {
            "name": "Review Code",
            "description": "Review code for quality, bugs, and improvements",
            "steps": [
                "Read and understand the code changes",
                "Check for potential bugs or errors",
                "Evaluate code style and best practices",
                "Check for security vulnerabilities",
                "Suggest improvements and optimizations",
                "Provide summary of findings"
            ],
            "required_context": ["files_to_review"],
            "optional_context": ["diff", "pr_description"]
        },
        "investigate_issue": {
            "name": "Investigate Issue",
            "description": "Investigate and diagnose an issue",
            "steps": [
                "Understand the reported issue",
                "Gather relevant information and logs",
                "Examine related code and configuration",
                "Identify potential causes",
                "Test hypotheses",
                "Document findings and recommendations"
            ],
            "required_context": ["issue_description"],
            "optional_context": ["logs", "error_messages", "reproduction_steps"]
        }
    }

    def __init__(self, agent: Agent, logger: Optional[logging.Logger] = None):
        """
        Initialize workflow manager.

        Args:
            agent: Agent instance to use for execution
            logger: Optional logger
        """
        self.agent = agent
        self.logger = logger or logging.getLogger(__name__)

    def execute_workflow(
        self,
        workflow_name: str,
        context: Dict[str, Any]
    ) -> AgentState:
        """
        Execute a named workflow.

        Args:
            workflow_name: Name of workflow to execute
            context: Context dictionary with required information

        Returns:
            Final agent state

        Raises:
            ValueError: If workflow not found or required context missing
        """
        if workflow_name not in self.WORKFLOWS:
            available = ", ".join(self.WORKFLOWS.keys())
            raise ValueError(f"Unknown workflow: {workflow_name}. Available: {available}")

        workflow = self.WORKFLOWS[workflow_name]

        # Validate context
        self._validate_context(workflow, context)

        # Create goal from workflow
        goal = self._create_goal_from_workflow(workflow, context)

        self.logger.info(f"Executing workflow: {workflow['name']}")

        # Execute through agent
        return self.agent.execute(goal, context)

    def list_workflows(self) -> List[Dict[str, Any]]:
        """List available workflows with descriptions."""
        return [
            {
                "name": name,
                "title": workflow["name"],
                "description": workflow["description"],
                "required_context": workflow.get("required_context", []),
                "optional_context": workflow.get("optional_context", [])
            }
            for name, workflow in self.WORKFLOWS.items()
        ]

    def get_workflow_info(self, workflow_name: str) -> Dict[str, Any]:
        """Get detailed information about a workflow."""
        if workflow_name not in self.WORKFLOWS:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        return self.WORKFLOWS[workflow_name]

    def _validate_context(self, workflow: Dict[str, Any], context: Dict[str, Any]) -> None:
        """
        Validate that required context is provided.

        Args:
            workflow: Workflow definition
            context: Provided context

        Raises:
            ValueError: If required context missing
        """
        required = workflow.get("required_context", [])
        missing = [key for key in required if key not in context]

        if missing:
            raise ValueError(
                f"Missing required context for workflow '{workflow['name']}': {', '.join(missing)}"
            )

    def _create_goal_from_workflow(self, workflow: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Create a goal string from workflow template.

        Args:
            workflow: Workflow definition
            context: Context dictionary

        Returns:
            Goal string for agent
        """
        goal_parts = [
            f"Workflow: {workflow['name']}",
            f"Description: {workflow['description']}",
            "",
            "Context:"
        ]

        # Add context information
        for key, value in context.items():
            if isinstance(value, str) and len(value) < 200:
                goal_parts.append(f"- {key}: {value}")
            else:
                goal_parts.append(f"- {key}: (provided)")

        goal_parts.extend([
            "",
            "Follow these steps:",
        ])

        # Add workflow steps
        for i, step in enumerate(workflow['steps'], 1):
            goal_parts.append(f"{i}. {step}")

        goal_parts.extend([
            "",
            "Complete as many steps as possible to achieve the workflow goal."
        ])

        return "\n".join(goal_parts)


# Pre-configured workflow instances for common use cases
class DebugWorkflow:
    """Helper for debug workflow."""

    @staticmethod
    def execute(agent: Agent, bug_description: str, **kwargs) -> AgentState:
        """Execute debug workflow."""
        context = {"bug_description": bug_description, **kwargs}
        manager = WorkflowManager(agent)
        return manager.execute_workflow("debug_bug", context)


class ImplementWorkflow:
    """Helper for implement feature workflow."""

    @staticmethod
    def execute(agent: Agent, feature_description: str, **kwargs) -> AgentState:
        """Execute implement feature workflow."""
        context = {"feature_description": feature_description, **kwargs}
        manager = WorkflowManager(agent)
        return manager.execute_workflow("implement_feature", context)


class RefactorWorkflow:
    """Helper for refactor workflow."""

    @staticmethod
    def execute(agent: Agent, target_file: str, **kwargs) -> AgentState:
        """Execute refactor workflow."""
        context = {"target_file": target_file, **kwargs}
        manager = WorkflowManager(agent)
        return manager.execute_workflow("refactor_code", context)
