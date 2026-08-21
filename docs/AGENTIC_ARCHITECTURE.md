# Agentic Loop Architecture Design

**Version:** 1.0
**Status:** Design Document
**Target Release:** v1.0

---

## Executive Summary

This document outlines the architecture for implementing autonomous agentic capabilities in yamllm, transforming it from a reactive tool-user to an autonomous task completer. The design follows the **ReAct (Reason + Act)** pattern with **multi-turn orchestration**, **task decomposition**, and **workflow management**.

### Goals

1. **Autonomous Task Completion**: Enable yamllm to independently complete complex tasks
2. **Multi-Turn Tool Orchestration**: Chain multiple tool calls intelligently
3. **Task Planning & Decomposition**: Break complex goals into manageable subtasks
4. **Progress Tracking**: Visualize task state and completion status
5. **Error Recovery**: Handle failures gracefully with retry/fallback strategies

### Non-Goals

- Full general AI (AGI) - focused on developer/productivity tasks
- Unsupervised operation - requires user-defined goals
- Learning from experience - no persistent model fine-tuning (v1.0)

---

## Current State vs. Target State

### Current Architecture (Reactive)

```
User Prompt → LLM → Tool Request → Tool Execution → LLM → Response
       ↓
   Single Turn
```

**Limitations:**
- One tool call per turn
- No task planning
- No autonomous iteration
- Limited error handling
- No progress tracking

### Target Architecture (Agentic)

```
User Goal → Task Planner → Agent Loop → Task Completion
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
                 Reasoner            Actor
                    ↓                   ↓
              Plan/Decide         Execute Tools
                    ↓                   ↓
                 Observer ← Results ← Tool Output
                    ↓
              Update State → Next Action
                    ↑
                    └──── Loop until goal achieved
```

**Capabilities:**
- Multi-turn autonomous operation
- Task decomposition and planning
- Smart tool selection and chaining
- Progress tracking and visualization
- Error recovery and adaptation

---

## Architecture Components

### 1. Agent Core (`yamllm/agent/core.py`)

The central orchestrator implementing the ReAct loop.

#### AgentState

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

@dataclass
class Task:
    id: str
    description: str
    status: TaskStatus
    dependencies: List[str]  # Task IDs that must complete first
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class AgentState:
    """Current state of the agent execution."""
    goal: str
    tasks: List[Task]
    current_task_id: Optional[str]
    iteration: int
    max_iterations: int
    thought_history: List[str]
    action_history: List[Dict[str, Any]]
    completed: bool
    success: bool
    error: Optional[str] = None
```

#### Agent Class

```python
class Agent:
    """
    Autonomous agent implementing ReAct loop.

    Coordinates between Planner, Reasoner, Actor, and Observer
    to complete complex tasks autonomously.
    """

    def __init__(
        self,
        llm,  # LLM instance
        max_iterations: int = 10,
        enable_planning: bool = True,
        enable_reflection: bool = True,
        logger = None
    ):
        self.llm = llm
        self.max_iterations = max_iterations
        self.enable_planning = enable_planning
        self.enable_reflection = enable_reflection
        self.logger = logger or logging.getLogger(__name__)

        # Initialize components
        self.planner = TaskPlanner(llm)
        self.reasoner = Reasoner(llm)
        self.actor = Actor(llm)
        self.observer = Observer(llm)

    async def execute(self, goal: str, context: Optional[Dict] = None) -> AgentState:
        """
        Execute the agentic loop to achieve the given goal.

        Args:
            goal: High-level goal to achieve
            context: Optional context (files, repo info, etc.)

        Returns:
            Final AgentState with results
        """
        # Initialize state
        state = AgentState(
            goal=goal,
            tasks=[],
            current_task_id=None,
            iteration=0,
            max_iterations=self.max_iterations,
            thought_history=[],
            action_history=[],
            completed=False,
            success=False
        )

        # Phase 1: Planning
        if self.enable_planning:
            state = await self.planner.decompose_goal(goal, context, state)

        # Phase 2: Execution Loop (ReAct)
        while not state.completed and state.iteration < state.max_iterations:
            state.iteration += 1

            # Step 1: REASON - What should I do next?
            thought, next_task = await self.reasoner.reason(state)
            state.thought_history.append(thought)
            state.current_task_id = next_task.id

            # Step 2: ACT - Execute the action
            action_result = await self.actor.act(next_task, state)
            state.action_history.append(action_result)

            # Step 3: OBSERVE - Interpret results and update state
            state = await self.observer.observe(action_result, state)

            # Step 4: Check completion
            state = self._check_goal_completion(state)

            # Optional: REFLECT - Learn from errors/successes
            if self.enable_reflection and state.iteration % 3 == 0:
                state = await self._reflect(state)

        return state
```

---

### 2. Task Planner (`yamllm/agent/planner.py`)

Decomposes high-level goals into executable subtasks.

#### TaskPlanner Class

```python
class TaskPlanner:
    """
    Breaks down high-level goals into actionable subtasks.

    Uses LLM to analyze goal and create dependency-ordered task list.
    """

    def __init__(self, llm):
        self.llm = llm

    async def decompose_goal(
        self,
        goal: str,
        context: Optional[Dict],
        state: AgentState
    ) -> AgentState:
        """
        Decompose goal into subtasks.

        Prompt engineering approach:
        1. Provide goal and context
        2. Ask LLM to break into steps
        3. Identify dependencies
        4. Create Task objects
        """

        prompt = self._build_planning_prompt(goal, context)

        # Get task decomposition from LLM
        response = await self.llm.query_async(prompt)

        # Parse response into Task objects
        tasks = self._parse_tasks(response)

        # Validate dependencies
        tasks = self._validate_dependencies(tasks)

        state.tasks = tasks
        return state

    def _build_planning_prompt(self, goal: str, context: Optional[Dict]) -> str:
        """Build prompt for task decomposition."""
        return f"""
You are a task planning assistant. Break down the following goal into concrete, actionable subtasks.

Goal: {goal}

Context:
{self._format_context(context)}

Requirements:
1. Each task should be specific and measurable
2. Identify dependencies between tasks
3. Order tasks logically
4. Use available tools effectively

Respond in JSON format:
{{
  "tasks": [
    {{
      "id": "task_1",
      "description": "...",
      "dependencies": [],
      "required_tools": ["tool_name"],
      "estimated_complexity": "low|medium|high"
    }},
    ...
  ]
}}
"""

    def _parse_tasks(self, response: str) -> List[Task]:
        """Parse LLM response into Task objects."""
        try:
            data = json.loads(response)
            tasks = []

            for i, task_data in enumerate(data.get("tasks", [])):
                task = Task(
                    id=task_data.get("id", f"task_{i}"),
                    description=task_data["description"],
                    status=TaskStatus.PENDING,
                    dependencies=task_data.get("dependencies", []),
                    metadata={
                        "tools": task_data.get("required_tools", []),
                        "complexity": task_data.get("estimated_complexity", "medium")
                    }
                )
                tasks.append(task)

            return tasks
        except Exception as e:
            # Fallback: create single task from goal
            return [Task(
                id="task_1",
                description=response,
                status=TaskStatus.PENDING,
                dependencies=[]
            )]
```

---

### 3. Reasoner (`yamllm/agent/reasoner.py`)

Decides what action to take next based on current state.

#### Reasoner Class

```python
class Reasoner:
    """
    Reasoning component - decides next actions.

    Implements the 'Reason' part of ReAct.
    """

    def __init__(self, llm):
        self.llm = llm

    async def reason(self, state: AgentState) -> Tuple[str, Task]:
        """
        Reason about next action to take.

        Returns:
            (thought, next_task): Reasoning and selected task
        """

        # Get available tasks (not blocked by dependencies)
        available_tasks = self._get_available_tasks(state)

        if not available_tasks:
            # All tasks blocked or completed
            thought = "No available tasks. Checking if goal is achieved."
            return thought, None

        # Build reasoning prompt
        prompt = self._build_reasoning_prompt(state, available_tasks)

        # Get LLM's reasoning
        response = await self.llm.query_async(prompt)

        # Parse response
        thought, selected_task_id = self._parse_reasoning(response)

        # Get the actual task
        next_task = next((t for t in available_tasks if t.id == selected_task_id), available_tasks[0])

        return thought, next_task

    def _get_available_tasks(self, state: AgentState) -> List[Task]:
        """Get tasks that can be executed (dependencies met)."""
        available = []

        for task in state.tasks:
            if task.status in [TaskStatus.COMPLETED, TaskStatus.IN_PROGRESS]:
                continue

            # Check if all dependencies are completed
            deps_met = all(
                self._get_task_by_id(dep_id, state).status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )

            if deps_met:
                available.append(task)

        return available

    def _build_reasoning_prompt(self, state: AgentState, available_tasks: List[Task]) -> str:
        """Build prompt for reasoning."""
        return f"""
You are working on this goal: {state.goal}

Progress so far:
- Iteration: {state.iteration}/{state.max_iterations}
- Completed tasks: {self._count_completed(state.tasks)}/{len(state.tasks)}
- Recent thoughts: {state.thought_history[-3:] if state.thought_history else 'None'}
- Recent actions: {state.action_history[-3:] if state.action_history else 'None'}

Available tasks to work on:
{self._format_tasks(available_tasks)}

Question: Which task should I work on next and why?

Think step-by-step:
1. What have I accomplished so far?
2. What is blocking me from achieving the goal?
3. Which task will make the most progress?
4. Are there any risks or dependencies I should consider?

Respond in JSON:
{{
  "thought": "My reasoning here...",
  "selected_task_id": "task_X",
  "rationale": "Why I chose this task..."
}}
"""
```

---

### 4. Actor (`yamllm/agent/actor.py`)

Executes actions (primarily tool calls).

#### Actor Class

```python
class Actor:
    """
    Action component - executes tasks.

    Implements the 'Act' part of ReAct.
    """

    def __init__(self, llm):
        self.llm = llm

    async def act(self, task: Task, state: AgentState) -> Dict[str, Any]:
        """
        Execute a task.

        Returns:
            Action result with tool calls, outputs, and status
        """

        # Update task status
        task.status = TaskStatus.IN_PROGRESS

        # Build action prompt
        prompt = self._build_action_prompt(task, state)

        try:
            # Execute with tools enabled
            response = await self.llm.get_completion_with_tools_async(prompt)

            action_result = {
                "task_id": task.id,
                "success": True,
                "response": response.get("content"),
                "tool_calls": response.get("tool_calls", []),
                "tool_results": response.get("tool_results", []),
                "error": None
            }

            # Update task
            task.status = TaskStatus.COMPLETED
            task.result = action_result

        except Exception as e:
            action_result = {
                "task_id": task.id,
                "success": False,
                "response": None,
                "tool_calls": [],
                "tool_results": [],
                "error": str(e)
            }

            task.status = TaskStatus.FAILED
            task.error = str(e)

        return action_result

    def _build_action_prompt(self, task: Task, state: AgentState) -> str:
        """Build prompt for action execution."""
        return f"""
Goal: {state.goal}

Current Task: {task.description}

Available Tools:
{self._list_available_tools()}

Context from previous tasks:
{self._format_completed_tasks(state)}

Execute this task using the available tools. Be specific and thorough.
"""
```

---

### 5. Observer (`yamllm/agent/observer.py`)

Interprets action results and updates state.

#### Observer Class

```python
class Observer:
    """
    Observation component - interprets results.

    Implements the 'Observe' part of ReAct.
    """

    def __init__(self, llm):
        self.llm = llm

    async def observe(self, action_result: Dict[str, Any], state: AgentState) -> AgentState:
        """
        Observe action result and update state.

        Analyzes:
        - Did the action succeed?
        - What was learned?
        - Does this unblock other tasks?
        - Are we closer to the goal?
        """

        # Build observation prompt
        prompt = self._build_observation_prompt(action_result, state)

        # Get LLM's interpretation
        response = await self.llm.query_async(prompt)

        # Parse observations
        observations = self._parse_observations(response)

        # Update state based on observations
        state = self._update_state(observations, state, action_result)

        return state

    def _build_observation_prompt(self, action_result: Dict, state: AgentState) -> str:
        """Build prompt for observation."""
        return f"""
Goal: {state.goal}

Action taken: {self._describe_action(action_result)}

Result:
- Success: {action_result['success']}
- Tools used: {[tc['function']['name'] for tc in action_result.get('tool_calls', [])]}
- Output: {action_result.get('response', 'N/A')}
- Error: {action_result.get('error', 'None')}

Questions:
1. Was this action successful?
2. What did we learn?
3. Does this unblock any other tasks?
4. Are we closer to achieving the goal?
5. Do we need to adjust our plan?

Respond in JSON:
{{
  "success_assessment": true/false,
  "learned": "What we learned...",
  "unblocked_tasks": ["task_id", ...],
  "progress_made": "Description of progress...",
  "plan_adjustments": "Any needed adjustments..."
}}
"""

    def _update_state(
        self,
        observations: Dict,
        state: AgentState,
        action_result: Dict
    ) -> AgentState:
        """Update state based on observations."""

        # Unblock dependent tasks if applicable
        if observations.get("success_assessment"):
            task_id = action_result["task_id"]
            for task in state.tasks:
                if task_id in task.dependencies and task.status == TaskStatus.BLOCKED:
                    task.status = TaskStatus.PENDING

        # Add learned info to state
        if "learned" in observations:
            if "learnings" not in state.metadata:
                state.metadata["learnings"] = []
            state.metadata["learnings"].append(observations["learned"])

        return state
```

---

### 6. Workflow Manager (`yamllm/agent/workflow.py`)

Manages common workflow patterns (e.g., "debug bug", "implement feature").

#### WorkflowManager Class

```python
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
                "Reproduce the bug",
                "Read relevant code files",
                "Identify root cause",
                "Propose fix",
                "Implement fix",
                "Test fix",
                "Commit changes"
            ]
        },
        "implement_feature": {
            "name": "Implement Feature",
            "description": "Design and implement a new feature",
            "steps": [
                "Understand requirements",
                "Design architecture",
                "Identify files to modify",
                "Implement core functionality",
                "Add error handling",
                "Write tests",
                "Update documentation",
                "Commit changes"
            ]
        },
        "refactor_code": {
            "name": "Refactor Code",
            "description": "Improve code quality without changing behavior",
            "steps": [
                "Analyze current code",
                "Identify improvement opportunities",
                "Plan refactoring steps",
                "Execute refactoring",
                "Run tests to verify behavior",
                "Commit changes"
            ]
        }
    }

    def __init__(self, agent: Agent):
        self.agent = agent

    async def execute_workflow(
        self,
        workflow_name: str,
        context: Dict[str, Any]
    ) -> AgentState:
        """Execute a named workflow."""

        if workflow_name not in self.WORKFLOWS:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        workflow = self.WORKFLOWS[workflow_name]

        # Create goal from workflow
        goal = self._create_goal_from_workflow(workflow, context)

        # Execute through agent
        return await self.agent.execute(goal, context)

    def _create_goal_from_workflow(self, workflow: Dict, context: Dict) -> str:
        """Create a goal string from workflow template."""
        goal = f"{workflow['description']}\n\n"
        goal += "Follow these steps:\n"
        for i, step in enumerate(workflow['steps'], 1):
            goal += f"{i}. {step}\n"
        goal += f"\nContext: {context}"
        return goal
```

---

## Integration with Existing Code

### Changes Required

#### 1. LLM Class (`yamllm/core/llm.py`)

Add async method for agent use:

```python
async def query_async(self, prompt: str, **kwargs) -> str:
    """Async version of query for agent use."""
    # Implementation using existing async provider if available
    pass

async def get_completion_with_tools_async(self, prompt: str) -> Dict[str, Any]:
    """Async tool completion for agent."""
    # Implementation
    pass
```

#### 2. CLI Integration (`yamllm/cli/agent.py`)

New command module for agent operations:

```python
def setup_agent_commands(subparsers):
    """Setup agent-related CLI commands."""

    # yamllm agent debug <issue_description>
    debug_parser = subparsers.add_parser("agent debug")

    # yamllm agent implement <feature_description>
    implement_parser = subparsers.add_parser("agent implement")

    # yamllm agent workflow <workflow_name>
    workflow_parser = subparsers.add_parser("agent workflow")
```

#### 3. UI Updates (`yamllm/ui/agent_ui.py`)

New UI components for agent visualization:

```python
class AgentUI:
    """UI for displaying agent progress."""

    def render_task_tree(self, state: AgentState):
        """Render task dependency tree with progress."""
        pass

    def stream_thought(self, thought: str):
        """Stream agent's reasoning in real-time."""
        pass

    def show_action(self, action_result: Dict):
        """Show action being executed."""
        pass
```

---

## Example Usage

### Programmatic

```python
from yamllm import LLM
from yamllm.agent import Agent

# Initialize LLM
llm = LLM(config_path="config.yaml")

# Create agent
agent = Agent(llm, max_iterations=15)

# Execute autonomous task
state = await agent.execute(
    goal="Fix the login bug in auth.py - users can't log in with special characters in password",
    context={
        "repo_path": "/path/to/repo",
        "files": ["auth.py", "test_auth.py"]
    }
)

# Check results
print(f"Success: {state.success}")
print(f"Tasks completed: {len([t for t in state.tasks if t.status == TaskStatus.COMPLETED])}")
```

### CLI

```bash
# Debug a bug autonomously
yamllm agent debug "Fix login issue with special characters" --files auth.py

# Implement a feature
yamllm agent implement "Add password reset functionality" --workflow implement_feature

# Run workflow
yamllm agent workflow debug_bug --context '{"file": "auth.py", "error": "..."}'
```

---

## Progress Visualization

### Terminal UI

```
┌─ yamllm Agent ─────────────────────────────────────────────┐
│ Goal: Fix login bug with special characters                │
│ Iteration: 3/15 | Progress: 40%                            │
├────────────────────────────────────────────────────────────┤
│ Tasks:                                                      │
│ ✓ task_1: Reproduce bug                    [COMPLETED]     │
│ ✓ task_2: Read auth.py                     [COMPLETED]     │
│ ▶ task_3: Identify root cause              [IN PROGRESS]   │
│ ⋯ task_4: Implement fix                    [PENDING]       │
│ ⋯ task_5: Test fix                         [PENDING]       │
├────────────────────────────────────────────────────────────┤
│ Current Thought:                                            │
│ "The bug appears to be in the password validation regex.   │
│  Need to check if special characters are properly escaped."│
├────────────────────────────────────────────────────────────┤
│ Action: Using tool [file_read] on auth.py:45-60           │
└────────────────────────────────────────────────────────────┘
```

---

## Error Handling & Recovery

### Retry Strategy

```python
class RetryStrategy:
    """Handle tool execution failures."""

    async def retry_with_backoff(
        self,
        action: Callable,
        max_retries: int = 3
    ) -> Any:
        """Retry action with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return await action()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
```

### Fallback Strategies

1. **Simplify Task**: Break complex task into simpler subtasks
2. **Alternative Tools**: Try different tool for same goal
3. **Ask User**: Prompt user for guidance when stuck
4. **Skip Task**: Mark as blocked and continue with others

---

## Testing Strategy

### Unit Tests

```python
# Test task decomposition
async def test_task_planner_decomposes_goal():
    planner = TaskPlanner(mock_llm)
    state = AgentState(goal="Fix bug", ...)

    result = await planner.decompose_goal("Fix bug in auth.py", {}, state)

    assert len(result.tasks) > 0
    assert all(t.status == TaskStatus.PENDING for t in result.tasks)

# Test reasoning
async def test_reasoner_selects_available_task():
    reasoner = Reasoner(mock_llm)
    state = AgentState(...)

    thought, task = await reasoner.reason(state)

    assert task is not None
    assert thought != ""
```

### Integration Tests

```python
# Test full agent loop
async def test_agent_completes_simple_goal():
    agent = Agent(mock_llm, max_iterations=5)

    state = await agent.execute("List files in current directory")

    assert state.completed
    assert state.success
    assert any("ls" in str(action) for action in state.action_history)
```

---

## Performance Considerations

### Optimization Targets

- **Planning Time**: < 2 seconds for goal decomposition
- **Reasoning Time**: < 1 second per iteration
- **Action Time**: Variable (depends on tool)
- **Total Time**: < 30 seconds for simple tasks, < 5 minutes for complex

### Caching Strategy

```python
# Cache task plans for similar goals
@lru_cache(maxsize=100)
def get_cached_plan(goal_hash: str) -> List[Task]:
    """Retrieve cached task plan for similar goal."""
    pass
```

---

## Security Considerations

### Tool Permissions

```python
class AgentSecurityManager:
    """Manage permissions for agent tool use."""

    REQUIRE_CONFIRMATION = [
        "git_push",      # Pushes to remote
        "git_commit",    # Creates commits
        "file_write",    # Writes files
        "shell_exec"     # Executes commands
    ]

    async def confirm_action(self, tool_name: str, args: Dict) -> bool:
        """Request user confirmation for sensitive actions."""
        if tool_name in self.REQUIRE_CONFIRMATION:
            # Show confirmation prompt
            return await self._prompt_user(
                f"Agent wants to use {tool_name} with args {args}. Allow?"
            )
        return True
```

---

## Future Enhancements (Post v1.0)

1. **Learning from Experience**: Store successful workflows in vector DB
2. **Multi-Agent Collaboration**: Multiple agents working on subtasks
3. **Human-in-the-Loop**: Interactive checkpoints for approval
4. **Advanced Planning**: Use graph-based planning algorithms
5. **Performance Profiling**: Track which strategies work best
6. **Custom Workflows**: User-defined workflow templates

---

## Implementation Roadmap

### Phase 1: Core Components (Weeks 1-2)
- [ ] Implement AgentState and Task models
- [ ] Build basic Agent class with ReAct loop
- [ ] Create TaskPlanner with simple decomposition
- [ ] Implement Reasoner for task selection
- [ ] Build Actor for tool execution

### Phase 2: Integration (Weeks 3-4)
- [ ] Integrate with existing LLM class
- [ ] Add async methods for agent use
- [ ] Create Observer for result interpretation
- [ ] Implement basic error handling

### Phase 3: UI & Workflows (Week 5)
- [ ] Build AgentUI for progress visualization
- [ ] Create WorkflowManager with templates
- [ ] Add CLI commands for agent operations
- [ ] Implement task tree visualization

### Phase 4: Testing & Polish (Week 6)
- [ ] Write comprehensive tests
- [ ] Add security confirmations
- [ ] Performance optimization
- [ ] Documentation and examples

---

## Success Metrics

- ✅ **Task Completion Rate**: > 80% for well-defined tasks
- ✅ **Average Iterations**: < 10 for typical developer tasks
- ✅ **User Satisfaction**: 4/5+ rating for autonomous mode
- ✅ **Error Recovery**: Graceful handling of failures without crashes
- ✅ **Performance**: < 3 minutes for bug fixes, < 10 minutes for features

---

## Appendix: Prompt Templates

### Planning Prompt

```
You are a task planning expert. Given a high-level goal, break it down into concrete, actionable subtasks.

Goal: {goal}
Context: {context}

Create a task plan with:
1. Clear, measurable tasks
2. Dependency relationships
3. Required tools for each task
4. Estimated complexity

Output as JSON...
```

### Reasoning Prompt

```
You are executing this goal: {goal}

Current state:
- Completed: {completed_tasks}
- In progress: {current_task}
- Remaining: {pending_tasks}

Question: What should I do next and why?

Think step-by-step and respond in JSON...
```

### Action Prompt

```
Execute this task: {task_description}

Available tools: {tools}
Context: {context}

Be specific and use tools effectively to complete the task.
```

---

**End of Document**
