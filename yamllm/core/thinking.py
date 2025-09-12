from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional
import time


class ThinkingStep(Enum):
    ANALYSIS = "analysis"
    TOOL_PLANNING = "tool_planning"
    EXECUTION_PLAN = "execution_plan"
    REFLECTION = "reflection"


@dataclass
class ThinkingBlock:
    step: ThinkingStep
    content: str
    timestamp: float
    tools_considered: Optional[List[str]] = None
    confidence: float = 0.0


class ThinkingManager:
    """Generate and format visible reasoning blocks for a prompt."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        show_tool_reasoning: bool = True,
        thinking_model: Optional[str] = None,
        max_thinking_tokens: int = 2000,
        stream_thinking: bool = True,
        save_thinking: bool = False,
        temperature: float = 0.7,
    ) -> None:
        self.enabled = enabled
        self.show_tool_reasoning = show_tool_reasoning
        self.thinking_model = thinking_model
        self.max_thinking_tokens = max_thinking_tokens
        self.stream_thinking = stream_thinking
        self.save_thinking = save_thinking
        self.temperature = temperature
        self.thinking_blocks: List[ThinkingBlock] = []
        # Runtime controls for display hygiene
        self.mode: str = "auto"  # off|on|auto
        self.redact_logs: bool = True

    def should_show(self, prompt: str, *, available_tools: List[str]) -> bool:
        """Decide whether to show thinking based on mode and heuristic complexity.

        - off: never
        - on: always (if enabled)
        - auto: show for complex tasks detected by a lightweight heuristic
        """
        if not self.enabled:
            return False
        mode = (self.mode or "auto").lower()
        if mode == "off":
            return False
        if mode == "on":
            return True
        # auto: complexity detection
        return self._is_complex(prompt, available_tools)

    def _is_complex(self, prompt: str, available_tools: List[str]) -> bool:
        """Simple heuristic: long prompts, multi-part questions, explicit planning words,
        code blocks, or when multiple tool intents likely.
        """
        p = (prompt or "").lower()
        if len(p) >= 220:
            return True
        # Multi-part cues
        multi_cues = ["first,", "second,", "third,", "step", "steps", "plan", "outline"]
        if any(c in p for c in multi_cues):
            return True
        # Code markers
        if "```" in p or "def " in p or "class " in p:
            return True
        # Multiple tool intents
        intents = 0
        intents += 1 if any(k in p for k in ("http://", "https://", "search", "scrape", "website")) else 0
        intents += 1 if any(k in p for k in ("calculate", "sum", "difference", "+", "-", "*", "/")) else 0
        intents += 1 if any(k in p for k in ("convert", "units", "km", "miles", "celsius", "fahrenheit")) else 0
        intents += 1 if any(k in p for k in ("timezone", "time in", "utc", "pst", "est")) else 0
        return intents >= 2

    def generate_thinking(
        self, prompt: str, available_tools: List[str], provider_client: Any, model_fallback: str
    ) -> List[ThinkingBlock]:
        if not self.enabled:
            return []

        blocks: List[ThinkingBlock] = []

        # 1) Analysis
        analysis_prompt = self._create_analysis_prompt(prompt, available_tools)
        analysis = self._get_thinking_response(analysis_prompt, provider_client, model_fallback)
        blocks.append(
            ThinkingBlock(step=ThinkingStep.ANALYSIS, content=analysis, timestamp=time.time())
        )

        # 2) Tool planning
        if available_tools and self.show_tool_reasoning:
            tool_prompt = self._create_tool_planning_prompt(prompt, available_tools, analysis)
            tool_thinking = self._get_thinking_response(tool_prompt, provider_client, model_fallback)
            blocks.append(
                ThinkingBlock(
                    step=ThinkingStep.TOOL_PLANNING,
                    content=tool_thinking,
                    timestamp=time.time(),
                    tools_considered=available_tools,
                )
            )

        # 3) Execution plan
        exec_prompt = self._create_execution_prompt(prompt, blocks)
        exec_plan = self._get_thinking_response(exec_prompt, provider_client, model_fallback)
        blocks.append(
            ThinkingBlock(step=ThinkingStep.EXECUTION_PLAN, content=exec_plan, timestamp=time.time())
        )

        self.thinking_blocks = blocks
        return blocks

    def _create_analysis_prompt(self, prompt: str, available_tools: List[str]) -> str:
        tools_text = ", ".join(available_tools) if available_tools else "None"
        return (
            "<thinking>\n"
            "I will analyze the user's request and plan a helpful response.\n\n"
            f"User request: {prompt}\n\n"
            f"Available tools: {tools_text}\n\n"
            "Consider:\n"
            "- The user's true intent\n"
            "- What output would be most useful\n"
            "- Whether tools or external info are needed\n"
            "- Any ambiguities to clarify\n\n"
            "Analysis:\n"
        )

    def _create_tool_planning_prompt(
        self, prompt: str, available_tools: List[str], analysis: str
    ) -> str:
        tools_text = ", ".join(available_tools)
        return (
            "<thinking>\n"
            "Based on my analysis, I will plan tool usage.\n\n"
            f"User request: {prompt}\n\n"
            f"My analysis: {analysis}\n\n"
            f"Available tools: {tools_text}\n\n"
            "Plan:\n"
            "- Which tools are most relevant and why\n"
            "- What to ask each tool and in what order\n"
            "- What to avoid\n\n"
            "Tool strategy:\n"
        )

    def _create_execution_prompt(self, prompt: str, blocks: List[ThinkingBlock]) -> str:
        prev = "\n".join(f"{b.step.value}: {b.content}" for b in blocks)
        return (
            "<thinking>\n"
            "Now I will outline the execution of my response.\n\n"
            f"User request: {prompt}\n\n"
            f"Previous thinking:\n{prev}\n\n"
            "Execution plan:\n"
            "- Structure of the response\n"
            "- Key steps and information needed\n"
            "- How to present the result clearly\n\n"
            "Final approach:\n"
        )

    def _get_thinking_response(self, prompt: str, provider_client: Any, model_fallback: str) -> str:
        try:
            model_name = self.thinking_model or model_fallback
            params = dict(
                messages=[{"role": "user", "content": prompt}],
                model=model_name,
                temperature=self.temperature,
                max_tokens=self.max_thinking_tokens,
                top_p=1.0,
                stream=False,
            )
            resp = provider_client.get_completion(**params)
            # Best-effort normalization
            if hasattr(resp, "choices"):
                # OpenAI-like
                return getattr(resp.choices[0].message, "content", "") or ""
            if isinstance(resp, dict):
                # Anthropic-like
                content = resp.get("content")
                if isinstance(content, list) and content:
                    first = content[0]
                    if isinstance(first, dict) and "text" in first:
                        return first.get("text", "")
                return str(content or "")
            return str(resp)
        except Exception as e:
            # Redact internal error text if redaction enabled
            return "[Thinking unavailable]" if self.redact_logs else f"[Thinking error: {e}]"

    def format_thinking_for_display(self, blocks: List[ThinkingBlock]) -> str:
        if not blocks:
            return ""
        parts: List[str] = ["<thinking>\n"]
        for b in blocks:
            parts.append(f"=== {b.step.value.upper()} ===\n{b.content}\n")
            if b.tools_considered:
                parts.append(f"\nTools considered: {', '.join(b.tools_considered)}\n")
            parts.append("=" * 50 + "\n")
        parts.append("</thinking>\n")
        return "".join(parts)

