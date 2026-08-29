"""Shared parsing helpers for LLM responses.

Centralises the JSON-extraction logic that was previously duplicated across
the planner, reasoner, and observer components.
"""

import json
import re
from typing import Any, Dict, Optional

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_block(text: str) -> str:
    """Extract the most likely JSON payload from an LLM response.

    Handles fenced markdown blocks (```json ... ```), leading/trailing prose,
    and bare JSON objects. Returns the original text when no candidate is
    found so callers can surface a meaningful parse error.
    """
    if not text:
        return text

    match = _FENCE_RE.search(text)
    if match:
        candidate = match.group(1).strip()
        if candidate:
            return candidate

    start = text.find("{")
    end = text.rfind("}") + 1
    if 0 <= start < end:
        return text[start:end]

    return text


def parse_json_response(
    text: str, default: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Parse a JSON object out of an LLM response.

    Args:
        text: Raw model output.
        default: Value returned when nothing parseable is found. When None,
            a JSONDecodeError/ValueError is raised instead.

    Returns:
        The parsed dictionary.
    """
    candidate = extract_json_block(text)
    try:
        data = json.loads(candidate)
        if isinstance(data, dict):
            return data
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")
    except (json.JSONDecodeError, ValueError):
        if default is not None:
            return default
        raise
