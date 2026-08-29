"""Shared parsing helpers for LLM responses.

Centralises the JSON-extraction logic that was previously duplicated across
the planner, reasoner, and observer components.
"""

import json
import re
from typing import Any, Dict, Iterator, Optional

# Fences may carry any label in any case (```json, ```JSON, ```python, ```)
_FENCE_RE = re.compile(r"```([A-Za-z]*)[ \t]*\n?(.*?)```", re.DOTALL)


def _candidates(text: str) -> Iterator[str]:
    """Yield plausible JSON payloads from an LLM response, best first."""
    match = _FENCE_RE.search(text)
    if match:
        candidate = match.group(2).strip()
        if candidate:
            yield candidate

    start = text.find("{")
    end = text.rfind("}") + 1
    if 0 <= start < end:
        yield text[start:end]

    yield text


def extract_json_block(text: str) -> str:
    """Extract the most likely JSON payload from an LLM response.

    Handles fenced markdown blocks (```json ... ```, any label or case),
    leading/trailing prose, and bare JSON objects. Returns the original
    text when no candidate is found so callers can surface a meaningful
    parse error.
    """
    if not text:
        return text
    return next(_candidates(text))


def parse_json_response(
    text: str, default: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Parse a JSON object out of an LLM response.

    Tries each extraction candidate in turn (fenced block, bare object,
    raw text), so a fence whose content is not valid JSON still falls
    back to brace extraction.

    Args:
        text: Raw model output.
        default: Value returned when nothing parseable is found. When None,
            a ValueError is raised instead.

    Returns:
        The parsed dictionary.
    """
    if text:
        for candidate in _candidates(text):
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data

    if default is not None:
        return default
    raise ValueError("No JSON object found in response")
