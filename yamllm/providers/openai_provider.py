"""
Compatibility shim for OpenAI provider.

This module re-exports the unified core provider to avoid duplication and
legacy UI behavior. Prefer importing providers from
`yamllm.core.providers` directly.
"""

from yamllm.providers.openai import OpenAIProvider  # re-export

__all__ = ["OpenAIProvider"]

