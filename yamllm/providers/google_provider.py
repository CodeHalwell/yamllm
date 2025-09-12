"""
Compatibility shim for Google Gemini provider.

This module re-exports the unified core provider to avoid duplication and
legacy UI behavior.
"""

from yamllm.providers.google import GoogleGeminiProvider  # re-export

__all__ = ["GoogleGeminiProvider"]

