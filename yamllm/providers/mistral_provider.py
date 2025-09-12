"""
Compatibility shim for Mistral provider.

This module re-exports the unified core provider to avoid duplication and
legacy UI behavior.
"""

from yamllm.providers.mistral import MistralProvider  # re-export

__all__ = ["MistralProvider"]

