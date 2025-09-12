"""
Compatibility shim for provider-layer exceptions.

Re-export the core ProviderError so providers raise the same exception type
used throughout the core. This avoids divergence between layers.
"""

from yamllm.core.exceptions import ProviderError  # re-export

__all__ = ["ProviderError"]
