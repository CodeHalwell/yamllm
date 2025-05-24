"""
Deprecated interface for YAMLLM.

This module is maintained for backward compatibility only and will be 
removed in a future version. Please use the LLM class instead.
"""

import warnings
from yamllm.core.llm import LLM

warnings.warn(
    "LLMv2 is deprecated and will be removed in a future version. "
    "Please use the LLM class instead, which now includes all provider functionality.",
    DeprecationWarning,
    stacklevel=2
)

class LLMv2(LLM):
    """
    Deprecated. Please use LLM class instead.
    
    This class is maintained for backward compatibility only and
    will be removed in a future version.
    """
    pass