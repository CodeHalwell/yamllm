# File: /yamllm/yamllm/src/yamllm/__init__.py

"""
This is the yamllm package.

The yamllm package provides an easy way to set up a Large Language Model (LLM) using YAML configuration files.
"""

from yamllm.src.yamllm.core.llm import LLM
from yamllm.src.yamllm.core.parser import parse_yaml
from yamllm.src.yamllm.core.utils import log_error, handle_exception

__all__ = ['LLM', 'parse_yaml', 'log_error', 'handle_exception']