import pytest
import logging
from unittest.mock import MagicMock, patch
from yamllm.core.llm import setup_logging, LLM
from yamllm.core.parser import YamlLMConfig
