import sys
import os
from pathlib import Path

# Get the project root directory
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

import pytest

@pytest.fixture
def config():
    from yamllm.config import Config
    return Config()

@pytest.fixture
def parser():
    from yamllm.parser import Parser
    return Parser()