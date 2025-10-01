"""
YAMLLM CLI - Backward compatibility wrapper.

This module provides backward compatibility by redirecting to the new
modular CLI structure in yamllm.cli package.

The monolithic CLI (1140+ lines) has been refactored into focused modules:
- yamllm.cli.main: Main entry point and command assembly
- yamllm.cli.tools: Tool management commands  
- yamllm.cli.config: Configuration commands
- yamllm.cli.chat: Chat interface commands
- yamllm.cli.memory: Memory management commands

This file remains to maintain backward compatibility for any code that
imports or executes yamllm/cli.py directly.
"""

import sys
from yamllm.cli.main import main

__version__ = "0.1.12"

# Expose main for direct script execution
if __name__ == "__main__":
    sys.exit(main())
