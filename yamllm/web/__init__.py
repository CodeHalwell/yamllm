"""
Web API module for YAMLLM.

This module provides a Flask-based web API for the YAMLLM library,
including CSRF protection and input validation.
"""

from .app import create_app
from .schemas import QuerySchema, ConversationSchema

__all__ = ['create_app', 'QuerySchema', 'ConversationSchema']