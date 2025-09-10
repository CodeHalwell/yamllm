"""
Marshmallow schemas for input validation in YAMLLM web API.

This module defines validation schemas for API endpoints
using Marshmallow for comprehensive input validation.
"""

from marshmallow import Schema, fields, validate, validates_schema, ValidationError


class QuerySchema(Schema):
    """
    Schema for validating LLM query requests.
    
    Validates prompt, model selection, and optional parameters
    for LLM query API endpoints.
    """
    
    prompt = fields.Str(
        required=True,
        validate=[
            validate.Length(min=1, max=10000, error="Prompt must be between 1 and 10,000 characters"),
        ],
        error_messages={
            'required': 'Prompt is required',
            'invalid': 'Prompt must be a valid string'
        }
    )
    
    model = fields.Str(
        validate=validate.OneOf([
            'gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo',
            'gemini-pro', 'gemini-1.5-pro',
            'deepseek-chat', 'deepseek-coder',
            'mistral-large', 'mistral-medium', 'mistral-small'
        ], error="Invalid model selection"),
        missing='gpt-4o-mini',
        error_messages={
            'invalid': 'Model must be a valid string'
        }
    )
    
    session_id = fields.Str(
        validate=[
            validate.Length(max=100, error="Session ID must not exceed 100 characters"),
            validate.Regexp(
                r'^[a-zA-Z0-9_-]*$',
                error="Session ID can only contain letters, numbers, underscores, and hyphens"
            )
        ],
        allow_none=True,
        missing=None,
        error_messages={
            'invalid': 'Session ID must be a valid string'
        }
    )
    
    temperature = fields.Float(
        validate=validate.Range(min=0.0, max=2.0, error="Temperature must be between 0.0 and 2.0"),
        missing=0.7,
        error_messages={
            'invalid': 'Temperature must be a valid number'
        }
    )
    
    max_tokens = fields.Int(
        validate=validate.Range(min=1, max=4096, error="Max tokens must be between 1 and 4096"),
        missing=1000,
        error_messages={
            'invalid': 'Max tokens must be a valid integer'
        }
    )
    
    system_prompt = fields.Str(
        validate=validate.Length(max=2000, error="System prompt must not exceed 2000 characters"),
        allow_none=True,
        missing=None,
        error_messages={
            'invalid': 'System prompt must be a valid string'
        }
    )
    
    @validates_schema
    def validate_query(self, data, **kwargs):
        """
        Cross-field validation for query parameters.
        
        Args:
            data (dict): The input data to validate.
            
        Raises:
            ValidationError: If validation fails.
        """
        # Check for potentially harmful content patterns
        prompt = data.get('prompt', '')
        
        # Basic content filtering
        harmful_patterns = [
            'DROP TABLE', 'DELETE FROM', 'UPDATE SET',
            '<script>', '</script>', 'javascript:',
            'eval(', 'exec(', 'system(',
        ]
        
        prompt_upper = prompt.upper()
        for pattern in harmful_patterns:
            if pattern.upper() in prompt_upper:
                raise ValidationError(f'Prompt contains potentially harmful content: {pattern}')
        
        # Check prompt isn't only whitespace
        if not prompt.strip():
            raise ValidationError('Prompt cannot be empty or only whitespace')


class ConversationSchema(Schema):
    """
    Schema for validating conversation history requests.
    
    Validates session identifiers and pagination parameters
    for conversation history API endpoints.
    """
    
    session_id = fields.Str(
        required=True,
        validate=[
            validate.Length(min=1, max=100, error="Session ID must be between 1 and 100 characters"),
            validate.Regexp(
                r'^[a-zA-Z0-9_-]+$',
                error="Session ID can only contain letters, numbers, underscores, and hyphens"
            )
        ],
        error_messages={
            'required': 'Session ID is required',
            'invalid': 'Session ID must be a valid string'
        }
    )
    
    limit = fields.Int(
        validate=validate.Range(min=1, max=100, error="Limit must be between 1 and 100"),
        missing=20,
        error_messages={
            'invalid': 'Limit must be a valid integer'
        }
    )
    
    offset = fields.Int(
        validate=validate.Range(min=0, error="Offset must be non-negative"),
        missing=0,
        error_messages={
            'invalid': 'Offset must be a valid integer'
        }
    )


class MessageSchema(Schema):
    """
    Schema for validating individual message objects.
    
    Used for validating messages in conversation history
    and new message submissions.
    """
    
    role = fields.Str(
        required=True,
        validate=validate.OneOf(
            ['system', 'user', 'assistant', 'tool'],
            error="Role must be one of: system, user, assistant, tool"
        ),
        error_messages={
            'required': 'Message role is required',
            'invalid': 'Role must be a valid string'
        }
    )
    
    content = fields.Str(
        required=True,
        validate=validate.Length(min=1, max=50000, error="Content must be between 1 and 50,000 characters"),
        error_messages={
            'required': 'Message content is required',
            'invalid': 'Content must be a valid string'
        }
    )
    
    name = fields.Str(
        validate=[
            validate.Length(max=64, error="Name must not exceed 64 characters"),
            validate.Regexp(
                r'^[a-zA-Z0-9_-]*$',
                error="Name can only contain letters, numbers, underscores, and hyphens"
            )
        ],
        allow_none=True,
        missing=None,
        error_messages={
            'invalid': 'Name must be a valid string'
        }
    )
    
    timestamp = fields.DateTime(
        allow_none=True,
        missing=None,
        error_messages={
            'invalid': 'Timestamp must be a valid datetime'
        }
    )


class SessionSchema(Schema):
    """
    Schema for validating session management requests.
    
    Used for creating, updating, and deleting conversation sessions.
    """
    
    session_id = fields.Str(
        validate=[
            validate.Length(min=1, max=100, error="Session ID must be between 1 and 100 characters"),
            validate.Regexp(
                r'^[a-zA-Z0-9_-]+$',
                error="Session ID can only contain letters, numbers, underscores, and hyphens"
            )
        ],
        allow_none=True,
        missing=None,
        error_messages={
            'invalid': 'Session ID must be a valid string'
        }
    )
    
    name = fields.Str(
        validate=validate.Length(max=200, error="Session name must not exceed 200 characters"),
        allow_none=True,
        missing=None,
        error_messages={
            'invalid': 'Session name must be a valid string'
        }
    )
    
    description = fields.Str(
        validate=validate.Length(max=1000, error="Description must not exceed 1000 characters"),
        allow_none=True,
        missing=None,
        error_messages={
            'invalid': 'Description must be a valid string'
        }
    )