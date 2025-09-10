"""
Flask-WTF forms for YAMLLM web interface.

This module defines forms with CSRF protection for the web interface,
providing secure form handling for LLM queries and settings.
"""

from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, TextAreaField, IntegerField, FloatField
from wtforms.validators import DataRequired, Length, NumberRange, Optional, Regexp


class QueryForm(FlaskForm):
    """
    Form for LLM query submission with CSRF protection.
    
    Provides secure form handling for user queries including
    prompt input, model selection, and optional parameters.
    """
    
    prompt = TextAreaField(
        'Prompt',
        validators=[
            DataRequired(message="Prompt is required"),
            Length(min=1, max=10000, message="Prompt must be between 1 and 10,000 characters")
        ],
        render_kw={
            'placeholder': 'Enter your query here...',
            'rows': 4,
            'cols': 50
        }
    )
    
    model = SelectField(
        'Model',
        choices=[
            ('gpt-4o-mini', 'GPT-4o Mini'),
            ('gpt-4o', 'GPT-4o'),
            ('gpt-3.5-turbo', 'GPT-3.5 Turbo'),
            ('gemini-pro', 'Gemini Pro'),
            ('gemini-1.5-pro', 'Gemini 1.5 Pro'),
            ('deepseek-chat', 'DeepSeek Chat'),
            ('deepseek-coder', 'DeepSeek Coder'),
            ('mistral-large', 'Mistral Large'),
            ('mistral-medium', 'Mistral Medium'),
            ('mistral-small', 'Mistral Small')
        ],
        default='gpt-4o-mini',
        validators=[DataRequired(message="Model selection is required")]
    )
    
    session_id = StringField(
        'Session ID',
        validators=[
            Optional(),
            Length(max=100, message="Session ID must not exceed 100 characters"),
            Regexp(
                r'^[a-zA-Z0-9_-]*$',
                message="Session ID can only contain letters, numbers, underscores, and hyphens"
            )
        ],
        render_kw={
            'placeholder': 'Optional: session-name-123'
        }
    )
    
    temperature = FloatField(
        'Temperature',
        validators=[
            Optional(),
            NumberRange(min=0.0, max=2.0, message="Temperature must be between 0.0 and 2.0")
        ],
        default=0.7,
        render_kw={
            'step': '0.1',
            'min': '0.0',
            'max': '2.0'
        }
    )
    
    max_tokens = IntegerField(
        'Max Tokens',
        validators=[
            Optional(),
            NumberRange(min=1, max=4096, message="Max tokens must be between 1 and 4096")
        ],
        default=1000,
        render_kw={
            'min': '1',
            'max': '4096'
        }
    )
    
    system_prompt = TextAreaField(
        'System Prompt',
        validators=[
            Optional(),
            Length(max=2000, message="System prompt must not exceed 2000 characters")
        ],
        render_kw={
            'placeholder': 'Optional: Custom system instructions...',
            'rows': 2,
            'cols': 50
        }
    )


class ConversationForm(FlaskForm):
    """
    Form for conversation history requests with CSRF protection.
    
    Provides secure form handling for retrieving and managing
    conversation history and session data.
    """
    
    session_id = StringField(
        'Session ID',
        validators=[
            DataRequired(message="Session ID is required"),
            Length(min=1, max=100, message="Session ID must be between 1 and 100 characters"),
            Regexp(
                r'^[a-zA-Z0-9_-]+$',
                message="Session ID can only contain letters, numbers, underscores, and hyphens"
            )
        ],
        render_kw={
            'placeholder': 'session-name-123'
        }
    )
    
    limit = IntegerField(
        'Limit',
        validators=[
            Optional(),
            NumberRange(min=1, max=100, message="Limit must be between 1 and 100")
        ],
        default=20,
        render_kw={
            'min': '1',
            'max': '100'
        }
    )
    
    offset = IntegerField(
        'Offset',
        validators=[
            Optional(),
            NumberRange(min=0, message="Offset must be non-negative")
        ],
        default=0,
        render_kw={
            'min': '0'
        }
    )


class SessionForm(FlaskForm):
    """
    Form for session management with CSRF protection.
    
    Provides secure form handling for creating, updating,
    and managing conversation sessions.
    """
    
    session_id = StringField(
        'Session ID',
        validators=[
            Optional(),
            Length(min=1, max=100, message="Session ID must be between 1 and 100 characters"),
            Regexp(
                r'^[a-zA-Z0-9_-]+$',
                message="Session ID can only contain letters, numbers, underscores, and hyphens"
            )
        ],
        render_kw={
            'placeholder': 'Leave empty for auto-generation'
        }
    )
    
    name = StringField(
        'Session Name',
        validators=[
            Optional(),
            Length(max=200, message="Session name must not exceed 200 characters")
        ],
        render_kw={
            'placeholder': 'My Conversation Session'
        }
    )
    
    description = TextAreaField(
        'Description',
        validators=[
            Optional(),
            Length(max=1000, message="Description must not exceed 1000 characters")
        ],
        render_kw={
            'placeholder': 'Optional description of this session...',
            'rows': 3,
            'cols': 50
        }
    )