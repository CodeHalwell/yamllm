"""
Minimal working tests for YAMLLM web security features.

This module demonstrates the Flask web API security features
without importing the problematic main package dependencies.
"""

import sys
import os
import tempfile
import pytest
from unittest.mock import patch

# Add the project directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import only the web components directly
from yamllm.web.schemas import QuerySchema, ConversationSchema, MessageSchema
from yamllm.web.forms import QueryForm
import marshmallow


class TestInputValidation:
    """Test Marshmallow schema validation for security."""
    
    def test_query_schema_basic_validation(self):
        """Test basic query schema validation."""
        schema = QuerySchema()
        
        # Test valid data
        valid_data = {
            'prompt': 'What is machine learning?',
            'model': 'gpt-4o-mini',
            'session_id': 'test-session-123'
        }
        result = schema.load(valid_data)
        assert result['prompt'] == 'What is machine learning?'
        assert result['model'] == 'gpt-4o-mini'
        assert result['session_id'] == 'test-session-123'
    
    def test_query_schema_prompt_validation(self):
        """Test prompt validation rules."""
        schema = QuerySchema()
        
        # Test empty prompt
        with pytest.raises(marshmallow.ValidationError) as exc_info:
            schema.load({'prompt': ''})
        assert 'Prompt must be between 1 and 10,000 characters' in str(exc_info.value)
        
        # Test too long prompt
        long_prompt = 'A' * 10001
        with pytest.raises(marshmallow.ValidationError) as exc_info:
            schema.load({'prompt': long_prompt})
        assert 'Prompt must be between 1 and 10,000 characters' in str(exc_info.value)
    
    def test_query_schema_security_validation(self):
        """Test security validation against harmful content."""
        schema = QuerySchema()
        
        # Test SQL injection attempt
        with pytest.raises(marshmallow.ValidationError) as exc_info:
            schema.load({'prompt': "'; DROP TABLE messages; --"})
        assert 'harmful content' in str(exc_info.value).lower()
        
        # Test XSS attempt
        with pytest.raises(marshmallow.ValidationError) as exc_info:
            schema.load({'prompt': "<script>alert('xss')</script>"})
        assert 'harmful content' in str(exc_info.value).lower()
        
        # Test code execution attempt
        with pytest.raises(marshmallow.ValidationError) as exc_info:
            schema.load({'prompt': "exec('import os; os.system(\"rm -rf /\")')"})
        assert 'harmful content' in str(exc_info.value).lower()
    
    def test_query_schema_model_validation(self):
        """Test model selection validation."""
        schema = QuerySchema()
        
        # Test invalid model
        with pytest.raises(marshmallow.ValidationError) as exc_info:
            schema.load({'prompt': 'test', 'model': 'invalid-model'})
        assert 'Invalid model selection' in str(exc_info.value)
        
        # Test valid models
        valid_models = ['gpt-4o-mini', 'gemini-pro', 'deepseek-chat', 'mistral-large']
        for model in valid_models:
            result = schema.load({'prompt': 'test prompt', 'model': model})
            assert result['model'] == model
    
    def test_query_schema_session_id_validation(self):
        """Test session ID validation rules."""
        schema = QuerySchema()
        
        # Test invalid characters
        with pytest.raises(marshmallow.ValidationError) as exc_info:
            schema.load({'prompt': 'test', 'session_id': 'invalid session id!'})
        assert 'can only contain letters, numbers, underscores, and hyphens' in str(exc_info.value)
        
        # Test too long session ID
        long_session = 'a' * 101
        with pytest.raises(marshmallow.ValidationError) as exc_info:
            schema.load({'prompt': 'test', 'session_id': long_session})
        assert 'must not exceed 100 characters' in str(exc_info.value)
        
        # Test valid session IDs
        valid_sessions = ['session-123', 'test_session', 'user123-session']
        for session in valid_sessions:
            result = schema.load({'prompt': 'test', 'session_id': session})
            assert result['session_id'] == session
    
    def test_query_schema_parameter_validation(self):
        """Test parameter validation for temperature and max_tokens."""
        schema = QuerySchema()
        
        # Test temperature validation
        with pytest.raises(marshmallow.ValidationError) as exc_info:
            schema.load({'prompt': 'test', 'temperature': 3.0})
        assert 'Temperature must be between 0.0 and 2.0' in str(exc_info.value)
        
        # Test max_tokens validation
        with pytest.raises(marshmallow.ValidationError) as exc_info:
            schema.load({'prompt': 'test', 'max_tokens': 5000})
        assert 'Max tokens must be between 1 and 4096' in str(exc_info.value)
        
        # Test valid parameters
        result = schema.load({
            'prompt': 'test',
            'temperature': 0.8,
            'max_tokens': 2000
        })
        assert result['temperature'] == 0.8
        assert result['max_tokens'] == 2000


class TestConversationValidation:
    """Test conversation schema validation."""
    
    def test_conversation_schema_validation(self):
        """Test conversation schema validation."""
        schema = ConversationSchema()
        
        # Test valid data
        valid_data = {
            'session_id': 'test-session-123',
            'limit': 20,
            'offset': 0
        }
        result = schema.load(valid_data)
        assert result['session_id'] == 'test-session-123'
        assert result['limit'] == 20
        assert result['offset'] == 0
    
    def test_conversation_schema_session_validation(self):
        """Test session ID validation in conversation schema."""
        schema = ConversationSchema()
        
        # Test missing session ID
        with pytest.raises(marshmallow.ValidationError) as exc_info:
            schema.load({})
        assert 'Session ID is required' in str(exc_info.value)
        
        # Test invalid session ID
        with pytest.raises(marshmallow.ValidationError) as exc_info:
            schema.load({'session_id': 'invalid session!'})
        assert 'can only contain letters, numbers, underscores, and hyphens' in str(exc_info.value)
    
    def test_conversation_schema_pagination(self):
        """Test pagination parameter validation."""
        schema = ConversationSchema()
        
        # Test invalid limit
        with pytest.raises(marshmallow.ValidationError) as exc_info:
            schema.load({'session_id': 'test', 'limit': 101})
        assert 'Limit must be between 1 and 100' in str(exc_info.value)
        
        # Test invalid offset
        with pytest.raises(marshmallow.ValidationError) as exc_info:
            schema.load({'session_id': 'test', 'offset': -1})
        assert 'Offset must be non-negative' in str(exc_info.value)


class TestMessageValidation:
    """Test message schema validation."""
    
    def test_message_schema_validation(self):
        """Test message schema validation."""
        schema = MessageSchema()
        
        # Test valid message
        valid_data = {
            'role': 'user',
            'content': 'Hello, how are you?',
            'name': 'user123'
        }
        result = schema.load(valid_data)
        assert result['role'] == 'user'
        assert result['content'] == 'Hello, how are you?'
        assert result['name'] == 'user123'
    
    def test_message_role_validation(self):
        """Test message role validation."""
        schema = MessageSchema()
        
        # Test invalid role
        with pytest.raises(marshmallow.ValidationError) as exc_info:
            schema.load({'role': 'invalid', 'content': 'test'})
        assert 'Role must be one of: system, user, assistant, tool' in str(exc_info.value)
        
        # Test valid roles
        valid_roles = ['system', 'user', 'assistant', 'tool']
        for role in valid_roles:
            result = schema.load({'role': role, 'content': 'test content'})
            assert result['role'] == role
    
    def test_message_content_validation(self):
        """Test message content validation."""
        schema = MessageSchema()
        
        # Test empty content
        with pytest.raises(marshmallow.ValidationError) as exc_info:
            schema.load({'role': 'user', 'content': ''})
        assert 'Content must be between 1 and 50,000 characters' in str(exc_info.value)
        
        # Test too long content
        long_content = 'A' * 50001
        with pytest.raises(marshmallow.ValidationError) as exc_info:
            schema.load({'role': 'user', 'content': long_content})
        assert 'Content must be between 1 and 50,000 characters' in str(exc_info.value)


class TestFormValidation:
    """Test Flask-WTF form validation."""
    
    @patch('flask_wtf.FlaskForm.__init__')
    def test_query_form_structure(self, mock_init):
        """Test that QueryForm has the correct structure."""
        mock_init.return_value = None
        
        # Import should work without Flask app context for structure testing
        form = QueryForm.__new__(QueryForm)
        
        # Check that form has expected fields
        assert hasattr(QueryForm, 'prompt')
        assert hasattr(QueryForm, 'model')
        assert hasattr(QueryForm, 'session_id')
        assert hasattr(QueryForm, 'temperature')
        assert hasattr(QueryForm, 'max_tokens')
        assert hasattr(QueryForm, 'system_prompt')


class TestSecurityConfiguration:
    """Test security configuration and constants."""
    
    def test_harmful_content_patterns(self):
        """Test that harmful content patterns are properly defined."""
        # These patterns should be caught by the schema validation
        harmful_patterns = [
            'DROP TABLE', 'DELETE FROM', 'UPDATE SET',
            '<script>', '</script>', 'javascript:',
            'eval(', 'exec(', 'system('
        ]
        
        schema = QuerySchema()
        for pattern in harmful_patterns:
            with pytest.raises(marshmallow.ValidationError) as exc_info:
                schema.load({'prompt': f'Test {pattern} malicious content'})
            assert 'harmful content' in str(exc_info.value).lower()
    
    def test_model_whitelist(self):
        """Test that model whitelist is properly enforced."""
        schema = QuerySchema()
        
        # Valid models should pass
        valid_models = [
            'gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo',
            'gemini-pro', 'gemini-1.5-pro',
            'deepseek-chat', 'deepseek-coder',
            'mistral-large', 'mistral-medium', 'mistral-small'
        ]
        
        for model in valid_models:
            result = schema.load({'prompt': 'test', 'model': model})
            assert result['model'] == model
        
        # Invalid models should fail
        invalid_models = ['unknown-model', 'malicious-model', '']
        for model in invalid_models:
            with pytest.raises(marshmallow.ValidationError):
                schema.load({'prompt': 'test', 'model': model})


if __name__ == '__main__':
    pytest.main([__file__, '-v'])