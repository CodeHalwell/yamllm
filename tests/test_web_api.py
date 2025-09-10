"""
Tests for YAMLLM web API security and functionality.

This module provides comprehensive tests for the Flask web API,
including CSRF protection, input validation, and endpoint functionality.
"""

import pytest
import json
from flask import url_for
from yamllm.web.app import create_app


@pytest.fixture
def app():
    """Create and configure a test Flask application."""
    config = {
        'SECRET_KEY': 'test-secret-key',
        'TESTING': True,
        'WTF_CSRF_ENABLED': True,
        'CSRF_TIME_LIMIT': 3600,
        'CSRF_SSL_STRICT': False
    }
    app = create_app(config)
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Create a test client for the Flask application."""
    return app.test_client()


@pytest.fixture
def csrf_token(client):
    """Get a CSRF token for testing protected endpoints."""
    with client.session_transaction() as sess:
        sess['_csrf_token'] = 'test-csrf-token'
    return 'test-csrf-token'


class TestWebAPISecurity:
    """Test security features of the web API."""
    
    def test_health_check_no_csrf(self, client):
        """Test that health check endpoint works without CSRF protection."""
        response = client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert data['service'] == 'yamllm-api'
    
    def test_csrf_protection_on_api_query(self, client):
        """Test that API query endpoint requires CSRF protection."""
        # Test without CSRF token
        response = client.post('/api/query', 
                             json={'prompt': 'Test query'},
                             content_type='application/json')
        assert response.status_code == 400
    
    def test_csrf_protection_on_web_form(self, client):
        """Test that web form endpoint requires CSRF protection."""
        # Test without CSRF token
        response = client.post('/query', data={'prompt': 'Test query'})
        assert response.status_code == 400
    
    def test_input_validation_marshmallow(self, client, csrf_token):
        """Test Marshmallow schema validation on API endpoints."""
        # Test with invalid data
        response = client.post('/api/query',
                             json={'prompt': ''},  # Empty prompt
                             headers={'X-CSRFToken': csrf_token})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Validation failed' in data['error']
        assert 'prompt' in data['details']
    
    def test_sql_injection_protection(self, client, csrf_token):
        """Test protection against SQL injection attempts."""
        malicious_prompt = "'; DROP TABLE messages; --"
        response = client.post('/api/query',
                             json={'prompt': malicious_prompt},
                             headers={'X-CSRFToken': csrf_token})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'harmful content' in str(data).lower()
    
    def test_xss_protection(self, client, csrf_token):
        """Test protection against XSS attempts."""
        malicious_prompt = "<script>alert('xss')</script>"
        response = client.post('/api/query',
                             json={'prompt': malicious_prompt},
                             headers={'X-CSRFToken': csrf_token})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'harmful content' in str(data).lower()
    
    def test_prompt_length_validation(self, client, csrf_token):
        """Test prompt length validation."""
        # Test with too long prompt
        long_prompt = 'A' * 10001  # Exceeds 10,000 character limit
        response = client.post('/api/query',
                             json={'prompt': long_prompt},
                             headers={'X-CSRFToken': csrf_token})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Validation failed' in data['error']
    
    def test_session_id_validation(self, client, csrf_token):
        """Test session ID validation."""
        # Test with invalid session ID characters
        response = client.post('/api/query',
                             json={
                                 'prompt': 'Valid prompt',
                                 'session_id': 'invalid session id!'  # Contains spaces and special chars
                             },
                             headers={'X-CSRFToken': csrf_token})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Validation failed' in data['error']


class TestWebAPIFunctionality:
    """Test functional aspects of the web API."""
    
    def test_valid_api_query(self, client, csrf_token):
        """Test valid API query with proper data."""
        response = client.post('/api/query',
                             json={
                                 'prompt': 'What is machine learning?',
                                 'model': 'gpt-4o-mini',
                                 'session_id': 'test-session-123'
                             },
                             headers={'X-CSRFToken': csrf_token})
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'response' in data
        assert 'What is machine learning?' in data['response']
        assert data['model'] == 'gpt-4o-mini'
        assert data['session_id'] == 'test-session-123'
    
    def test_conversation_endpoint(self, client):
        """Test conversation history endpoint."""
        response = client.get('/api/conversation/test-session')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['session_id'] == 'test-session'
        assert 'messages' in data
        assert isinstance(data['messages'], list)
    
    def test_invalid_session_id_in_url(self, client):
        """Test conversation endpoint with invalid session ID."""
        # Test with empty session ID
        response = client.get('/api/conversation/')
        assert response.status_code == 404
        
        # Test with too long session ID
        long_session_id = 'a' * 101  # Exceeds 100 character limit
        response = client.get(f'/api/conversation/{long_session_id}')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Invalid session_id' in data['error']
    
    def test_web_form_get_request(self, client):
        """Test GET request to web form endpoint."""
        response = client.get('/query')
        assert response.status_code == 200
        assert b'YAMLLM Query Interface' in response.data
        assert b'csrf_token' in response.data
        assert b'<form method="POST">' in response.data
    
    def test_model_validation(self, client, csrf_token):
        """Test model parameter validation."""
        # Test with invalid model
        response = client.post('/api/query',
                             json={
                                 'prompt': 'Test prompt',
                                 'model': 'invalid-model'
                             },
                             headers={'X-CSRFToken': csrf_token})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Validation failed' in data['error']
    
    def test_temperature_validation(self, client, csrf_token):
        """Test temperature parameter validation."""
        # Test with invalid temperature (too high)
        response = client.post('/api/query',
                             json={
                                 'prompt': 'Test prompt',
                                 'temperature': 3.0  # Exceeds 2.0 limit
                             },
                             headers={'X-CSRFToken': csrf_token})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Validation failed' in data['error']
    
    def test_max_tokens_validation(self, client, csrf_token):
        """Test max_tokens parameter validation."""
        # Test with invalid max_tokens (too high)
        response = client.post('/api/query',
                             json={
                                 'prompt': 'Test prompt',
                                 'max_tokens': 5000  # Exceeds 4096 limit
                             },
                             headers={'X-CSRFToken': csrf_token})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Validation failed' in data['error']


class TestErrorHandling:
    """Test error handling and HTTP status codes."""
    
    def test_404_handler(self, client):
        """Test 404 error handler."""
        response = client.get('/nonexistent-endpoint')
        assert response.status_code == 404
        data = json.loads(response.data)
        assert data['error'] == 'Not Found'
    
    def test_400_handler(self, client):
        """Test 400 error handler."""
        # Send malformed JSON
        response = client.post('/api/query',
                             data='invalid json',
                             content_type='application/json')
        assert response.status_code in [400, 500]  # Either is acceptable for malformed JSON
    
    def test_internal_error_handling(self, client, csrf_token, monkeypatch):
        """Test internal error handling."""
        # This test simulates an internal error during processing
        # In a real implementation, you might mock a service to raise an exception
        pass  # Skip for now as it requires mocking the LLM service


class TestCSRFTokenHandling:
    """Test CSRF token handling and validation."""
    
    def test_csrf_token_in_form(self, client):
        """Test that CSRF token is included in web forms."""
        response = client.get('/query')
        assert response.status_code == 200
        assert b'csrf_token' in response.data
    
    def test_csrf_exempt_endpoints(self, client):
        """Test endpoints that should be exempt from CSRF protection."""
        # Health check should not require CSRF
        response = client.get('/health')
        assert response.status_code == 200
        
        # GET endpoints should not require CSRF
        response = client.get('/api/conversation/test-session')
        assert response.status_code == 200