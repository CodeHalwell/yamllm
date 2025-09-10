"""
Flask application for YAMLLM web API.

This module provides a Flask-based web API with CSRF protection,
input validation, and endpoints for LLM interactions.
"""

import os
from flask import Flask, request, jsonify
from flask_wtf.csrf import CSRFProtect
from marshmallow import ValidationError

from .schemas import QuerySchema, ConversationSchema
from .forms import QueryForm


def create_app(config=None):
    """
    Create and configure the Flask application.
    
    Args:
        config (dict, optional): Configuration dictionary for the app.
        
    Returns:
        Flask: Configured Flask application instance.
    """
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = config.get('SECRET_KEY', os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')) if config else os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['WTF_CSRF_TIME_LIMIT'] = config.get('CSRF_TIME_LIMIT', 3600) if config else 3600
    app.config['WTF_CSRF_SSL_STRICT'] = config.get('CSRF_SSL_STRICT', False) if config else False
    
    # Initialize CSRF protection
    csrf = CSRFProtect(app)
    
    # Initialize schemas for validation
    query_schema = QuerySchema()
    conversation_schema = ConversationSchema()
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """
        Health check endpoint.
        
        Returns:
            dict: Health status response.
        """
        return jsonify({'status': 'healthy', 'service': 'yamllm-api'})
    
    @app.route('/api/query', methods=['POST'])
    def api_query():
        """
        API endpoint for LLM queries with JSON input.
        
        Returns:
            dict: Query response or error.
        """
        try:
            # Validate input using Marshmallow schema
            data = query_schema.load(request.get_json() or {})
            
            # TODO: Integrate with actual LLM processing
            # For now, return a mock response
            response = {
                'response': f"Mock response to: {data['prompt']}",
                'model': data.get('model', 'default'),
                'session_id': data.get('session_id', 'default')
            }
            
            return jsonify(response)
            
        except ValidationError as err:
            return jsonify({'error': 'Validation failed', 'details': err.messages}), 400
        except Exception as e:
            return jsonify({'error': 'Internal server error', 'details': str(e)}), 500
    
    @app.route('/api/conversation/<session_id>', methods=['GET'])
    def get_conversation(session_id):
        """
        Get conversation history for a session.
        
        Args:
            session_id (str): Session identifier.
            
        Returns:
            dict: Conversation history.
        """
        try:
            # Validate session_id
            if not session_id or len(session_id) > 100:
                return jsonify({'error': 'Invalid session_id'}), 400
            
            # TODO: Initialize conversation store with proper path
            # For now, return mock data
            conversation = {
                'session_id': session_id,
                'messages': [
                    {'role': 'user', 'content': 'Hello'},
                    {'role': 'assistant', 'content': 'Hi there!'}
                ]
            }
            
            return jsonify(conversation)
            
        except Exception as e:
            return jsonify({'error': 'Internal server error', 'details': str(e)}), 500
    
    @app.route('/query', methods=['GET', 'POST'])
    def web_query():
        """
        Web form endpoint for LLM queries with CSRF protection.
        
        Returns:
            str: HTML response or JSON for AJAX requests.
        """
        form = QueryForm()
        
        if form.validate_on_submit():
            try:
                # Process the query
                prompt = form.prompt.data
                model = form.model.data
                session_id = form.session_id.data or 'default'
                
                # TODO: Integrate with actual LLM processing
                response = f"Mock response to: {prompt}"
                
                if request.is_json or request.headers.get('Accept') == 'application/json':
                    return jsonify({
                        'response': response,
                        'model': model,
                        'session_id': session_id
                    })
                else:
                    # Return HTML response for regular form submission
                    return f"""
                    <html>
                    <head><title>YAMLLM Query Result</title></head>
                    <body>
                        <h1>Query Result</h1>
                        <p><strong>Prompt:</strong> {prompt}</p>
                        <p><strong>Response:</strong> {response}</p>
                        <p><strong>Model:</strong> {model}</p>
                        <p><strong>Session:</strong> {session_id}</p>
                        <a href="/query">Submit another query</a>
                    </body>
                    </html>
                    """
                    
            except Exception as e:
                if request.is_json or request.headers.get('Accept') == 'application/json':
                    return jsonify({'error': 'Processing failed', 'details': str(e)}), 500
                else:
                    return f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>", 500
        
        # Show form for GET requests or validation errors
        if request.is_json:
            return jsonify({
                'error': 'Validation failed',
                'details': form.errors
            }), 400
        
        # Return HTML form
        csrf_token = form.csrf_token._value() if form.csrf_token else ''
        return f"""
        <html>
        <head><title>YAMLLM Query</title></head>
        <body>
            <h1>YAMLLM Query Interface</h1>
            <form method="POST">
                <input type="hidden" name="csrf_token" value="{csrf_token}"/>
                <p>
                    <label for="prompt">Prompt:</label><br>
                    <textarea name="prompt" rows="4" cols="50" required>{form.prompt.data or ''}</textarea>
                </p>
                <p>
                    <label for="model">Model:</label><br>
                    <select name="model">
                        <option value="gpt-4o-mini">GPT-4o Mini</option>
                        <option value="gemini-pro">Gemini Pro</option>
                        <option value="deepseek-chat">DeepSeek Chat</option>
                        <option value="mistral-large">Mistral Large</option>
                    </select>
                </p>
                <p>
                    <label for="session_id">Session ID (optional):</label><br>
                    <input type="text" name="session_id" value="{form.session_id.data or ''}">
                </p>
                <p>
                    <input type="submit" value="Submit Query">
                </p>
            </form>
            {form.errors if form.errors else ''}
        </body>
        </html>
        """
    
    @app.errorhandler(400)
    def bad_request(error):
        """Handle 400 Bad Request errors."""
        return jsonify({'error': 'Bad Request', 'details': str(error)}), 400
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 Not Found errors."""
        return jsonify({'error': 'Not Found', 'details': 'The requested resource was not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 Internal Server Error."""
        return jsonify({'error': 'Internal Server Error', 'details': 'An unexpected error occurred'}), 500
    
    return app


if __name__ == '__main__':
    # Development server
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)