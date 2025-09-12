"""
Tests for improved error handling system.
"""

import pytest
import time

from yamllm.core.exceptions import (
    YAMLLMException, ProviderError, ToolExecutionError,
    NetworkError, RateLimitError, AuthenticationError
)
from yamllm.core.error_handler import ErrorHandler, CircuitBreaker


class TestEnhancedExceptions:
    """Test enhanced exception classes."""
    
    def test_base_exception_details(self):
        """Test base exception with details."""
        details = {"key": "value", "number": 42}
        exc = YAMLLMException("Test error", details)
        
        assert str(exc) == "Test error"
        assert exc.details == details
        assert exc.to_dict() == {
            "error_type": "YAMLLMException",
            "message": "Test error",
            "details": details
        }
    
    def test_provider_error_with_traceback(self):
        """Test provider error captures traceback."""
        try:
            raise ValueError("Original error")
        except ValueError as e:
            error = ProviderError(
                provider="test_provider",
                message="Provider failed",
                original_error=e,
                request_params={"model": "test"}
            )
            
            assert error.provider == "test_provider"
            assert error.original_error == e
            assert "traceback" in error.details
            assert len(error.details["traceback"]) > 0
    
    def test_tool_execution_error_details(self):
        """Test tool execution error with full context."""
        error = ToolExecutionError(
            tool_name="calculator",
            message="Division by zero",
            tool_args={"a": 10, "b": 0},
            execution_time=1.5
        )
        
        assert error.tool_name == "calculator"
        assert error.details["tool_args"] == {"a": 10, "b": 0}
        assert error.details["execution_time"] == 1.5
    
    def test_rate_limit_error(self):
        """Test rate limit error with retry info."""
        error = RateLimitError(
            message="Too many requests",
            retry_after=60,
            limit_type="api_calls"
        )
        
        assert error.details["retry_after"] == 60
        assert error.details["limit_type"] == "api_calls"


class TestErrorHandler:
    """Test error handler utilities."""
    
    def test_retry_decorator_success(self):
        """Test retry decorator with eventual success."""
        handler = ErrorHandler()
        attempt_count = 0
        
        @handler.with_retry(max_attempts=3, initial_delay=0.1)
        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise NetworkError("Connection failed")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert attempt_count == 2
    
    def test_retry_decorator_all_failures(self):
        """Test retry decorator when all attempts fail."""
        handler = ErrorHandler()
        
        @handler.with_retry(max_attempts=2, initial_delay=0.1)
        def always_fails():
            raise NetworkError("Connection failed")
        
        with pytest.raises(NetworkError):
            always_fails()
    
    def test_retry_with_callback(self):
        """Test retry decorator with callback."""
        handler = ErrorHandler()
        retry_calls = []
        
        def on_retry(exc, attempt):
            retry_calls.append((type(exc).__name__, attempt))
        
        @handler.with_retry(
            max_attempts=3, 
            initial_delay=0.1,
            on_retry=on_retry
        )
        def flaky_function():
            raise NetworkError("Failed")
        
        with pytest.raises(NetworkError):
            flaky_function()
        
        assert len(retry_calls) == 2  # Called on retry, not on final failure
        assert retry_calls[0] == ("NetworkError", 1)
        assert retry_calls[1] == ("NetworkError", 2)
    
    def test_fallback_decorator_with_value(self):
        """Test fallback decorator with static value."""
        handler = ErrorHandler()
        
        @handler.with_fallback(fallback_value="default")
        def may_fail(should_fail=False):
            if should_fail:
                raise ValueError("Failed")
            return "success"
        
        assert may_fail(should_fail=False) == "success"
        assert may_fail(should_fail=True) == "default"
    
    def test_fallback_decorator_with_function(self):
        """Test fallback decorator with fallback function."""
        handler = ErrorHandler()
        
        def fallback_func(x):
            return f"fallback_{x}"
        
        @handler.with_fallback(fallback_func=fallback_func)
        def may_fail(x):
            if x == "fail":
                raise ValueError("Failed")
            return f"success_{x}"
        
        assert may_fail("ok") == "success_ok"
        assert may_fail("fail") == "fallback_fail"
    
    def test_handle_provider_error_rate_limit(self):
        """Test provider error handling for rate limits."""
        handler = ErrorHandler()
        original = Exception("Rate limit exceeded. Retry after 60 seconds.")
        
        with pytest.raises(RateLimitError) as exc_info:
            handler.handle_provider_error(original, "openai")
        
        assert exc_info.value.details["retry_after"] == 60
    
    def test_handle_provider_error_auth(self):
        """Test provider error handling for auth errors."""
        handler = ErrorHandler()
        original = Exception("Unauthorized: Invalid API key")
        
        with pytest.raises(AuthenticationError) as exc_info:
            handler.handle_provider_error(original, "openai")
        
        assert exc_info.value.details["provider"] == "openai"
    
    def test_safe_execute(self):
        """Test safe execute utility."""
        def risky_function(x):
            if x == 0:
                raise ValueError("Cannot be zero")
            return 10 / x
        
        # Success case
        result = ErrorHandler.safe_execute(risky_function, 5)
        assert result == 2.0
        
        # Failure case with default
        result = ErrorHandler.safe_execute(
            risky_function, 0, default=-1
        )
        assert result == -1


class TestCircuitBreaker:
    """Test circuit breaker pattern."""
    
    def test_circuit_breaker_normal_operation(self):
        """Test circuit breaker in normal operation."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        
        @breaker
        def protected_function(should_fail=False):
            if should_fail:
                raise NetworkError("Failed")
            return "success"
        
        # Normal operation
        assert protected_function() == "success"
        assert protected_function() == "success"
    
    def test_circuit_breaker_opens(self):
        """Test circuit breaker opens after threshold."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.5)
        
        @breaker
        def protected_function():
            raise NetworkError("Failed")
        
        # First attempts fail normally
        with pytest.raises(NetworkError):
            protected_function()
        
        with pytest.raises(NetworkError):
            protected_function()
        
        # Circuit should now be open
        with pytest.raises(NetworkError) as exc_info:
            protected_function()
        assert "Circuit breaker is open" in str(exc_info.value)
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        success_flag = False
        
        @breaker
        def protected_function():
            if not success_flag:
                raise NetworkError("Failed")
            return "success"
        
        # Trigger circuit breaker
        for _ in range(2):
            with pytest.raises(NetworkError):
                protected_function()
        
        # Circuit is open
        with pytest.raises(NetworkError) as exc_info:
            protected_function()
        assert "Circuit breaker is open" in str(exc_info.value)
        
        # Wait for recovery timeout
        time.sleep(0.2)
        success_flag = True
        
        # Should work now (half-open -> closed)
        assert protected_function() == "success"
        
        # Circuit should be closed
        assert protected_function() == "success"