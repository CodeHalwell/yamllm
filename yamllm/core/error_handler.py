"""
Error handling utilities for YAMLLM.

This module provides consistent error handling and recovery strategies
across the YAMLLM codebase.
"""

import functools
import logging
from typing import Callable, TypeVar, Optional, Dict, Any, Type
import time

from yamllm.core.exceptions import (
    ProviderError, ToolExecutionError,
    NetworkError, RateLimitError, TimeoutError
)


T = TypeVar('T')


class ErrorHandler:
    """
    Centralized error handling with retry logic and fallback strategies.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def with_retry(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        exceptions: tuple = (NetworkError, TimeoutError),
        on_retry: Optional[Callable[[Exception, int], None]] = None
    ):
        """
        Decorator for automatic retry with exponential backoff.
        
        Args:
            max_attempts: Maximum number of attempts
            initial_delay: Initial delay between retries in seconds
            backoff_factor: Factor to multiply delay by after each retry
            exceptions: Tuple of exception types to retry on
            on_retry: Optional callback called on each retry
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> T:
                last_exception = None
                delay = initial_delay
                
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        
                        if attempt == max_attempts - 1:
                            # Last attempt, re-raise
                            raise
                        
                        # Log retry
                        self.logger.warning(
                            f"Retry {attempt + 1}/{max_attempts} for {func.__name__} "
                            f"after {type(e).__name__}: {str(e)}"
                        )
                        
                        # Call retry callback if provided
                        if on_retry:
                            on_retry(e, attempt + 1)
                        
                        # Wait before retry
                        time.sleep(delay)
                        delay *= backoff_factor
                
                # Should never reach here, but just in case
                if last_exception:
                    raise last_exception
                    
            return wrapper
        return decorator
    
    def with_fallback(
        self,
        fallback_value: Any = None,
        fallback_func: Optional[Callable[..., T]] = None,
        exceptions: tuple = (Exception,),
        log_level: str = "error"
    ):
        """
        Decorator to provide fallback behavior on exceptions.
        
        Args:
            fallback_value: Static fallback value
            fallback_func: Function to call for fallback (receives original args)
            exceptions: Tuple of exception types to catch
            log_level: Logging level for caught exceptions
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> T:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    # Log the error
                    log_method = getattr(self.logger, log_level, self.logger.error)
                    log_method(
                        f"Fallback triggered for {func.__name__}: "
                        f"{type(e).__name__}: {str(e)}"
                    )
                    
                    # Return fallback
                    if fallback_func:
                        return fallback_func(*args, **kwargs)
                    return fallback_value
                    
            return wrapper
        return decorator
    
    def handle_provider_error(
        self, error: Exception, provider: str, 
        request_params: Optional[Dict[str, Any]] = None
    ) -> ProviderError:
        """
        Convert provider-specific exceptions to ProviderError.
        
        Args:
            error: Original exception
            provider: Provider name
            request_params: Request parameters for debugging
            
        Returns:
            ProviderError with full context
        """
        # Check for specific error types
        error_message = str(error)
        
        if "rate limit" in error_message.lower():
            # Extract retry-after if available
            retry_after = self._extract_retry_after(error_message)
            raise RateLimitError(
                f"Rate limit exceeded for {provider}",
                retry_after=retry_after,
                limit_type="api_calls"
            )
        
        if "unauthorized" in error_message.lower() or "authentication" in error_message.lower():
            from yamllm.core.exceptions import AuthenticationError
            raise AuthenticationError(
                f"Authentication failed for {provider}",
                provider=provider
            )
        
        if "timeout" in error_message.lower():
            raise TimeoutError(
                f"Request timeout for {provider}",
                operation="api_call"
            )
        
        # Generic provider error
        provider_error = ProviderError(
            provider=provider,
            message=error_message,
            original_error=error,
            request_params=request_params
        )
        provider_error.log_error()
        return provider_error
    
    def handle_tool_error(
        self, error: Exception, tool_name: str,
        tool_args: Optional[Dict[str, Any]] = None,
        execution_time: Optional[float] = None
    ) -> ToolExecutionError:
        """
        Convert tool exceptions to ToolExecutionError.
        
        Args:
            error: Original exception
            tool_name: Name of the tool
            tool_args: Tool arguments
            execution_time: Execution time before error
            
        Returns:
            ToolExecutionError with full context
        """
        tool_error = ToolExecutionError(
            tool_name=tool_name,
            message=str(error),
            original_error=error,
            tool_args=tool_args,
            execution_time=execution_time
        )
        tool_error.log_error()
        return tool_error
    
    def _extract_retry_after(self, error_message: str) -> Optional[int]:
        """Extract retry-after value from error message."""
        import re
        
        # Common patterns for retry-after
        patterns = [
            r"retry after (\d+) seconds",
            r"retry-after: (\d+)",
            r"wait (\d+) seconds"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_message, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None
    
    @staticmethod
    def safe_execute(
        func: Callable[..., T],
        *args,
        default: Any = None,
        exceptions: tuple = (Exception,),
        logger: Optional[logging.Logger] = None,
        **kwargs
    ) -> T:
        """
        Safely execute a function with exception handling.
        
        Args:
            func: Function to execute
            *args: Positional arguments for func
            default: Default value on exception
            exceptions: Exceptions to catch
            logger: Logger for error messages
            **kwargs: Keyword arguments for func
            
        Returns:
            Function result or default value
        """
        try:
            return func(*args, **kwargs)
        except exceptions as e:
            if logger:
                logger.error(f"Error in {func.__name__}: {type(e).__name__}: {str(e)}")
            return default


class CircuitBreaker:
    """
    Circuit breaker pattern for handling repeated failures.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            expected_exception: Exception type to track
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self._failure_count = 0
        self._last_failure_time = None
        self._state = "closed"  # closed, open, half-open
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for circuit breaker pattern."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if self._state == "open":
                # Check if we should try recovery
                if (time.time() - self._last_failure_time) > self.recovery_timeout:
                    self._state = "half-open"
                else:
                    raise NetworkError("Circuit breaker is open")
            
            try:
                result = func(*args, **kwargs)
                
                # Success - reset if we were in half-open
                if self._state == "half-open":
                    self._state = "closed"
                    self._failure_count = 0
                
                return result
                
            except self.expected_exception:
                self._failure_count += 1
                self._last_failure_time = time.time()
                
                if self._failure_count >= self.failure_threshold:
                    self._state = "open"
                
                raise
                
        return wrapper
    
    def reset(self):
        """Manually reset the circuit breaker."""
        self._failure_count = 0
        self._last_failure_time = None
        self._state = "closed"
