from typing import Optional, Dict, Any, List
import traceback
import logging


class YAMLLMException(Exception):
    """
    Base exception for YAMLLM with enhanced error tracking.
    
    All YAMLLM exceptions should inherit from this class to provide
    consistent error handling and debugging information.
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}
        self.logger = logging.getLogger(self.__class__.__module__)
    
    def log_error(self):
        """Log the error with appropriate level and details."""
        self.logger.error(f"{self.__class__.__name__}: {str(self)}", extra=self.details)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "details": self.details
        }


class ConfigurationError(YAMLLMException):
    """Configuration validation errors."""
    
    def __init__(self, message: str, config_path: Optional[str] = None, 
                 validation_errors: Optional[List[str]] = None):
        details = {
            "config_path": config_path,
            "validation_errors": validation_errors or []
        }
        super().__init__(message, details)


class MemoryError(YAMLLMException):
    """Memory storage/retrieval errors (conversation/vector stores)."""
    
    def __init__(self, message: str, operation: Optional[str] = None,
                 store_type: Optional[str] = None):
        details = {
            "operation": operation,
            "store_type": store_type
        }
        super().__init__(message, details)


class ProviderError(YAMLLMException):
    """Provider-related errors with enhanced debugging."""
    
    def __init__(self, provider: str, message: str, 
                 original_error: Optional[Exception] = None,
                 request_params: Optional[Dict[str, Any]] = None):
        self.provider = provider
        self.original_error = original_error
        
        details = {
            "provider": provider,
            "original_error_type": type(original_error).__name__ if original_error else None,
            "original_error_message": str(original_error) if original_error else None,
            "request_params": request_params
        }
        
        # Add traceback if available
        if original_error:
            details["traceback"] = traceback.format_exception(
                type(original_error), original_error, original_error.__traceback__
            )
        
        super().__init__(f"Provider {provider}: {message}", details)


class ToolExecutionError(YAMLLMException):
    """Tool execution errors with enhanced context."""
    
    def __init__(self, tool_name: str, message: str, 
                 original_error: Optional[Exception] = None,
                 tool_args: Optional[Dict[str, Any]] = None,
                 execution_time: Optional[float] = None):
        self.tool_name = tool_name
        self.original_error = original_error
        
        details = {
            "tool_name": tool_name,
            "tool_args": tool_args,
            "execution_time": execution_time,
            "original_error_type": type(original_error).__name__ if original_error else None,
            "original_error_message": str(original_error) if original_error else None
        }
        
        # Add traceback if available
        if original_error:
            details["traceback"] = traceback.format_exception(
                type(original_error), original_error, original_error.__traceback__
            )
        
        super().__init__(f"Tool {tool_name}: {message}", details)


class NetworkError(YAMLLMException):
    """Network-related errors with request details."""
    
    def __init__(self, message: str, url: Optional[str] = None,
                 status_code: Optional[int] = None,
                 response_body: Optional[str] = None):
        details = {
            "url": url,
            "status_code": status_code,
            "response_body": response_body
        }
        super().__init__(message, details)


class ValidationError(YAMLLMException):
    """Data validation errors with field information."""
    
    def __init__(self, message: str, field: Optional[str] = None,
                 value: Any = None, expected_type: Optional[str] = None):
        details = {
            "field": field,
            "value": value,
            "value_type": type(value).__name__ if value is not None else None,
            "expected_type": expected_type
        }
        super().__init__(message, details)


class RateLimitError(YAMLLMException):
    """Rate limiting errors with retry information."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None,
                 limit_type: Optional[str] = None):
        details = {
            "retry_after": retry_after,
            "limit_type": limit_type
        }
        super().__init__(message, details)


class AuthenticationError(YAMLLMException):
    """Authentication/authorization errors."""
    
    def __init__(self, message: str, provider: Optional[str] = None,
                 auth_type: Optional[str] = None):
        details = {
            "provider": provider,
            "auth_type": auth_type
        }
        super().__init__(message, details)


class TimeoutError(YAMLLMException):
    """Timeout errors with operation details."""
    
    def __init__(self, message: str, operation: Optional[str] = None,
                 timeout_seconds: Optional[float] = None):
        details = {
            "operation": operation,
            "timeout_seconds": timeout_seconds
        }
        super().__init__(message, details)
