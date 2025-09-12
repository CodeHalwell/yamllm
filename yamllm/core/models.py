
"""
Response models for provider compatibility and type safety.

Standardizes responses across different LLM providers with Pydantic validation.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, model_validator


class MessageRole(str, Enum):
    """Standard message roles across providers."""
    USER = "user"
    ASSISTANT = "assistant" 
    SYSTEM = "system"
    TOOL = "tool"


class Message(BaseModel):
    """Standardized message format."""
    role: MessageRole
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class Usage(BaseModel):
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @model_validator(mode='before')
    @classmethod
    def calculate_total(cls, data: Any) -> Any:
        """Auto-calculate total tokens if not provided."""
        if isinstance(data, dict) and ('total_tokens' not in data or data['total_tokens'] == 0):
            prompt_tokens = data.get('prompt_tokens', 0)
            completion_tokens = data.get('completion_tokens', 0)
            data['total_tokens'] = int(prompt_tokens) + int(completion_tokens)
        return data


class ToolFunction(BaseModel):
    """Tool function definition."""
    name: str
    arguments: str  # JSON string of arguments
    
    def get_arguments_dict(self) -> Dict[str, Any]:
        """Parse arguments JSON string to dictionary."""
        try:
            result = json.loads(self.arguments)
            return result if isinstance(result, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}


class ToolCall(BaseModel):
    """Tool call information."""
    id: str
    type: str = "function"
    function: ToolFunction


class CompletionResponse(BaseModel):
    """Standardized completion response across providers."""
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    usage: Optional[Usage] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    provider: Optional[str] = None


class StreamingChunk(BaseModel):
    """Streaming response chunk."""
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: Optional[str] = None
    delta: bool = True  # Indicates this is a delta update


class EmbeddingResponse(BaseModel):
    """Embedding creation response."""
    embedding: List[float]
    model: Optional[str] = None
    usage: Optional[Usage] = None


class ResponseAdapter:
    """Adapters to convert provider-specific responses to standard format."""
    
    @staticmethod
    def adapt_openai_response(response: Any) -> CompletionResponse:
        """Adapt OpenAI API response to standard format."""
        try:
            choice = response.choices[0]
            message = choice.message
            
            # Parse tool calls
            tool_calls: List[ToolCall] = []
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        type=tc.type,
                        function=ToolFunction(
                            name=tc.function.name,
                            arguments=tc.function.arguments
                        )
                    ))
            
            return CompletionResponse(
                content=message.content,
                tool_calls=tool_calls if tool_calls else None,
                usage=Usage(**response.usage.model_dump()) if response.usage else None,
                model=response.model,
                finish_reason=choice.finish_reason,
                provider="openai"
            )
        except (AttributeError, IndexError, KeyError) as e:
            raise ValueError(f"Invalid OpenAI response format: {e}") from e
    
    @staticmethod 
    def adapt_anthropic_response(response: Dict[str, Any]) -> CompletionResponse:
        """Adapt Anthropic API response to standard format."""
        try:
            content_blocks = response.get("content", [])
            text_content = ""
            tool_calls = []
            
            for block in content_blocks:
                if block.get("type") == "text":
                    text_content += block.get("text", "")
                elif block.get("type") == "tool_use":
                    tool_calls.append(ToolCall(
                        id=block.get("id", ""),
                        type="function",
                        function=ToolFunction(
                            name=block.get("name", ""),
                            arguments=json.dumps(block.get("input", {}))
                        )
                    ))
            
            # Extract usage information
            usage = None
            if "usage" in response:
                usage_data = response["usage"]
                usage = Usage(
                    prompt_tokens=usage_data.get("input_tokens", 0),
                    completion_tokens=usage_data.get("output_tokens", 0)
                )
            
            return CompletionResponse(
                content=text_content if text_content else None,
                tool_calls=tool_calls if tool_calls else None,
                usage=usage,
                model=response.get("model"),
                finish_reason=response.get("stop_reason"),
                provider="anthropic"
            )
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid Anthropic response format: {e}") from e
    
    @staticmethod
    def adapt_google_response(response: Dict[str, Any]) -> CompletionResponse:
        """Adapt Google Gemini API response to standard format."""
        try:
            candidates = response.get("candidates", [])
            if not candidates:
                return CompletionResponse(content=None, provider="google")
            
            candidate = candidates[0]
            content_data = candidate.get("content", {})
            parts = content_data.get("parts", [])
            
            text_content = ""
            tool_calls: List[ToolCall] = []
            
            for part in parts:
                if "text" in part:
                    text_content += part["text"]
                elif "functionCall" in part:
                    func_call = part["functionCall"]
                    tool_calls.append(ToolCall(
                        id=f"call_{len(tool_calls)}",
                        type="function", 
                        function=ToolFunction(
                            name=func_call.get("name", ""),
                            arguments=json.dumps(func_call.get("args", {}))
                        )
                    ))
            
            # Extract usage if available
            usage = None
            if "usageMetadata" in response:
                usage_data = response["usageMetadata"]
                usage = Usage(
                    prompt_tokens=usage_data.get("promptTokenCount", 0),
                    completion_tokens=usage_data.get("candidatesTokenCount", 0)
                )
            
            return CompletionResponse(
                content=text_content if text_content else None,
                tool_calls=tool_calls if tool_calls else None,
                usage=usage,
                model=response.get("model"),
                finish_reason=candidate.get("finishReason"),
                provider="google"
            )
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid Google response format: {e}") from e
    
    @staticmethod
    def adapt_generic_response(response: Any, provider: str) -> CompletionResponse:
        """Generic adapter for unknown or simple response formats."""
        try:
            if isinstance(response, dict):
                return CompletionResponse(
                    content=response.get("content"),
                    model=response.get("model"),
                    provider=provider
                )
            elif hasattr(response, 'content'):
                return CompletionResponse(
                    content=response.content,
                    provider=provider
                )
            else:
                return CompletionResponse(
                    content=str(response),
                    provider=provider
                )
        except Exception as e:
            raise ValueError(f"Failed to adapt {provider} response: {e}") from e


class ProviderResponseValidator:
    """Validates and normalizes provider responses."""
    
    @staticmethod
    def validate_and_adapt(response: Any, provider: str) -> CompletionResponse:
        """Validate and adapt response based on provider type."""
        if response is None:
            raise ValueError(f"Received None response from {provider}")
        
        try:
            if provider.lower() == "openai":
                return ResponseAdapter.adapt_openai_response(response)
            elif provider.lower() == "anthropic":
                return ResponseAdapter.adapt_anthropic_response(response)
            elif provider.lower() == "google":
                return ResponseAdapter.adapt_google_response(response)
            else:
                return ResponseAdapter.adapt_generic_response(response, provider)
        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Wrap other errors
            raise ValueError(f"Response validation failed for {provider}: {e}") from e
