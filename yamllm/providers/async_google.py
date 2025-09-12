"""
Async Google Gemini provider implementation for YAMLLM.

This module implements the AsyncBaseProvider interface for Google's Gemini API
with full async/await support.
"""

import json
import logging
from typing import Dict, List, Any, Optional, AsyncIterator
import asyncio
from concurrent.futures import ThreadPoolExecutor

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from yamllm.providers.async_base import AsyncBaseProvider
from yamllm.providers.exceptions import ProviderError


logger = logging.getLogger(__name__)


class AsyncGoogleProvider(AsyncBaseProvider):
    """
    Async Google Gemini provider implementation.
    
    This class provides async support for Google Gemini API calls.
    Note: Since google-generativeai doesn't have native async support yet,
    we use ThreadPoolExecutor for async behavior.
    """
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        """
        Initialize the async Google provider.
        
        Args:
            api_key: Google API key
            base_url: Not used for Google
            **kwargs: Additional parameters
        """
        self.api_key = api_key
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Safety settings
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
    
    async def __aenter__(self):
        """Initialize Google client."""
        genai.configure(api_key=self.api_key)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources."""
        self.executor.shutdown(wait=True)
    
    async def get_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        **kwargs
    ) -> Any:
        """
        Get an async completion from Google Gemini.
        
        Args:
            messages: List of message objects
            model: Google model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stop_sequences: Optional stop sequences
            tools: Optional tool definitions
            tool_choice: Tool choice strategy
            **kwargs: Additional parameters
            
        Returns:
            Response object
        """
        loop = asyncio.get_event_loop()
        
        # Run synchronous method in executor
        response = await loop.run_in_executor(
            self.executor,
            self._get_completion_sync,
            messages, model, temperature, max_tokens, top_p,
            stop_sequences, tools, tool_choice, kwargs
        )
        
        return response
    
    def _get_completion_sync(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        kwargs: Dict = None
    ) -> Any:
        """Synchronous completion method to run in executor."""
        try:
            # Initialize model
            if not self.model or self.model.model_name != model:
                generation_config = genai.GenerationConfig(
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=max_tokens,
                    stop_sequences=stop_sequences
                )
                
                if tools:
                    # Convert tools to Google format
                    google_tools = self._convert_tools(tools)
                    self.model = genai.GenerativeModel(
                        model_name=model,
                        generation_config=generation_config,
                        safety_settings=self.safety_settings,
                        tools=google_tools
                    )
                else:
                    self.model = genai.GenerativeModel(
                        model_name=model,
                        generation_config=generation_config,
                        safety_settings=self.safety_settings
                    )
            
            # Convert messages to Google format
            google_messages = self._convert_messages(messages)
            
            # Create chat
            chat = self.model.start_chat(history=google_messages[:-1])
            
            # Send the last message
            response = chat.send_message(google_messages[-1]["parts"][0])
            
            # Format response
            result = {
                "text": response.text,
                "usage": {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count
                }
            }
            
            # Check for function calls
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call'):
                        if "tool_calls" not in result:
                            result["tool_calls"] = []
                        
                        result["tool_calls"].append({
                            "id": f"call_{len(result['tool_calls'])}",
                            "type": "function",
                            "function": {
                                "name": part.function_call.name,
                                "arguments": json.dumps(dict(part.function_call.args))
                            }
                        })
            
            return result
            
        except Exception as e:
            logger.error(f"Google API error: {str(e)}")
            raise ProviderError("Google", f"API error: {str(e)}") from e
    
    async def get_streaming_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Get an async streaming completion from Google.
        
        Note: Uses sync streaming in a thread since google-generativeai
        doesn't support async streaming yet.
        """
        loop = asyncio.get_event_loop()
        
        # Create async generator from sync stream
        sync_stream = await loop.run_in_executor(
            self.executor,
            self._get_streaming_sync,
            messages, model, temperature, max_tokens, top_p,
            stop_sequences, tools, tool_choice, kwargs
        )
        
        # Convert to async generator
        for chunk in sync_stream:
            yield chunk
    
    def _get_streaming_sync(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        kwargs: Dict = None
    ):
        """Synchronous streaming method."""
        try:
            # Initialize model
            if not self.model or self.model.model_name != model:
                generation_config = genai.GenerationConfig(
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=max_tokens,
                    stop_sequences=stop_sequences
                )
                
                self.model = genai.GenerativeModel(
                    model_name=model,
                    generation_config=generation_config,
                    safety_settings=self.safety_settings
                )
            
            # Convert messages
            google_messages = self._convert_messages(messages)
            
            # Create chat and stream
            chat = self.model.start_chat(history=google_messages[:-1])
            
            for chunk in chat.send_message(google_messages[-1]["parts"][0], stream=True):
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            logger.error(f"Google streaming error: {str(e)}")
            raise ProviderError("Google", f"Streaming error: {str(e)}") from e
    
    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Convert messages to Google format."""
        google_messages = []
        
        for msg in messages:
            role = msg["role"]
            
            # Skip system messages (handle separately if needed)
            if role == "system":
                continue
            
            # Convert role names
            if role == "assistant":
                role = "model"
            
            # Handle tool results
            if role == "tool":
                google_messages.append({
                    "role": "function",
                    "parts": [{
                        "function_response": {
                            "name": msg.get("name", "unknown"),
                            "response": json.loads(msg["content"]) if msg["content"].startswith("{") else {"result": msg["content"]}
                        }
                    }]
                })
            else:
                google_messages.append({
                    "role": role,
                    "parts": [msg["content"]]
                })
        
        return google_messages
    
    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[genai.Tool]:
        """Convert tools to Google format."""
        functions = []
        
        for tool in tools:
            if tool["type"] == "function":
                func = tool["function"]
                functions.append({
                    "name": func["name"],
                    "description": func["description"],
                    "parameters": func["parameters"]
                })
        
        return [genai.Tool(function_declarations=functions)]
    
    async def create_embedding(self, text: str, model: str = "models/embedding-001") -> List[float]:
        """Create an embedding asynchronously."""
        loop = asyncio.get_event_loop()
        
        def _create_embedding():
            embed_model = genai.GenerativeModel(model)
            result = embed_model.embed_content(text)
            return result.embedding
        
        embedding = await loop.run_in_executor(self.executor, _create_embedding)
        return embedding
    
    def format_tool_calls(self, tool_calls: Any) -> List[Dict[str, Any]]:
        """Format tool calls to standardized format."""
        # Already formatted in get_completion
        return tool_calls