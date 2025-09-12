"""
Azure OpenAI provider implementation for unified core providers.

Implements BaseProvider using the Azure OpenAI SDK. Compatible with
OpenAI-style chat.completions and function tool calling.
"""

from typing import Dict, List, Any, Optional, Iterator
import logging

from openai import AzureOpenAI
from yamllm.providers.exceptions import ProviderError
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from yamllm.providers.base import BaseProvider


logger = logging.getLogger(__name__)


class AzureOpenAIProvider(BaseProvider):
    """Azure OpenAI implementation of BaseProvider."""

    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        # Azure-specific parameters
        self.api_version = kwargs.get("api_version", "2023-05-15")

        # Initialize clients
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.base_url,
        )
        # Reuse same client for embeddings
        self.embedding_client = self.client

    def get_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs,
    ) -> ChatCompletion:
        params: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream,
        }
        if stop_sequences:
            params["stop"] = stop_sequences
        if tools:
            params["tools"] = tools
            if "tool_choice" in kwargs:
                params["tool_choice"] = kwargs["tool_choice"]
        # pass through any extra kwargs
        for k, v in kwargs.items():
            if k not in params and k != "tool_choice":
                params[k] = v
        try:
            return self.client.chat.completions.create(**params)
        except Exception as e:
            logger.error(f"Azure OpenAI error: {e}")
            raise ProviderError(f"Azure OpenAI error: {e}") from e

    def get_streaming_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Iterator[ChatCompletionChunk]:
        return self.get_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop_sequences=stop_sequences,
            tools=tools,
            stream=True,
            **kwargs,
        )

    def create_embedding(self, text: str, model: str = "text-embedding-ada-002") -> List[float]:
        try:
            resp = self.embedding_client.embeddings.create(model=model, input=text)
            return list(resp.data[0].embedding)
        except Exception as e:
            logger.error(f"Azure OpenAI embedding error: {e}")
            raise ProviderError(f"Azure OpenAI embedding error: {e}") from e

    def format_tool_calls(self, tool_calls: Any) -> List[Dict[str, Any]]:
        if not tool_calls:
            return []
        formatted: List[Dict[str, Any]] = []
        for tc in tool_calls:
            if hasattr(tc, "function") and hasattr(tc, "id"):
                formatted.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )
            else:
                fn = getattr(tc, "function", None) or tc.get("function", {})
                formatted.append(
                    {
                        "id": getattr(tc, "id", None) or tc.get("id") or f"call_{len(formatted)}",
                        "type": "function",
                        "function": {
                            "name": getattr(fn, "name", None) or fn.get("name"),
                            "arguments": getattr(fn, "arguments", None) or fn.get("arguments"),
                        },
                    }
                )
        return formatted

    def format_tool_results(self, tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in tool_results:
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": r.get("tool_call_id"),
                    "content": r.get("content"),
                }
            )
        return out

    def close(self):
        # AzureOpenAI client does not require explicit close
        return None

    def process_streaming_tool_calls(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        tools: List[Dict[str, Any]],
        tool_executor,
        stop_sequences: Optional[List[str]] = None,
        max_iterations: int = 10,
        **kwargs,
    ) -> Iterator[ChatCompletionChunk]:
        """
        Azure OpenAI version of the streaming+tools loop (OpenAI-compatible).
        Yields status dicts during tool phases, then streams final chunks.
        """
        current_messages = messages.copy()
        iterations = 0

        # Non-streaming tool loop
        while iterations < max_iterations:
            iterations += 1
            # Yield processing status
            yield {"status": "processing", "iteration": iterations, "max_iterations": max_iterations}

            try:
                response = self.get_completion(
                    messages=current_messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop_sequences=stop_sequences,
                    tools=tools,
                    stream=False,
                    **kwargs,
                )
                assistant_message = response.choices[0].message

                # Append assistant message
                current_messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_message.content,
                        **({"tool_calls": assistant_message.tool_calls} if assistant_message.tool_calls else {}),
                    }
                )

                # If no tool calls, break to final streaming
                if not getattr(assistant_message, "tool_calls", None):
                    break

                # Format and execute tool calls
                formatted = self.format_tool_calls(assistant_message.tool_calls)
                yield {"status": "tool_calls", "tool_calls": formatted}
                tool_results = tool_executor(formatted)
                yield {"status": "tool_results", "tool_results": tool_results}
                current_messages.extend(self.format_tool_results(tool_results))

            except Exception as e:
                logger.error(f"Azure OpenAI tool processing error: {e}")
                yield {"status": "error", "error": str(e)}
                raise ProviderError(f"Azure OpenAI tool processing error: {e}") from e

        # Stream the final response
        try:
            yield {"status": "streaming"}
            stream = self.get_streaming_completion(
                messages=current_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop_sequences=stop_sequences,
                tools=None,
                **kwargs,
            )
            for chunk in stream:
                yield chunk
        except Exception as e:
            logger.error(f"Azure OpenAI streaming error: {e}")
            yield {"status": "error", "error": str(e)}
            raise ProviderError(f"Azure OpenAI streaming error: {e}") from e
