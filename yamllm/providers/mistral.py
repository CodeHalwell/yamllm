"""
Mistral provider for unified core providers using mistralai SDK.
"""

from typing import Dict, List, Any, Optional, Iterator
import logging

from mistralai.sdk import Mistral
from yamllm.providers.exceptions import ProviderError

from yamllm.providers.base import BaseProvider


logger = logging.getLogger(__name__)


class MistralProvider(BaseProvider):
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self.client = Mistral(api_key=self.api_key, server_url=self.base_url)

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
        **kwargs: Any,
    ):
        params: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if stop_sequences:
            params["stop"] = stop_sequences
        if tools:
            # Mistral uses OpenAI-compatible tool format
            params["tools"] = tools
            params["tool_choice"] = kwargs.get("tool_choice", "auto")

        try:
            if stream:
                return self.client.chat.complete(stream=True, **params)
            return self.client.chat.complete(**params)
        except Exception as e:
            logger.error(f"Mistral error: {e}")
            raise ProviderError(f"Mistral error: {e}") from e

    def get_streaming_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Iterator:
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

    def create_embedding(self, text: str, model: str = "mistral-embed") -> List[float]:
        try:
            resp = self.client.embeddings.create(model=model, inputs=text)
            return list(resp.data[0].embedding)
        except Exception as e:
            logger.error(f"Mistral embedding error: {e}")
            raise ProviderError(f"Mistral embedding error: {e}") from e

    def format_tool_calls(self, tool_calls: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for tc in tool_calls or []:
            if hasattr(tc, "function") and hasattr(tc, "id"):
                out.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )
            elif isinstance(tc, dict):
                out.append(
                    {
                        "id": tc.get("id"),
                        "type": "function",
                        "function": {
                            "name": tc.get("function", {}).get("name"),
                            "arguments": tc.get("function", {}).get("arguments"),
                        },
                    }
                )
        return out

    def format_tool_results(self, tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "role": "tool",
                "tool_call_id": r.get("tool_call_id"),
                "name": r.get("name"),
                "content": r.get("content"),
            }
            for r in tool_results
        ]

    def close(self):
        return None
