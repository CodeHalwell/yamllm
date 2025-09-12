"""
Azure AI Foundry provider implementation for unified core providers.
"""

from typing import Dict, List, Any, Optional
import logging

from yamllm.providers.base import BaseProvider
from yamllm.providers.exceptions import ProviderError


logger = logging.getLogger(__name__)


class AzureFoundryProvider(BaseProvider):
    """Azure AI Foundry implementation of BaseProvider."""

    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        self.api_key = api_key or "default"
        self.base_url = base_url
        self.project_id = kwargs.get("project_id")

        # Lazy import Azure SDKs so library works without them installed
        try:
            from azure.ai.inference import ChatCompletionsClient, EmbeddingsClient
            from azure.identity import DefaultAzureCredential
            from azure.core.credentials import AzureKeyCredential
        except Exception as e:
            raise ImportError(
                "azure-ai-inference and azure-identity are required for AzureFoundryProvider"
            ) from e

        if (self.api_key or "").lower() == "default":
            credential = DefaultAzureCredential()
            self.chat_completions_client = ChatCompletionsClient(endpoint=self.base_url, credential=credential)
            self.embeddings_client = EmbeddingsClient(endpoint=self.base_url, credential=credential)
        else:
            credential = AzureKeyCredential(self.api_key)
            self.chat_completions_client = ChatCompletionsClient(endpoint=self.base_url, credential=credential)
            self.embeddings_client = EmbeddingsClient(endpoint=self.base_url, credential=credential)

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
    ):
        params: Dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if stop_sequences:
            params["stop"] = stop_sequences

        try:
            if stream:
                return self.chat_completions_client.create_stream(deployment_name=model, **params)
            return self.chat_completions_client.create(deployment_name=model, **params)
        except Exception as e:
            logger.error(f"Azure Foundry error: {e}")
            raise ProviderError(f"Azure Foundry error: {e}") from e

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
    ):
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
            resp = self.embeddings_client.create(deployment_name=model, input=text)
            return list(resp.data[0].embedding)
        except Exception as e:
            logger.error(f"Azure Foundry embedding error: {e}")
            raise ProviderError(f"Azure Foundry embedding error: {e}") from e

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
            elif isinstance(tc, dict):
                formatted.append(
                    {
                        "id": tc.get("id") or f"call_{len(formatted)}",
                        "type": "function",
                        "function": {
                            "name": tc.get("function", {}).get("name"),
                            "arguments": tc.get("function", {}).get("arguments"),
                        },
                    }
                )
        return formatted

    def format_tool_results(self, tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "role": "tool",
                "tool_call_id": r.get("tool_call_id"),
                "content": r.get("content"),
            }
            for r in tool_results
        ]

    def close(self):
        return None
