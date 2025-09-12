"""
Google Gemini provider for unified core providers.
"""

from typing import Dict, List, Any, Optional, Iterator
import logging
import google.generativeai as genai
from yamllm.providers.exceptions import ProviderError

from yamllm.providers.base import BaseProvider


logger = logging.getLogger(__name__)


class GoogleGeminiProvider(BaseProvider):
    """Google Gemini implementation of BaseProvider using google-generativeai SDK."""

    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self.extra = kwargs

        # Configure client
        genai.configure(api_key=self.api_key)
        if self.base_url:
            genai.configure(transport="rest", client_options={"api_endpoint": self.base_url})

        # Model is passed per-call; keep a default if provided
        self.default_model = kwargs.get("model")

    # Message conversion
    def _to_google_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []
        # Combine system messages into a user prompt header
        sys_lines: List[str] = [m.get("content", "") for m in messages if m.get("role") == "system"]
        if sys_lines:
            processed.append({"role": "user", "parts": [{"text": "System instructions:\n" + "\n".join(sys_lines)}]})
            processed.append({"role": "model", "parts": [{"text": "I'll follow these instructions."}]})

        for m in messages:
            role = m.get("role")
            if role == "system":
                continue
            if role == "user":
                processed.append({"role": "user", "parts": [{"text": m.get("content", "")}]} )
            elif role == "assistant":
                processed.append({"role": "model", "parts": [{"text": m.get("content", "")}]} )
            # tool messages are not passed here; Gemini expects function responses appended as parts
        return processed

    def _to_google_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for t in tools or []:
            fn = t.get("function", {}) if isinstance(t, dict) else {}
            out.append({
                "function_declarations": [{
                    "name": fn.get("name"),
                    "description": fn.get("description"),
                    "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                }]
            })
        return out

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
        contents = self._to_google_messages(messages)
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens,
            "stop_sequences": stop_sequences or None,
        }
        mdl = model or self.default_model
        gm = genai.GenerativeModel(model_name=mdl)
        try:
            if tools:
                return gm.generate_content(contents=contents, generation_config=generation_config, tools=self._to_google_tools(tools))
            return gm.generate_content(contents=contents, generation_config=generation_config)
        except Exception as e:
            logger.error(f"Google Gemini error: {e}")
            raise ProviderError("Google", f"API error: {e}", original_error=e) from e

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
        contents = self._to_google_messages(messages)
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens,
            "stop_sequences": stop_sequences or None,
        }
        mdl = model or self.default_model
        gm = genai.GenerativeModel(model_name=mdl)
        try:
            if tools:
                return gm.generate_content(contents=contents, generation_config=generation_config, tools=self._to_google_tools(tools), stream=True)
            return gm.generate_content(contents=contents, generation_config=generation_config, stream=True)
        except Exception as e:
            logger.error(f"Google Gemini streaming error: {e}")
            raise ProviderError("Google", f"Streaming error: {e}", original_error=e) from e

    def create_embedding(self, text: str, model: str = "models/text-embedding-004") -> List[float]:
        try:
            resp = genai.embed_content(model=model, content=text)
            vec = resp.get("embedding") if isinstance(resp, dict) else getattr(resp, "embedding", None)
            return list(vec or [])
        except Exception as e:
            logger.error(f"Google Gemini embedding error: {e}")
            raise ProviderError("Google", f"Embedding error: {e}", original_error=e) from e

    def format_tool_calls(self, tool_calls: Any) -> List[Dict[str, Any]]:
        # Google returns function_call parts; callers using this provider should
        # extract from response.candidates[0].content.parts
        calls: List[Dict[str, Any]] = []
        try:
            parts = tool_calls or []
            for idx, p in enumerate(parts):
                fc = getattr(p, "function_call", None) or p.get("function_call")
                if not fc:
                    continue
                name = getattr(fc, "name", None) or fc.get("name")
                args = getattr(fc, "args", None) or fc.get("args") or {}
                calls.append({
                    "id": f"call_{idx}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args if isinstance(args, str) else json_dumps(args),
                    },
                })
        except Exception:
            pass
        return calls

    def format_tool_results(self, tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Convert to Google function_response part format
        res: List[Dict[str, Any]] = []
        for r in tool_results:
            res.append({
                "role": "user",
                "parts": [{
                    "function_response": {
                        "name": r.get("name"),
                        "response": {"content": r.get("content")},
                    }
                }]
            })
        return res

    def close(self):
        return None


def json_dumps(obj: Any) -> str:
    import json
    try:
        return json.dumps(obj)
    except Exception:
        return "{}"
