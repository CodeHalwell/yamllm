from __future__ import annotations

from abc import ABC
import time
from typing import Optional

import requests

from .base import Tool
from .security import SecurityManager, ToolExecutionError


class NetworkError(Exception):
    pass


class NetworkTool(Tool, ABC):
    """Base class for networked tools with retry/backoff, session, and security checks."""

    def __init__(
        self,
        name: str,
        description: str,
        timeout: int = 15,
        max_retries: int = 3,
        security_manager: Optional[SecurityManager] = None,
    ):
        super().__init__(name=name, description=description)
        # Clamp to safe bounds
        try:
            clamped_timeout = int(timeout)
        except Exception:
            clamped_timeout = 15
        self.timeout = max(1, min(clamped_timeout, 30))

        try:
            clamped_retries = int(max_retries)
        except Exception:
            clamped_retries = 3
        self.max_retries = max(0, min(clamped_retries, 5))
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "yamllm-tool/1.0"})
        self.security: Optional[SecurityManager] = security_manager

    def make_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """HTTP request with retry/backoff on 429/5xx/Timeout.

        Raises NetworkError on final failure.
        """
        # Merge default timeout
        kwargs.setdefault("timeout", self.timeout)
        # Disallow insecure TLS
        if kwargs.get("verify") is False:
            raise NetworkError("Insecure TLS configuration: verify=False is not allowed")
        # Security check
        try:
            if self.security:
                self.security.check_network_permission(url)
        except ToolExecutionError as e:
            raise NetworkError(str(e))
        attempt = 0
        last_error: Optional[Exception] = None
        while attempt < self.max_retries:
            try:
                resp = self.session.request(method, url, **kwargs)
                # Handle rate limit explicitly
                if resp.status_code == 429:
                    raise requests.HTTPError("429 rate limited", response=resp)
                resp.raise_for_status()
                return resp
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
                last_error = e
                attempt += 1
                if attempt >= self.max_retries:
                    break
                # Exponential backoff with jitter-ish additive term
                sleep_time = (2 ** (attempt - 1)) * 0.5 + (attempt * 0.05)
                try:
                    time.sleep(sleep_time)
                except Exception:
                    pass
        raise NetworkError(f"Request failed after {self.max_retries} attempts: {last_error}")
