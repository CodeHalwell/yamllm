from __future__ import annotations

import os
from pathlib import Path
from typing import List, Set, Optional
from urllib.parse import urlparse
import ipaddress


class ToolExecutionError(Exception):
    """Raised when a tool is executed in a disallowed or unsafe way."""


class SecurityManager:
    """
    Centralized security checks for tools.

    - File access allowlist restricted to workspace by default.
    - Safe mode to disable network and/or filesystem access.
    - Blocked domains list and internal IP protection for network calls.
    """

    def __init__(
        self,
        allowed_paths: Optional[List[str]] = None,
        *,
        safe_mode: bool = False,
        allow_network: bool = True,
        allow_filesystem: bool = True,
        blocked_domains: Optional[List[str]] = None,
    ) -> None:
        self.workspace_root = os.path.abspath(os.getcwd())
        base_allowed = set(os.path.abspath(p) for p in (allowed_paths or []))
        base_allowed.add(self.workspace_root)
        self.allowed_paths: Set[str] = base_allowed

        self.safe_mode = bool(safe_mode)
        self.allow_network = bool(allow_network)
        self.allow_filesystem = bool(allow_filesystem)
        self.blocked_domains: Set[str] = set((blocked_domains or []))

    # Filesystem -----------------------------------------------------------
    def validate_file_access(self, file_path: str) -> str:
        """Return a safe absolute path if access is allowed, else raise ToolExecutionError."""
        if self.safe_mode or not self.allow_filesystem:
            raise ToolExecutionError("Filesystem access disabled by configuration")

        # Early null-byte check to avoid OS errors and provide consistent message
        if "\x00" in (file_path or ""):
            raise ToolExecutionError("Null byte in path not allowed")

        # Resolve symlinks and normalize the path to prevent traversal attacks
        try:
            # Expand user home directory and get the real, absolute path
            expanded_path = os.path.expanduser(file_path)
            abs_path = os.path.abspath(os.path.realpath(expanded_path))
        except (OSError, ValueError) as e:
            raise ToolExecutionError(f"Invalid file path: {e}")

        # Additional checks for path traversal attempts
        path_parts = Path(abs_path).parts
        if ".." in path_parts:
            # Normalize message so tests and callers can match consistently
            raise ToolExecutionError("Access denied: path traversal attempt detected")

        # Verify the resolved path is within allowed directories
        is_allowed = False
        for allowed_path in self.allowed_paths:
            try:
                allowed_real = os.path.abspath(os.path.realpath(allowed_path))
                # Use os.path.commonpath to ensure abs_path is truly under allowed_path
                if os.path.commonpath([abs_path, allowed_real]) == allowed_real:
                    is_allowed = True
                    break
            except (OSError, ValueError):
                # Skip invalid allowed paths
                continue

        if not is_allowed:
            raise ToolExecutionError(f"Access denied: {file_path} is outside allowed directories")

        return abs_path

    # Network --------------------------------------------------------------
    def check_network_permission(self, url: Optional[str] = None) -> None:
        if self.safe_mode or not self.allow_network:
            raise ToolExecutionError("Network access disabled by configuration")

        if url is None:
            return

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ToolExecutionError("Only http/https URLs are allowed")

        host = parsed.hostname or ""
        # Block explicit localhost
        if host in {"localhost", "127.0.0.1", "::1"}:
            raise ToolExecutionError("Access to localhost is not allowed")

        # Block private/internal/special IPs
        try:
            ip = ipaddress.ip_address(host)
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_unspecified  # 0.0.0.0 / ::
                or ip.is_multicast
                or ip.is_reserved
            ):
                raise ToolExecutionError(f"Access to internal IP address {ip} is not allowed")
        except ValueError:
            # Not a raw IP; check blocked domains list
            # Block mDNS/zeroconf style hostnames and user-specified blocked domains
            if host.endswith(".local"):
                raise ToolExecutionError("Access to .local domains is not allowed")
            # Only block exact domain or subdomains, not arbitrary substring matches
            def _is_blocked_domain(blocked: str, hostname: str) -> bool:
                blocked = blocked.strip('.').lower()
                hostname = (hostname or '').strip('.').lower()
                return hostname == blocked or hostname.endswith('.' + blocked)

            if any(_is_blocked_domain(bd, host) for bd in self.blocked_domains if bd):
                raise ToolExecutionError(f"Access to domain {host} is blocked")
