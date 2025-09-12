from __future__ import annotations

from typing import Dict, Any, List
import os



class ConfigValidator:
    """Validate parsed YAML configuration and mask sensitive values for logs."""

    API_KEY_PATTERNS = {
        "openai": r"^sk-[A-Za-z0-9]{20,}$",
        "anthropic": r"^sk-ant-[A-Za-z0-9_-]{40,}$",
        "google": r"^[A-Za-z0-9_-]{25,}$",
        "mistral": r"^[A-Za-z0-9]{20,}$",
        # add more as needed
    }

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> List[str]:
        """Return a list of validation error strings (empty if valid)."""
        errors: List[str] = []

        provider = (config.get("provider") or {})
        if not provider:
            errors.append("Missing 'provider' section")
        else:
            name = (provider.get("name") or "").strip()
            model = (provider.get("model") or "").strip()
            if not name:
                errors.append("Missing provider name (provider.name)")
            if not model:
                errors.append("Missing model (provider.model)")

            # Validate API key format if provided
            api_key = provider.get("api_key")
            if api_key:
                # Strictly enforce env-only storage for API keys
                errors.append(
                    "API keys must not be stored in configuration files. "
                    f"Remove 'provider.api_key' and set {name.upper()}_API_KEY environment variable instead."
                )

        # Validate memory directories if enabled
        context = config.get("context") or {}
        memory = (context.get("memory") or {})
        if memory.get("enabled"):
            conv_db = memory.get("conversation_db")
            if conv_db:
                conv_dir = os.path.dirname(conv_db)
                if conv_dir and not os.path.exists(conv_dir):
                    try:
                        os.makedirs(conv_dir, exist_ok=True)
                    except PermissionError:
                        errors.append(f"Cannot create memory directory: {conv_dir}")

            vstore = memory.get("vector_store") or {}
            for pkey in ("index_path", "metadata_path"):
                pval = vstore.get(pkey)
                if pval:
                    pdir = os.path.dirname(pval)
                    if pdir and not os.path.exists(pdir):
                        try:
                            os.makedirs(pdir, exist_ok=True)
                        except PermissionError:
                            errors.append(f"Cannot create vector_store directory: {pdir}")

        return errors

    @staticmethod
    def mask_sensitive_data(config: Dict[str, Any]) -> Dict[str, Any]:
        """Return a deep copy with common secret-like values masked for safe logging."""
        import copy

        masked = copy.deepcopy(config)

        def mask_value(v: str) -> str:
            if not isinstance(v, str):
                return v
            length = len(v)
            if length <= 4:
                return "*" * length
            elif length <= 8:
                return v[0] + "*" * (length - 2) + v[-1]
            elif length <= 16:
                return v[:2] + "*" * (length - 4) + v[-2:]
            else:
                # For longer values, show first 4 and last 4 characters (tests expect e.g., 'sk-t' prefix)
                return v[:4] + "*" * max(4, length - 8) + v[-4:]

        def recurse(obj: Any):
            if isinstance(obj, dict):
                for k, v in list(obj.items()):
                    if any(t in k.lower() for t in ("key", "secret", "password", "token")):
                        obj[k] = mask_value(v) if isinstance(v, str) else v
                    else:
                        recurse(v)
            elif isinstance(obj, list):
                for i in range(len(obj)):
                    recurse(obj[i])

        recurse(masked)
        return masked
