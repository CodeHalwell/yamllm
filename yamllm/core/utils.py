def log_message(message: str) -> None:
    """Logs a message to the console."""
    print(f"[LOG] {message}")

def handle_error(error: Exception) -> None:
    """Handles errors by logging the error message."""
    print(f"[ERROR] {str(error)}")

def mask_string(sensitive_string: str, unmasked_prefix_len: int = 4, unmasked_suffix_len: int = 4) -> str:
    """Masks a sensitive string, leaving a prefix and suffix unmasked."""
    if not isinstance(sensitive_string, str) or len(sensitive_string) <= unmasked_prefix_len + unmasked_suffix_len:
        return "********"
    
    prefix = sensitive_string[:unmasked_prefix_len]
    suffix = sensitive_string[-unmasked_suffix_len:]
    
    return f"{prefix}{'*' * (len(sensitive_string) - unmasked_prefix_len - unmasked_suffix_len)}{suffix}"