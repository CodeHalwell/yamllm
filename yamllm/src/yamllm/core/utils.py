def log_message(message: str) -> None:
    """Logs a message to the console."""
    print(f"[LOG] {message}")

def handle_error(error: Exception) -> None:
    """Handles errors by logging the error message."""
    print(f"[ERROR] {str(error)}")