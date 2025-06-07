import hashlib

def calculate_sha256(text: str) -> str:
    """Calculates the SHA256 hash of a string."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

__all__ = ["calculate_sha256"]
