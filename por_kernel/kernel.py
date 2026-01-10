"""Core functionality for Proof-of-Resonance."""


def resonate(signal: str) -> str:
    """Return a normalized resonance signature for the given signal."""
    cleaned = signal.strip()
    if not cleaned:
        return "silent"
    return f"resonance:{cleaned.lower()}"
