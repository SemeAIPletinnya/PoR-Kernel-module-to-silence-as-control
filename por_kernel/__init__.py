"""
PoR Kernel v0.2.0 - Proof-of-Resonance Core
============================================

Production-ready implementation of the Silence-as-Control axiom:
"If coherence cannot be guaranteed, intentional silence is preferred."

Usage:
    >>> from por_kernel import por_kernel, silence_guard
    >>> from por_kernel import SilenceToken, ProceedToken

    # Basic usage
    >>> result = por_kernel(drift=0.05, coherence=0.9)
    >>> if isinstance(result, ProceedToken):
    ...     print("Safe to respond")

    # High-level guard
    >>> result, output = silence_guard("Hello", drift=0.05, coherence=0.9)
    >>> if result:
    ...     print(output)

    # With real metrics
    >>> from por_kernel import compute_drift_cosine, compute_coherence_entropy
    >>> drift = compute_drift_cosine(current_emb, reference_emb)
    >>> coherence = compute_coherence_entropy(token_probs)
    >>> result = por_kernel(drift=drift, coherence=coherence)
"""

from .kernel import (
    # Core tokens and types
    Decision,
    SilenceToken,
    ProceedToken,
    KernelResult,
    KernelConfig,
    DEFAULT_CONFIG,
    # Core functions
    por_kernel,
    silence_guard,
    consensus_kernel,
    # Metric functions
    compute_drift_cosine,
    compute_drift_euclidean,
    compute_coherence_entropy,
    compute_coherence_consistency,
)
from .core import resonate

__all__ = [
    # Core tokens and types
    "Decision",
    "SilenceToken",
    "ProceedToken",
    "KernelResult",
    "KernelConfig",
    "DEFAULT_CONFIG",
    # Core functions
    "por_kernel",
    "silence_guard",
    "consensus_kernel",
    # Metric functions
    "compute_drift_cosine",
    "compute_drift_euclidean",
    "compute_coherence_entropy",
    "compute_coherence_consistency",
    # Legacy core function
    "resonate",
]

__version__ = "0.2.0"
