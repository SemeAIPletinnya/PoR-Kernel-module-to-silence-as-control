"""Backwards-compatible shim for the metrics module."""

from .drift import (
    MetricResult,
    compute_coherence_entropy,
    compute_coherence_factual_grounding,
    compute_coherence_perplexity,
    compute_coherence_self_consistency,
    compute_drift_embedding,
    compute_drift_sliding_window,
    compute_drift_token_overlap,
    compute_stability_score,
)

__all__ = [
    "MetricResult",
    "compute_drift_embedding",
    "compute_drift_token_overlap",
    "compute_drift_sliding_window",
    "compute_coherence_perplexity",
    "compute_coherence_self_consistency",
    "compute_coherence_entropy",
    "compute_coherence_factual_grounding",
    "compute_stability_score",
]
