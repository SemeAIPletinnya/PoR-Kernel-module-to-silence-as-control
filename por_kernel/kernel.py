"""Core Proof-of-Resonance kernel utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .drift import compute_coherence_entropy as _compute_coherence_entropy
from .drift import compute_coherence_self_consistency as _compute_coherence_self_consistency
from .drift import compute_drift_embedding
from .silence import gate_silence


class Decision(str, Enum):
    """High-level decision emitted by the PoR kernel."""

    SILENCE = "silence"
    PROCEED = "proceed"


@dataclass(frozen=True)
class SilenceToken:
    """Token representing a decision to remain silent."""

    decision: Decision = Decision.SILENCE
    reason: str = "silence"


@dataclass(frozen=True)
class ProceedToken:
    """Token representing a decision to proceed."""

    decision: Decision = Decision.PROCEED
    reason: str = "stable"


KernelResult = SilenceToken | ProceedToken


@dataclass(frozen=True)
class KernelConfig:
    """Configuration for PoR kernel thresholds."""

    drift_tolerance: float = 0.2
    coherence_threshold: float = 0.8


DEFAULT_CONFIG = KernelConfig()


def por_kernel(*, drift: float, coherence: float, config: KernelConfig = DEFAULT_CONFIG) -> KernelResult:
    """Evaluate drift/coherence and return a decision token."""
    decision = gate_silence(
        drift=drift,
        coherence=coherence,
        tolerance=config.drift_tolerance,
        threshold=config.coherence_threshold,
    )
    if decision.should_silence:
        return SilenceToken(reason=decision.reason)
    return ProceedToken(reason=decision.reason)


def silence_guard(
    text: str,
    *,
    drift: float,
    coherence: float,
    config: KernelConfig = DEFAULT_CONFIG,
) -> tuple[bool, str]:
    """Return (should_proceed, output) for a candidate response."""
    result = por_kernel(drift=drift, coherence=coherence, config=config)
    if isinstance(result, ProceedToken):
        return True, text
    return False, ""


def consensus_kernel(results: Sequence[KernelResult]) -> KernelResult:
    """Aggregate multiple kernel results into a single consensus decision."""
    if not results:
        return SilenceToken(reason="no_votes")

    proceed_votes = sum(
        1 for result in results if isinstance(result, ProceedToken) or result.decision == Decision.PROCEED
    )
    silence_votes = len(results) - proceed_votes

    if proceed_votes >= silence_votes:
        return ProceedToken(reason="consensus")
    return SilenceToken(reason="consensus_silence")


def compute_drift_cosine(
    current_embedding: Sequence[float],
    reference_embedding: Sequence[float],
) -> float:
    """Compute cosine drift between two embeddings."""
    return float(
        compute_drift_embedding(current_embedding, reference_embedding, metric="cosine").value
    )


def compute_drift_euclidean(
    current_embedding: Sequence[float],
    reference_embedding: Sequence[float],
) -> float:
    """Compute euclidean drift between two embeddings."""
    return float(
        compute_drift_embedding(current_embedding, reference_embedding, metric="euclidean").value
    )


def compute_coherence_entropy(token_probs: Sequence[float]) -> float:
    """Compute coherence from token entropy."""
    return float(_compute_coherence_entropy(token_probs).value)


def compute_coherence_consistency(samples: Sequence[str]) -> float:
    """Compute coherence from self-consistency across samples."""
    return float(_compute_coherence_self_consistency(samples).value)


__all__ = [
    "Decision",
    "SilenceToken",
    "ProceedToken",
    "KernelResult",
    "KernelConfig",
    "DEFAULT_CONFIG",
    "por_kernel",
    "silence_guard",
    "consensus_kernel",
    "compute_drift_cosine",
    "compute_drift_euclidean",
    "compute_coherence_entropy",
    "compute_coherence_consistency",
]
