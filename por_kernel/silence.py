"""Silence-gating logic for Proof-of-Resonance."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SilenceDecision:
    """Result of silence gating."""

    should_silence: bool
    reason: str


def gate_silence(
    *,
    drift: float,
    coherence: float,
    tolerance: float,
    threshold: float,
) -> SilenceDecision:
    """
    Apply the silence-as-control rule.

    If drift exceeds tolerance or coherence drops below threshold, gate output.
    """
    if drift > tolerance:
        return SilenceDecision(True, "drift_exceeded")
    if coherence < threshold:
        return SilenceDecision(True, "coherence_below_threshold")
    return SilenceDecision(False, "stable")
