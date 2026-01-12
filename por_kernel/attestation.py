"""Attestation helpers for Proof-of-Resonance runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Attestation:
    """Attestation record for a resonance decision."""

    resonance: str
    decision: str
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["metadata"] is None:
            data["metadata"] = {}
        return data


def generate_attestation(
    *,
    resonance: str,
    decision: str,
    metadata: Mapping[str, Any] | None = None,
) -> Attestation:
    """Create an attestation record for a run."""
    return Attestation(resonance=resonance, decision=decision, metadata=metadata)
