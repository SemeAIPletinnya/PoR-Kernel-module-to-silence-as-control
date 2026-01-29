"""CLI entrypoint for Proof-of-Resonance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .attestation import generate_attestation
from .core import resonate
from .silence import gate_silence


def _load_config(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping")
    return data


def _write_attestation(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def run(config_path: Path) -> int:
    config = _load_config(config_path)

    print("Running Proof-of-Resonance core validation")
    signal = str(config.get("signal", ""))
    resonance_signature = resonate(signal)

    print("Running Resonant Stability Field validation")
    drift = float(config.get("drift", 0.0))
    coherence = float(config.get("coherence", 1.0))
    tolerance = float(config.get("tolerance", 0.5))
    threshold = float(config.get("threshold", 0.5))
    decision = gate_silence(
        drift=drift,
        coherence=coherence,
        tolerance=tolerance,
        threshold=threshold,
    )

    print("Generating Build Provenance")
    metadata = config.get("metadata", {})
    attestation = generate_attestation(
        resonance=resonance_signature,
        decision="silence" if decision.should_silence else "proceed",
        metadata=metadata if isinstance(metadata, dict) else {},
    )

    attestation_path = config.get("attestation_path")
    if attestation_path:
        _write_attestation(Path(attestation_path), attestation.to_dict())

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="por", description="Proof-of-Resonance CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run resonance checks")
    run_parser.add_argument("--config", required=True, help="Path to YAML configuration file")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        return run(Path(args.config))

    parser.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
