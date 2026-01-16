"""Run synthetic drift/coherence cases through the PoR silence guard."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from por_kernel import silence_guard


@dataclass(frozen=True)
class BenchmarkCase:
    """Single benchmark input with expected behavior."""

    case_id: str
    category: str
    description: str
    drift: float
    coherence: float
    expected_stability: str


def load_cases(dataset_path: Path) -> list[BenchmarkCase]:
    """Load benchmark cases from the dataset JSON file."""
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases: list[BenchmarkCase] = []

    for entry in data.get("cases", []):
        signal = entry.get("signal", {})
        expected = entry.get("expected", {})
        drift = _compute_drift(signal)
        coherence = _compute_coherence(signal, expected)
        cases.append(
            BenchmarkCase(
                case_id=entry.get("id", "unknown"),
                category=entry.get("category", "unknown"),
                description=entry.get("description", ""),
                drift=drift,
                coherence=coherence,
                expected_stability=expected.get("stability", "unknown"),
            )
        )
    return cases


def _compute_drift(signal: dict[str, Any]) -> float:
    """Compute a synthetic drift score from the signal definition."""
    drift_rate = float(signal.get("drift_rate", 0.0))
    step_change = abs(float(signal.get("step_change_radians", 0.0)))
    phase_offset = abs(float(signal.get("phase_offset_radians", 0.0)))
    return drift_rate + step_change + phase_offset


def _compute_coherence(signal: dict[str, Any], expected: dict[str, Any]) -> float:
    """Compute a synthetic coherence score using expected values when present."""
    if "coherence_score" in expected:
        return float(expected["coherence_score"])
    noise_sigma = float(signal.get("noise_sigma", 0.0))
    phase_jitter = float(signal.get("phase_jitter_radians", 0.0))
    return max(0.0, 1.0 - (noise_sigma + phase_jitter))


def evaluate_cases(cases: list[BenchmarkCase]) -> list[dict[str, Any]]:
    """Evaluate benchmark cases using the silence guard."""
    results: list[dict[str, Any]] = []
    for case in cases:
        should_proceed, _ = silence_guard(
            "benchmark-output",
            drift=case.drift,
            coherence=case.coherence,
        )
        expected_proceed = case.expected_stability == "stable"
        results.append(
            {
                "id": case.case_id,
                "category": case.category,
                "description": case.description,
                "drift": case.drift,
                "coherence": case.coherence,
                "expected_stability": case.expected_stability,
                "expected_proceed": expected_proceed,
                "actual_proceed": should_proceed,
                "passed": expected_proceed == should_proceed,
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PoR benchmark cases.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset.json"),
        help="Path to dataset.json",
    )
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    results = evaluate_cases(cases)

    total = len(results)
    passed = sum(1 for result in results if result["passed"])
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"[{status}] {result['id']} | drift={result['drift']:.3f} | "
            f"coherence={result['coherence']:.3f} | expected={result['expected_stability']} | "
            f"proceed={result['actual_proceed']}"
        )

    print(f"\nSummary: {passed}/{total} cases passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
