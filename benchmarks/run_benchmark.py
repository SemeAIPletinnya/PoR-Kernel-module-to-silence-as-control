#!/usr/bin/env python3
"""
PoR Kernel Benchmark Runner
============================

Runs benchmarks against the silence_guard function to validate
drift/coherence thresholds and abstention behavior.

Usage:
    python run_benchmark.py
    python run_benchmark.py --dataset custom_dataset.json --output custom_results.json
    python run_benchmark.py --verbose --env-metadata

Exit codes:
    0 - All benchmarks passed
    1 - One or more benchmarks failed
    2 - Error loading/parsing dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkCase:
    """A single benchmark test case."""
    case_id: str
    description: str
    signal: list[float]
    expected_stable: bool
    coherence_score: float
    tags: list[str] | None = None


@dataclass
class BenchmarkResult:
    """Result of running a single benchmark case."""
    case_id: str
    description: str
    drift: float
    coherence: float
    silence_triggered: bool
    expected_stable: bool
    passed: bool
    execution_time_ms: float


@dataclass
class EnvironmentMetadata:
    """Metadata about the execution environment."""
    timestamp: str
    git_commit: str | None
    git_branch: str | None
    python_version: str
    platform: str
    platform_release: str
    platform_machine: str
    por_kernel_version: str | None


def get_git_info() -> tuple[str | None, str | None]:
    """Get current git commit SHA and branch name."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return (
            commit.stdout.strip() if commit.returncode == 0 else None,
            branch.stdout.strip() if branch.returncode == 0 else None
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None, None


def get_por_kernel_version() -> str | None:
    """Get the installed por_kernel version."""
    try:
        # Try importing from the local package first
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from por_kernel import __version__
        return __version__
    except (ImportError, AttributeError):
        return None


def collect_environment_metadata() -> EnvironmentMetadata:
    """Collect metadata about the execution environment."""
    git_commit, git_branch = get_git_info()
    return EnvironmentMetadata(
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=git_commit,
        git_branch=git_branch,
        python_version=platform.python_version(),
        platform=platform.system(),
        platform_release=platform.release(),
        platform_machine=platform.machine(),
        por_kernel_version=get_por_kernel_version()
    )


def load_cases(dataset_path: Path) -> list[BenchmarkCase]:
    """
    Load benchmark cases from a JSON file.

    Args:
        dataset_path: Path to the dataset JSON file

    Returns:
        List of BenchmarkCase objects

    Raises:
        FileNotFoundError: If dataset file doesn't exist
        json.JSONDecodeError: If JSON is malformed
        ValueError: If required fields are missing
    """
    logger.info(f"Loading dataset from: {dataset_path}")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = []
    for i, item in enumerate(data.get("cases", data if isinstance(data, list) else [])):
        try:
            case = BenchmarkCase(
                case_id=item["case_id"],
                description=item.get("description", ""),
                signal=item["signal"],
                expected_stable=item["expected_stable"],
                coherence_score=item["coherence_score"],
                tags=item.get("tags")
            )
            cases.append(case)
        except KeyError as e:
            raise ValueError(f"Case {i} missing required field: {e}")

    logger.info(f"Loaded {len(cases)} benchmark cases")
    return cases


def _compute_drift(signal: list[float], window: int = 5) -> float:
    """
    Compute semantic drift from a signal.

    Uses variance of differences between consecutive values
    as a proxy for drift/instability.

    Args:
        signal: List of float values representing system state over time
        window: Window size for computing local variance

    Returns:
        Drift score between 0.0 and 1.0
    """
    if len(signal) < 2:
        return 0.0

    # Compute differences between consecutive values
    diffs = [abs(signal[i] - signal[i-1]) for i in range(1, len(signal))]

    if not diffs:
        return 0.0

    # Compute mean absolute difference
    mean_diff = sum(diffs) / len(diffs)

    # Compute variance of differences
    if len(diffs) > 1:
        variance = sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1)
    else:
        variance = 0.0

    # Normalize to 0-1 range (assuming max reasonable variance of 1.0)
    drift = min(1.0, math.sqrt(variance) + mean_diff * 0.5)

    return round(drift, 4)


def _compute_coherence(signal: list[float], baseline_score: float) -> float:
    """
    Compute coherence score from signal and baseline.

    Args:
        signal: List of float values
        baseline_score: Pre-computed coherence score from dataset

    Returns:
        Coherence score between 0.0 and 1.0
    """
    if not signal:
        return baseline_score

    # Use signal stability as coherence modifier
    if len(signal) > 1:
        std_dev = math.sqrt(sum((x - sum(signal)/len(signal))**2 for x in signal) / len(signal))
        stability_factor = max(0.0, 1.0 - std_dev)
    else:
        stability_factor = 1.0

    # Blend baseline score with stability factor
    coherence = baseline_score * 0.7 + stability_factor * 0.3

    return round(min(1.0, max(0.0, coherence)), 4)


def silence_guard(
    output: str,
    drift: float,
    coherence: float,
    drift_tol: float = 0.1,
    coherence_thresh: float = 0.7
) -> tuple[bool, str | None]:
    """
    Determine if output should be silenced based on drift and coherence.

    This is the core PoR Kernel decision function.

    Args:
        output: The output to potentially emit
        drift: Current drift score (0.0 = no drift, 1.0 = complete drift)
        coherence: Current coherence score (0.0 = incoherent, 1.0 = coherent)
        drift_tol: Maximum acceptable drift (default: 0.1)
        coherence_thresh: Minimum required coherence (default: 0.7)

    Returns:
        Tuple of (silence_triggered, output_or_none)
    """
    if drift > drift_tol or coherence < coherence_thresh:
        return True, None
    return False, output


def evaluate_cases(
    cases: list[BenchmarkCase],
    drift_tol: float = 0.1,
    coherence_thresh: float = 0.7,
    verbose: bool = False
) -> list[BenchmarkResult]:
    """
    Evaluate all benchmark cases.

    Args:
        cases: List of benchmark cases to evaluate
        drift_tol: Drift tolerance threshold
        coherence_thresh: Coherence threshold
        verbose: Whether to print detailed output

    Returns:
        List of BenchmarkResult objects
    """
    import time

    results = []

    for case in cases:
        start_time = time.perf_counter()

        # Compute metrics
        drift = _compute_drift(case.signal)
        coherence = _compute_coherence(case.signal, case.coherence_score)

        # Run silence guard
        silence_triggered, _ = silence_guard(
            output="benchmark-output",
            drift=drift,
            coherence=coherence,
            drift_tol=drift_tol,
            coherence_thresh=coherence_thresh
        )

        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        # Determine if result matches expectation
        # If expected_stable=True, we expect NO silence (system is stable)
        # If expected_stable=False, we expect silence (system is unstable)
        expected_silence = not case.expected_stable
        passed = silence_triggered == expected_silence

        result = BenchmarkResult(
            case_id=case.case_id,
            description=case.description,
            drift=drift,
            coherence=coherence,
            silence_triggered=silence_triggered,
            expected_stable=case.expected_stable,
            passed=passed,
            execution_time_ms=round(execution_time_ms, 4)
        )
        results.append(result)

        if verbose:
            status = "PASS" if passed else "FAIL"
            logger.info(
                f"[{status}] {case.case_id}: drift={drift:.4f}, "
                f"coherence={coherence:.4f}, silence={silence_triggered}"
            )

    return results


def generate_summary(results: list[BenchmarkResult]) -> dict[str, Any]:
    """Generate summary statistics from results."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    execution_times = [r.execution_time_ms for r in results]
    avg_time = sum(execution_times) / len(execution_times) if execution_times else 0

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total * 100, 2) if total > 0 else 0,
        "avg_execution_time_ms": round(avg_time, 4),
        "min_execution_time_ms": round(min(execution_times), 4) if execution_times else 0,
        "max_execution_time_ms": round(max(execution_times), 4) if execution_times else 0
    }


def save_results(
    results: list[BenchmarkResult],
    output_path: Path,
    include_metadata: bool = True
) -> None:
    """
    Save benchmark results to JSON file.

    Args:
        results: List of benchmark results
        output_path: Path to output JSON file
        include_metadata: Whether to include environment metadata
    """
    output_data: dict[str, Any] = {
        "summary": generate_summary(results),
        "results": [asdict(r) for r in results]
    }

    if include_metadata:
        metadata = collect_environment_metadata()
        output_data["metadata"] = asdict(metadata)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print a summary of benchmark results to console."""
    summary = generate_summary(results)

    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"Total cases:    {summary['total']}")
    print(f"Passed:         {summary['passed']}")
    print(f"Failed:         {summary['failed']}")
    print(f"Pass rate:      {summary['pass_rate']}%")
    print(f"Avg time:       {summary['avg_execution_time_ms']:.4f} ms")
    print("=" * 50)

    # Print failed cases
    failed_cases = [r for r in results if not r.passed]
    if failed_cases:
        print("\nFAILED CASES:")
        for r in failed_cases:
            print(f"  - {r.case_id}: expected_stable={r.expected_stable}, "
                  f"silence_triggered={r.silence_triggered}")
    print()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PoR Kernel Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--dataset", "-d",
        type=Path,
        default=Path(__file__).parent / "dataset.json",
        help="Path to benchmark dataset JSON file (default: dataset.json)"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path(__file__).parent / "results.json",
        help="Path to output results JSON file (default: results.json)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--env-metadata",
        action="store_true",
        dest="env_metadata",
        help="Include environment metadata in results"
    )

    parser.add_argument(
        "--drift-tol",
        type=float,
        default=0.1,
        help="Drift tolerance threshold (default: 0.1)"
    )

    parser.add_argument(
        "--coherence-thresh",
        type=float,
        default=0.7,
        help="Coherence threshold (default: 0.7)"
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for benchmark runner.

    Returns:
        Exit code: 0 for success, 1 for test failures, 2 for errors
    """
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting PoR Kernel Benchmark Runner")
    logger.info(f"Drift tolerance: {args.drift_tol}")
    logger.info(f"Coherence threshold: {args.coherence_thresh}")

    try:
        # Load benchmark cases
        cases = load_cases(args.dataset)
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        return 2
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in dataset: {e}")
        return 2
    except ValueError as e:
        logger.error(f"Invalid dataset format: {e}")
        return 2

    try:
        # Run benchmarks
        results = evaluate_cases(
            cases=cases,
            drift_tol=args.drift_tol,
            coherence_thresh=args.coherence_thresh,
            verbose=args.verbose
        )
    except Exception as e:
        logger.error(f"Error running benchmarks: {e}")
        return 2

    # Print summary
    print_summary(results)

    # Save results
    if not args.no_save:
        try:
            save_results(
                results=results,
                output_path=args.output,
                include_metadata=args.env_metadata
            )
        except IOError as e:
            logger.error(f"Failed to save results: {e}")
            return 2

    # Return exit code based on test results
    summary = generate_summary(results)
    if summary["failed"] > 0:
        logger.warning(f"{summary['failed']} benchmark(s) failed")
        return 1

    logger.info("All benchmarks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
