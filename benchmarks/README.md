# PoR Kernel Benchmarks

This directory contains synthetic benchmarks for validating the PoR Kernel's
silence-as-control behavior.

## Overview

The benchmark suite tests the `silence_guard` function against various
drift and coherence scenarios to ensure deterministic, reproducible abstention
behavior.

## Quick Start

```bash
# Run all benchmarks with default settings
python run_benchmark.py

# Run with verbose output
python run_benchmark.py --verbose

# Run with environment metadata (for CI/reproducibility)
python run_benchmark.py --env-metadata

# Use custom thresholds
python run_benchmark.py --drift-tol 0.15 --coherence-thresh 0.65
```

## Files

| File | Description |
|------|-------------|
| `run_benchmark.py` | Main benchmark runner script |
| `dataset.json` | Test cases with expected outcomes |
| `results.json` | Output from last benchmark run |
| `README.md` | This file |

## CLI Options

```
usage: run_benchmark.py [-h] [--dataset DATASET] [--output OUTPUT]
                        [--verbose] [--env-metadata]
                        [--drift-tol DRIFT_TOL]
                        [--coherence-thresh COHERENCE_THRESH]
                        [--no-save]

Options:
  -h, --help            Show help message
  -d, --dataset PATH    Path to dataset JSON (default: dataset.json)
  -o, --output PATH     Path to results JSON (default: results.json)
  -v, --verbose         Enable verbose output
  --env-metadata        Include environment metadata in results
  --drift-tol FLOAT     Drift tolerance threshold (default: 0.1)
  --coherence-thresh FLOAT  Coherence threshold (default: 0.7)
  --no-save             Don't save results to file
```

## Dataset Format

Each test case in `dataset.json` follows this schema:

```json
{
  "case_id": "unique-identifier",
  "description": "Human-readable description",
  "signal": [0.5, 0.51, 0.52, ...],
  "expected_stable": true,
  "coherence_score": 0.85,
  "tags": ["drift", "stable"]
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `case_id` | string | Unique identifier for the test case |
| `description` | string | Human-readable description |
| `signal` | float[] | Time series of values representing system state |
| `expected_stable` | bool | If `true`, silence should NOT trigger |
| `coherence_score` | float | Pre-computed coherence score (0.0-1.0) |
| `tags` | string[] | Optional tags for filtering/categorization |

## Results Format

The `results.json` output includes:

```json
{
  "metadata": {
    "timestamp": "2026-01-16T12:00:00Z",
    "git_commit": "abc123...",
    "git_branch": "main",
    "python_version": "3.11.0",
    "platform": "Linux",
    "por_kernel_version": "0.1.0"
  },
  "summary": {
    "total": 16,
    "passed": 16,
    "failed": 0,
    "pass_rate": 100.0,
    "avg_execution_time_ms": 0.05
  },
  "results": [
    {
      "case_id": "drift-small-phase",
      "drift": 0.0234,
      "coherence": 0.912,
      "silence_triggered": false,
      "expected_stable": true,
      "passed": true,
      "execution_time_ms": 0.045
    }
  ]
}
```

## Adding New Benchmarks

1. Add a new case to `dataset.json`:
   ```json
   {
     "case_id": "my-new-case",
     "description": "Description of what this tests",
     "signal": [0.1, 0.2, 0.3, ...],
     "expected_stable": true,
     "coherence_score": 0.85,
     "tags": ["custom"]
   }
   ```

2. Run benchmarks to verify:
   ```bash
   python run_benchmark.py --verbose
   ```

## Reproducibility Checklist

For reproducible benchmark results:

- [ ] Pin Python version (3.10+)
- [ ] Run with `--env-metadata` to capture environment info
- [ ] Use fixed random seed if applicable
- [ ] Record git commit SHA in results
- [ ] Document any system-specific configurations

## CI Integration

Add to GitHub Actions:

```yaml
- name: Run PoR Kernel Benchmarks
  run: |
    cd benchmarks
    python run_benchmark.py --env-metadata --verbose

- name: Upload Benchmark Results
  uses: actions/upload-artifact@v3
  with:
    name: benchmark-results
    path: benchmarks/results.json
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All benchmarks passed |
| 1 | One or more benchmarks failed |
| 2 | Error loading dataset or running benchmarks |

## Metrics

### Drift Computation

Drift is computed as:
1. Calculate absolute differences between consecutive signal values
2. Compute mean absolute difference
3. Compute variance of differences
4. Normalize to 0-1 range

### Coherence Computation

Coherence blends:
- Pre-computed `coherence_score` (70% weight)
- Signal stability factor (30% weight)

## Thresholds

Default thresholds match PoR Kernel defaults:

| Metric | Threshold | Behavior |
|--------|-----------|----------|
| Drift | > 0.1 | Trigger silence |
| Coherence | < 0.7 | Trigger silence |

## Contributing

When adding new test cases:

1. Ensure `case_id` is unique and descriptive
2. Document the expected behavior in `description`
3. Use appropriate `tags` for categorization
4. Verify the case passes/fails as expected
5. Consider edge cases and boundary conditions
