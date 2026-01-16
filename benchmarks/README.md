# Benchmarks

This directory captures performance experiments for the Proof of Resonance project.
Use it to store benchmark plans, raw results, and any supporting scripts so that
performance changes can be validated alongside functional changes.

## What to include

- **Plans**: The scenario being measured, input sizes, and the target metrics.
- **Scripts**: Any harnesses or helpers needed to reproduce the measurements.
- **Results**: Raw outputs, summarized tables, and notes on the environment used.

## Adding a new benchmark

1. Create a subdirectory for the experiment (for example `benchmarks/core-cycle/`).
2. Document the setup, dataset, and steps to reproduce results in a `README.md`
   inside that subdirectory.
3. Include raw results and a short summary describing the key observations.

## Reproducibility checklist

- Record the hardware, OS, and kernel/module versions used.
- Note any build flags, environment variables, or configuration toggles.
- Prefer deterministic inputs and fixed seeds where applicable.

When a benchmark is added, keep everything needed to rerun it in the same
subdirectory to make comparisons easy.
