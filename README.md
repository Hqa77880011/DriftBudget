# DriftBudget

Implementation of **Drift-conditioned value of information for costly delayed-label streams**.

Fits a conditional Gaussian feature model and a masked multi-label predictor from an independent warm-start sample, then performs budgeted risk-value acquisition using only arrived label-local drift evidence.

## Install

Use Python 3.11 or newer.

```bash
git clone https://github.com/Hqa77880011/DriftBudget.git
cd DriftBudget
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install.

```bash
python -m pip install -e .
```

## Data preparation

Each scenario contains warm_x/warm_y, chronological x/y, per-event delay, public feature costs, mandatory features and acquisition budget. Test labels are opened only by the arrival queue; drift annotations are evaluator-only.

```bash
python -m driftbudget prepare --count 1 --seed 17 --output data/workload.jsonl
```

The generator creates finite workloads from the formal model. Each JSONL row is one scenario. `--count` controls corpus size and `--seed` controls generation. Imported cases use the same schema. Generate calibration and test scenarios independently.

## Execute and compare

```bash
python -m driftbudget run --data data/workload.jsonl --output runs/main
python -m driftbudget suite --data data/workload.jsonl --output runs/comparison
```

Select methods with `--methods`. driftbudget, global_drift, static_cost, cheap_only, full_features and no_delayed_credit.

## Evaluation and visualization

Class-balanced risk, macro F1, tail recall, acquisition cost, budget compliance, posterior localization delay and selected-feature predicted/realized value correlation.

Each method writes `records.jsonl` and `metrics.json` with means, sample standard deviations, usable counts and P50/P95/P99. Rates are fractions. Undefined conditional metrics remain unavailable. Method timers exclude reference scoring. RSS includes the Python runtime; serialized sizes measure JSON bytes.

```bash
python -m driftbudget plot --inputs runs/comparison/driftbudget --output figures
```

Pass multiple method directories to compare them. Plotting writes `summary.csv` and metric PNGs.

## Implementation conventions

Risk values are model-based conditional Monte Carlo estimates. Delayed credit corrects each purchased feature's label-wise value only when its label arrives. Full_features is an information/cost reference and may exceed the acquisition budget. The controlled generator changes one label's dependence at a known evaluator-only boundary.


## Uncertainty and paired comparisons

```bash
python -m driftbudget.statistics --records runs/main/driftbudget/records.jsonl --output results/intervals.json
```

Supply --reference with a matched record file for paired differences. Resampling uses independent case IDs, or explicit cluster IDs when provided. Repeated observations from one source unit must share cluster. Zero-event bounds refer to an event in an independent cluster.

## Registered compact reproduction setting

Eight independent finite streams, eight features/three labels, 100 independent warm events and 240 online events per stream. The linear masked predictor and Gaussian conditional feature law implement the declared drift-conditioned acquisition problem.

Run from the repository root after installation. The following commands use the registered setting; they are provided for execution and have not been run as part of the source review.

```bash
python -m driftbudget prepare --config configs/reproduce.json --output data/reproduce.jsonl
python -m driftbudget suite --data data/reproduce.jsonl --output runs/reproduce
```

This setting needs no external dataset or pretrained weights. Generated inputs and declared fault/shift laws define the finite task; reported quantities retain that provenance.
