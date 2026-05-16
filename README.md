# flowshop-tardiness

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Permutation flow shop scheduling to minimize **total tardiness** (`Σ Tj`).
Implements and benchmarks CP-SAT, MILP (CPLEX), and genetic algorithm solvers.

## Quick start

```bash
uv sync        # install dependencies
uv run python main.py
```

## Problem

Given `n` jobs on `m` stages (flow shop), find a permutation of jobs
that minimizes the sum of completion times past their due dates.
Each job visits stages 1..m in order; stages sequence jobs in the
same order (permutation flow shop).

## Solvers

| Solver  | Model                  | Runner                   |
|---------|------------------------|--------------------------|
| CP-SAT  | Indirect precedence    | `main.py`                |
| CP-SAT  | Position-based (+LNS)  | `main_metadata.yaml`     |
| CPLEX   | Assignment formulation  | `tbb_2018_mhx1_main.py`  |
| Genetic | EDD-seeded NEH         | `ga_ctrlr_main.py`       |

- **CP-SAT** with Large Neighborhood Search (LNS) for the primary solver
- **Prefix-Window CP (PW-CP)**: incremental CP-CP decomposition
- **PW-CP + insertion improvement**: PW-CP followed by NEH-like insertion passes
- Due-date-driven heuristics: EDD, MDD, Slack, SRMWK, NEHedd, NEH-MS

## Architecture

```
main.py  ──►  FsMultiScenarioRunner  ──►  FsMultiInstanceRunner  ──►  FsSingleInstanceRunner
                 (scenario configs)       (parallel instances)         (per-instance solve)
```

Each runner supports three modes: `FULL_RUN`, `RESUME`, and
`POST_PROCESS_ONLY`. Results are written as per-instance summary CSVs
and aggregated into multi-scenario dashboards (Excel + HTML).

## Project structure

```
├── flowshop_tardiness/
│   ├── controller/         # Solver controllers and subroutines
│   ├── cpsat_model_2/       # CP-SAT formulations (indirect_prec, position)
│   ├── cplex_model/         # CPLEX MILP formulation
│   ├── genetic_algorithm_model/  # GA implementation
│   ├── report/dashboards/   # Analysis dashboards (RPDf scatter, method mean, pivot)
│   └── painter/             # Gantt chart generation
├── scripts/                # Analysis and post-processing tools
├── configs_*/              # Experiment flow configurations
├── tests/                  # Test suite (pytest)
└── docs/                   # Design docs and TODOs
```

## Mode support

- **FULL_RUN**: fresh solve from scratch
- **RESUME**: continue a previous partial run, reusing solution/obj-log artifacts
- **POST_PROCESS_ONLY**: regenerate dashboards from existing results

CP-SAT wall-clock overruns (observed under CPU contention) are
mitigated post-hoc by trimming obj_log time series to the configured
timelimit via `apply_timelimit_trim`.

## Requirements

Python 3.11. Dependencies managed with `uv`:

- OR-Tools (CP-SAT), CPLEX (optional), docplex (optional)
- `routix` — runner and experiment orchestration framework
- `schore` — scheduling problem instance library
- `mbls` — CP-SAT model building utilities
- `pandas`, `plotly`, `xlsxwriter` — reporting
