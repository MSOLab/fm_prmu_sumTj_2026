# TODO

## Hard cutoff for CP-SAT solver wall-time overruns

### Problem

OR-Tools CP-SAT can run well past `max_time_in_seconds` under heavy CPU
contention. Observed on run `20260513T142520_492897` with
`instance_worker_cnt: 12` × `solver_thread_cnt: 8` (96 OS threads):

| insName | scenario | timelimit | actual elapsed |
|--------:|:---------|----------:|---------------:|
| 297     | c1       | 787.5 s   | 4076.9 s       |
| 117     | c1       | 787.5 s   | 3057.4 s       |
| 51      | c1       | 472.5 s   | 1432.3 s       |

Code path is correct — `fs_single_instance_runner.py:73-89` sets
`stopping_criteria.timelimit = n * m * 0.045`, and
`controller_core.py:474-484` passes the remaining budget to
`SolveConfig.time_limit_s` → `parameters.max_time_in_seconds`. The solver
just doesn't honor it under contention.

Currently mitigated post-hoc by `apply_timelimit_trim` in
`flowshop_tardiness/report/dashboards/obj_log_trim.py`, which trims
`bestObj` / `bestBound` / `totalElapsedTime` to the deadline using the
recorded obj_log time series. Good enough for analysis, but the solver
still wastes hours of wall clock on every overrun.

### Approach: external watchdog thread + `solver.stop_search()`

The reliable interrupt hook OR-Tools provides is `CpSolver.stop_search()`.
Wrap the `solver.solve(...)` call with a `threading.Timer` that fires
`stop_search` after the budget elapses. The timer thread is OS-scheduled
so it stays responsive even when CP-SAT internal threads are starved.

```python
import threading

_timelimit = self.get_remaining_time_limit(computational_time)
hard_deadline = _timelimit + hard_cutoff_margin  # e.g. +5s grace
watchdog = threading.Timer(hard_deadline, self.solver.stop_search)
watchdog.daemon = True
watchdog.start()
try:
    cp_solver_status = self.solver.solve(mdl, solution_callback=...)
finally:
    watchdog.cancel()
```

**Why not in-callback?** `solution_callback` only fires on new feasible
solutions and `best_bound_callback` only on bound updates — neither
guarantees a timely fire when the solver is stuck not improving (the very
case we need to interrupt).

**Why not OS-level kill?** Process kill loses the obj_log writes and
solution state. `stop_search` lets OR-Tools return its current best
gracefully.

### Where to wire it

Two options:

1. **In `flowshop_tardiness/cpsat_model_2/solver.py`** — add a
   `solve_with_watchdog(solver, mdl, callback, hard_deadline_s)` helper
   alongside `configure_solver`. Keep `configure_solver` stateless and
   let the caller manage watchdog lifecycle.
2. **In `controller_core.py:484-498`** — wrap the existing
   `self.solver.solve(...)` line directly. Less invasive but the watchdog
   pattern then lives inside controller code.

Prefer (1) — keeps the cpsat_model_2 layer responsible for solver
configuration and lifecycle, controller stays focused on flow.

### Configuration

Add a `hard_cutoff_margin_s: float = 0.0` knob (or similar) to
`SolveConfig` or `StoppingCriteria`. Margin > 0 lets CP-SAT's own deadline
fire first when behaving normally; watchdog only triggers on actual
overrun.

### Validation

- Reproduce a known overrun (e.g. instance 297 with c1 flow) with
  `instance_worker_cnt=12, solver_thread_cnt=8` and confirm wall time
  stops at `timelimit + margin` instead of 5×.
- Check status code is still `FEASIBLE` (not crashed) and obj_log is
  written.
- Verify trimmed analysis pipeline gives the same `bestObj` as the
  watchdog-truncated run (within the margin slop).

### Complementary mitigation

Even with the watchdog, **`instance_worker_cnt × solver_thread_cnt`
should not exceed physical core count** to avoid the underlying
starvation. Document this in `metadata_cp_lns_20260512.yaml` and similar
configs.

### Related artifacts

- Post-hoc trimming utility: `flowshop_tardiness/report/dashboards/obj_log_trim.py`
- Tests: `tests/test_obj_log_trim.py`
- Aggregation hook: `fs_multi_scenario_runner.py` (look for
  `apply_timelimit_trim` call after `raw_summary_df` is built)
