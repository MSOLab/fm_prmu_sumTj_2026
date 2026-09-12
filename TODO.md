# TODO

Future refactor ideas and design notes that are not urgent enough to act on
today but worth capturing so future work does not re-derive the reasoning.

## Collapse the four run entry points into one

**Problem**

`main.py`, `ga_ctrlr_main.py`, `tbb_2018_mhx1_main.py` and `cp_lns_006nc_main.py`
are ~350 lines each and differ in two things: the controller class handed to
`SubroutineFlowValidator` and the `MAIN_METADATA_FILENAME` constant. The newest
copy, `cp_lns_006nc_main.py`, was made from `main.py` and already drifted:
its `determine_run_mode_and_base_dir` (`:236-262`) resolves `analysis_dir_path`
and `analysis_timestamp` with `if`/`elif`, while `main.py:243-260` still lets
`analysis_timestamp` overwrite the path name. Four copies now mean four fixes
for one bug.

**Approach**

Pick the metadata filename and the controller class from one place — a
`--metadata` argument, or one `run_experiment_main.py` that takes the controller
class as a parameter — and let the per-experiment file keep only the constants.
Before merging copies, decide whether the drifted `analysis_dir_path` resolution
in `cp_lns_006nc_main.py` is the intended behavior and back-port it or revert it.

## Decide which metadata file each entry point reads by default

**Problem**

Each entry point hardcodes the metadata of the experiment that was run last:
`main.py:26` → `metadata_cp_lns_20260609_smoke.yaml`, `ga_ctrlr_main.py:28` →
`ga_ctrlr_metadata_gapr_20reps.yaml`. `ga_ctrlr_metadata.yaml` is still updated
by the rename work but no longer read by any entry point, and `main_metadata.yaml`
is now only a commented-out catalogue. `README.md` says "run `main.py`" without
saying what that runs, and does not list `cp_lns_006nc_main.py` at all.

**Approach**

Either document the "edit the constant before you run" convention next to each
entry point and in the `README.md` solver table, or drop the defaults in favor of
an explicit `--metadata` argument (see the entry-point item above). Then update
`README.md`: add the 006nc runner to the solver table and name the metadata file
the Quick-start command reads.

## Test the runtime guards and `incremental_sw_cp`

**Problem**

The `Literal` typing on new subroutine arguments is not enforceable through routix
(`getattr` + `**kwargs`), so each one has a hand-written runtime guard, and none of
these paths have tests:

- `fm_sumtj_cp_lns.py:1233` — `init_method` must be `dispatch` / `neh-ms` / `lb_only`.
- `genetic_algorithm.py:573` — `speedup` must be `fv2020` / `vr2010` / `none`.
- `fm_sumtj_cp_lns.py:1694,1696` — `incremental_sw_cp` rejects `start_batch_size < 1`
  and `end_batch_size < start_batch_size`.

`incremental_sw_cp` (`fm_sumtj_cp_lns.py:1643-1774`) has no test at all, and it
drives the whole `configs_cp_lns_006nc/20260607_01..08` family. Two behaviors are
worth pinning: the ramp-up visits every batch size once with
`profile_fixed_cnt=0` and `step_size_on_improve=batch_size`, and the polish phase
(`while True` at `:1750`) has no repeat cap — it only ends when a repetition stops
improving or `is_stopping_condition()` fires. If a step ever keeps improving
forever (for example by re-accepting an equal objective elsewhere), the loop only
ends at the time limit.

**Approach**

Reuse the `MagicMock(spec=SwCpContext)` style of `tests/test_sw_cp.py`: drive
`incremental_sw_cp` on a controller stub whose `sw_cp` records its call kwargs, and
assert the batch-size sequence, the two `ValueError` guards, and one
non-improving polish exit. Assert the two `Literal` guards reject a bad config
value, since the flow validator only checks the method name and argument names, not
the values.

## Give ruff a home and a baseline

**Problem**

`pyproject.toml:15` puts `ruff` in the runtime `dependencies`, so every
`uv sync` installs it, and only `[tool.ruff.format]` is configured. Formatting is
clean (`ruff format --check` passes), but `uv run ruff check .` reports 331
violations, mostly pre-existing (the same command reports 422 on `origin/main`):
249 `LOG015` (root-logger calls), 23 `BLE001` (blind `except`), 22 `G201`, 8 `DTZ005`.
The repo also has no CI, so nothing is holding either number in check.

**Approach**

Decide what ruff is for here. If it is a formatting-only tool, move it to a
`[dependency-groups] dev` group and say so in the README. If lint is wanted, add a
`[tool.ruff.lint]` selection with per-file ignores for `checks/` and `scripts/`, and
clear the remaining rules in a dedicated branch rather than in a results branch.

## Index the report directory

**Problem**

`docs/report/README.md` is a title and one sentence, but `AGENTS.md` says it indexes
the write-ups. The two reports added on this branch — `20260607/README.md` and
`20260912_sw_cp_non_monotone.md` — are not listed.

**Approach**

Add one line per write-up: date, one-line subject, and the module or experiment it
covers. Keep the pre-rename output policy in `20260912_sw_cp_non_monotone.md`
("Pre-rename `pw_cp` outputs are frozen") findable from that line.

## Rename the stale `new_acc` insertion helper and its docstring

**Problem**

`genetic_algorithm.py:847` is still called `_get_best_pos_list_and_metric_new_acc`
and its docstring at `:853` still says the job is scored with "NEW acceleration
(Fernandez-Viagas et al., 2020) evaluator". Since `1ccfb49` the helper dispatches
through `_get_insertion_evaluator`, so it may use `vr2010` or `none` instead. The
docstring also describes a `tuple[list[int], int]` return while the method returns
only the position list.

**Approach**

Rename to something mode-neutral (`_get_best_insertion_positions`) and restate the
docstring as "the evaluator selected by `self._insertion_speedup`" with the actual
return type. The only caller is `genetic_algorithm.py:841`.

Do not rename the same-named method in `fm_sumtj_cp_lns.py:743` (callers `:810`,
`:1027`). That CP-LNS copy always uses the Fernandez-Viagas evaluator and is not
switched by the GAPR `speedup` flag, so the two can be renamed independently.

## Remove the unreachable speedup guard

**Problem**

`genetic_algorithm.py:941` raises `Unknown speedup option: {mode!r}` for a mode
outside `_INSERTION_SPEEDUPS`. `mode` is read at `:924` from
`self._insertion_speedup`, which is only ever written at `:575` after `gapr`
validated it at `:573`, and `_get_insertion_evaluator` has no other caller.

**Approach**

Delete the `else` branch. If the defensive check is worth keeping for a future
caller that sets `_insertion_speedup` directly, keep only one of the two guards so
the two copies cannot drift.

## Unwrap the parentheses left by the max() rewrite

**Problem**

`8ba2b30` rewrote `a if a > b else b` into `max(...)` across the DP hot paths, and
three sites kept the original parentheses around the `max` call:
`flowshop_batch_eval.py:111` (`(max(left, up)) + p[i][job]`) and
`flowshop_new_acc.py:80`, `:161` (same shape with `self.p` / `self.p[i][sigma]`).
Harmless, but they read as a leftover and stand out next to the ~15 other rewritten
sites.

**Approach**

Drop the outer parentheses: `max(left, up) + p[i][job]`.

## Hard cutoff for CP-SAT solver wall-time overruns

**Problem**

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

**Approach**: external watchdog thread + `solver.stop_search()`

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

## Accelerate integer completion-time DP (ΣTj evaluation)

**Problem**

The hot op across solvers is the integer completion-time recurrence
`C[i] = max(prev[i], C[i-1]) + p[i][job]`. It is profiled via
`cprofile_main_simulate_append.py` (targets
`FlowshopTardinessCpLnsController._simulate_append`). The current cost is
**Python interpreter overhead, not arithmetic throughput**: the same DP is
duplicated in 4+ places, the CP-LNS hot path uses `dict[str,int]` keyed by
stage names (`fm_sumtj_cp_lns.py:444-467,573-673`) and rebuilds
`ScheduleMetric.p_ij` per insertion position (`:660-669`). No
numba/cupy/torch/jax in the stack — pure Python lists/dicts. Integers only
(confirmed).

**Approach**

Compile first, GPU later. Staged, result-invariant (must not change any
objective value by a single integer — existing equivalence tests are the
oracle):

- **Phase 1 (main, no GPU):** consolidate the DP into one `@njit` kernel
  module (DRY single source of truth), route all evaluators through it as
  thin wrappers, and de-dict the CP-LNS hot path (int-indexed arrays, drop
  per-position `p_ij` rebuild). Expect ~10–100× from compilation alone.
- **Phase 2 (optional, gated on Phase 1 measurement):** GPU
  permutation-across batch (one thread per sequence) for GA
  offspring/multistart scoring. Use position-independent *naive* insertion
  on GPU, not the FV2020 boundary-walk (not position-parallel). CP-SAT /
  CPLEX solve is out of scope.

### Plan doc

Full staged plan with file refs, test strategy (reuses
`test_insertion_speedup_equivalence.py` et al. as oracle), branch + work
order, and acceptance criteria:
**`plans/20260613_numba_jit_dp_eval_accel.md`**.

Do this on a separate branch (`20260613_numba_dp_eval`), not on the
results-for-defence branch.

## Decide whether the SW-CP monotonicity diagnostics stay

**Problem**

`sw_cp.py:845-937` holds ~90 lines of investigation instrumentation inside the
main sliding-window loop, and it runs unconditionally in a time-budgeted solver.
Every iteration calls `st.time_fixed_sol.get_total_tardiness()` and
`get_stage_2_makespan_map()`; every iteration that raises the full objective also
rebuilds a whole `PermutationFlowshopScheduleLite` in incumbent order
(`:906-916`) just to log the last-stage makespan the reorder spent. The
investigation that motivated this is closed — `docs/report/20260912_sw_cp_non_monotone.md`
has the answer, and `refresh_deadline_every_step` has the fix.

**Approach**

Split the block by who still needs it. The `window_violations` check (`:864-893`)
guards an invariant that must hold and is cheap, so keep it unconditional. The
`FULL OBJECTIVE INCREASED` reconstruction is the expensive half and only ever
served the investigation: drop it, or gate it behind a `debug_monotonicity: bool
= False` argument threaded the same way `draw_gantt` already is. Measure one
600 s run before and after so the cost is a number rather than a guess.

## Trim the MCF LB diagnostic log

**Problem**

`single_mc_pmtn.py:113-135` logs an instance characterization on every
`_define_parameters` call. Two costs are hidden in it:

- `max_unit_cost` is computed by scanning `self.c`, which holds `|J| × t_max`
  entries. The dict is built anyway at `:103-110`, but this adds another full
  pass over it purely for one log field.
- `min(slacks)` and `sum(slacks) / n` raise `ValueError` / `ZeroDivisionError`
  when `calJ` is empty, so a constructor that previously could not fail now can.

**Approach**

Derive `max_unit_cost` from `d` and `p` directly — the cost function is
`ceil((t - d_j) / p_j)`, so the maximum is `max(ceil((t_max - d_j) / p_j))` over
jobs, an `O(|J|)` computation instead of `O(|J| · t_max)`. Wrap the whole block
in `if n:` so a zero-job instance logs nothing instead of raising.
