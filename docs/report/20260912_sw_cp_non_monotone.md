# SW-CP is not monotone across its own iterations

> Written: 2026-09-12
>
> Module: `flowshop_tardiness/controller/sw_cp.py`
>
> Related plans: `plans/20260607_pw_cp_increasing_step_investigation.md`,
> `plans/20260607_pw_cp_refresh_deadline_every_step.md`
>
> Naming: the rename landed in `e9ea6dd`, so this document uses the current
> name (`sw_cp` / SW-CP) throughout, including the log file name that runs
> write today. The plan paths above keep the old name (`pw_cp`) because that is
> what the files on disk are called, and so do the log files from the 2026-06
> runs under `Outputs_scenarios/`. See
> [Pre-rename `pw_cp` outputs are frozen](#pre-rename-pw_cp-outputs-are-frozen)
> for what that means for those run directories.

## Summary

SW-CP is non-increasing **only relative to the incumbent it starts from**. It is
not monotone across its own internal sliding-window iterations, so the
per-iteration `*-sw_cp_obj_log.yaml` ("ObjVal after dispatch") can show small
increases. That is a property of the current design, not a window-bound bug.

## Root cause

Confirmed by a minimal reproduction and a 4/4000 multi-iteration brute force.

The LCT (per-stage latest-completion bound) comes from a right-justified
reference schedule $S^R$ computed **once** from the original incumbent, so it
carries slack relative to already-improved intermediate states.

The CP phase-2 secondary objective minimizes the **sum** of per-stage makespans
$\sum_i C_i$. Within that slack it can therefore pick a batch order that lowers
an earlier stage while raising the last-stage makespan. The higher last-stage
makespan delays the greedily dispatched tail (semi-active and optimal for its
own release), which raises tail tardiness against the previous iteration, while
still staying at or below the starting incumbent.

## What it is not

- The per-stage window bound is respected: greedy ≤ CP solution ≤ LCT, with
  `window_violations=0`.
- `push_back_tail_jobs_keep_tardiness` preserves per-job tardiness (verified
  2000/2000).
- The decode is correct.

## The fix: `refresh_deadline_every_step`

Recompute $S^R$ at each step from the immediately preceding schedule (committed
plus the remainder in incumbent order), right-justify it, and read the LCT from
that.

Brute force puts the current design at 14/8000 increasing trajectories and the
refreshed one at 0/8000, with near-zero change in final quality (better 15 /
worse 11 / equal 7974).

**Current state**: implemented in `sw_cp.py` as the `refresh_deadline_every_step`
argument, defaulting to `False` (the fixed-$S^R$ behavior). Set it to `True` to
get per-iteration monotonicity. Among the experiment configs,
`configs_cp_lns/20260609_ablation_c3_refresh.yaml` and
`20260609_ablation_c4_refresh.yaml` turn it on.

Correction to older notes: an earlier claim that refreshing $S^R$ does not fix
this was wrong. Refresh works precisely because it re-bases the reference on the
current, already-improved tardiness instead of the original incumbent's.

## Pre-rename `pw_cp` outputs are frozen

Decision for the rename: run directories produced before `e9ea6dd` are
read-only history. They stay in `Outputs_scenarios/` and their numbers stay
quotable, but the code no longer reads them back, and no alias is kept for the
old name. Concretely, for a directory whose flow still says `pw_cp`:

- `RESUME` is rejected. The cached subroutine flow is re-validated against
  `FlowshopTardinessCpLnsController`, which no longer has `pw_cp`, so routix
  reports `Method 'pw_cp' not found`. Flows that also carry
  `init_by_neh_ms: true` are rejected the same way, because that argument is now
  `init_method`.
- `POST_PROCESS_ONLY` is not affected by validation (the flow is only validated
  for `FULL_RUN` and `RESUME`), so those directories can still be re-analyzed.
- Their per-iteration notes keep the `pw_cp` label, while
  `SUBROUTINE_SYMBOL_MAP` in `flowshop_tardiness/report/dashboards/_chart_internals.py`
  and the method list in `scripts/analysis_metadata.py` now key on `sw_cp`. The
  old series therefore keep their own `pw_cp` legend entry and fall back to the
  default chart symbol instead of the `sw_cp` circle.
- Timelimit trimming is unaffected: `obj_log_trim` reads only the obj_log
  timestamps, never a method name.

To re-run an old experiment, copy its flow into a new config file, apply the
renames (`pw_cp` → `sw_cp`, `init_by_neh_ms: true` → `init_method: neh-ms`), and
start a fresh run instead of resuming.

## Diagnostics

`_run_loop` reports `full_obj=[committed+tail] window_violations=N` per
iteration and warns on a window-bound violation (which should never fire). On a
full-objective increase it logs the incumbent-order against the CP-order
last-stage makespan against the LCT; that difference is the slack actually
spent.
