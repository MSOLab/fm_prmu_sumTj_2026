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
