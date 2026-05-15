# Cross-run analysis aggregator (`analysis/<ts>/`)

Approach **(A)**: small refactor of `FsMultiScenarioRunner.write_excel_report` so
it can be called standalone, plus a new CLI under `scripts/` that gathers N
scenario folders (from any number of parent run dirs) into a timestamped
`analysis/` sub-dir and runs the existing dashboard pipeline.

---

## 1. Goal

Given a list of scenario directories — e.g.

- `Outputs_scenarios/20260513T142520_492897/20260512_ablation_c1`
- `Outputs_scenarios/20260513T142520_492897/20260512_ablation_c2`
- `Outputs_scenarios/20260515T024302_304232/20260515_ablation_c3`
- `Outputs_scenarios/20260513T142520_492897/20260512_ablation_c4`
- `Outputs_scenarios/20260515T024302_304232/20260515_ablation_c5`

…produce inside `analysis/<YYYYMMDDTHHmmss_xxxxxx>/`:

| Artifact | Source |
| --- | --- |
| `all_scenarios_summary.csv`, `all_scenarios_summary_endpoint.csv` | aggregated from each scenario's `multi_instance_summary.csv`, with `apply_timelimit_trim` applied |
| `multi_scenario_report.xlsx` | refactored `write_excel_report_standalone` |
| `<ts>_rpdf_comparison.csv`, `<ts>_rpdf_dashboard.html` | `write_post_run_dashboard_artifacts` |
| `<ts>_multi_scenario_subroutine_flow_comparison.html` | same |
| `<ts>_multi_scenario_method_mean_rpdf_and_mean_norm_time_scatter.html` | same |
| `<scenario_link>/summary_method_*_scatter.html` (per scenario) | same |
| `analysis_manifest.yaml` | what scenarios were included, source paths, baseline, timestamp |

`<ts>` mirrors the existing convention: `datetime.now().strftime("%Y%m%dT%H%M%S_%f")` (microseconds for collision avoidance, like the existing run dirs do).

---

## 2. Inputs to the CLI

`scripts/aggregate_analysis.py`:

```sh
uv run python scripts/aggregate_analysis.py \
    --scenario Outputs_scenarios/20260513T142520_492897/20260512_ablation_c1 \
    --scenario Outputs_scenarios/20260513T142520_492897/20260512_ablation_c2 \
    --scenario Outputs_scenarios/20260515T024302_304232/20260515_ablation_c3 \
    --scenario Outputs_scenarios/20260513T142520_492897/20260512_ablation_c4 \
    --scenario Outputs_scenarios/20260515T024302_304232/20260515_ablation_c5 \
    [--baseline resources/vrm_ref/naderi2023CPOCodeResult.csv]
    [--label ablation_c1_to_c5]
    [--out-root analysis]
    [--link {symlink|copy}]   # default: symlink
```

- **Required**: `--scenario` (≥1, repeatable). Each path must be a directory
  containing `multi_instance_summary.csv`.
- **Optional `--baseline`**: explicit baseline CSV. If omitted, auto-discover:
  read each scenario's parent dir's `metadata_*.yaml` and pick
  `baseline_csv_path`. Warn if scenarios disagree; use the first one (same dir
  rule already used by `scripts/generate_dashboards.py:31`).
- **`--label`**: optional human slug appended to the output dir name. Default:
  joined scenario basenames (truncated to ~40 chars).
- **`--out-root`**: defaults to repo-root `analysis/`. Created if missing.
- **`--link`**: `symlink` (default, fast, preserves originals) or `copy` (for
  exporting/archiving). Symlink target = absolute path of the source scenario
  dir.

---

## 3. Output directory layout

```plaintext
analysis/
└── 20260515T143012_481923_ablation_c1_to_c5/
    ├── 20260512_ablation_c1 -> /abs/path/.../20260513T142520_492897/20260512_ablation_c1
    ├── 20260512_ablation_c2 -> /abs/path/.../20260513T142520_492897/20260512_ablation_c2
    ├── 20260515_ablation_c3 -> /abs/path/.../20260515T024302_304232/20260515_ablation_c3
    ├── 20260512_ablation_c4 -> /abs/path/.../20260513T142520_492897/20260512_ablation_c4
    ├── 20260515_ablation_c5 -> /abs/path/.../20260515T024302_304232/20260515_ablation_c5
    ├── all_scenarios_summary.csv
    ├── all_scenarios_summary_endpoint.csv
    ├── multi_scenario_report.xlsx
    ├── 20260515T143012_481923_rpdf_comparison.csv
    ├── 20260515T143012_481923_rpdf_dashboard.html
    ├── 20260515T143012_481923_multi_scenario_subroutine_flow_comparison.html
    ├── 20260515T143012_481923_multi_scenario_method_mean_rpdf_and_mean_norm_time_scatter.html
    └── analysis_manifest.yaml
```

Scenario sub-dir name = basename of the source path (matches what the
runner originally stored as `output_subdir`). If two sources have the
**same basename** (e.g. two runs both named `20260512_ablation_c3`), the
CLI must error early and ask the user to rename or exclude — the dashboard
generator uses scenario name as a primary key everywhere.

`add_no_clobber`: if the analysis dir already exists (race), regenerate the
timestamp once and retry.

---

## 4. Code changes

### 4.0 Dependency isolation (critical)

`fs_multi_scenario_runner.py` transitively imports CP-SAT via
`fs_multi_instance_runner` → `fs_single_instance_runner` → `mbls.cpsat`, and
`flowshop_tardiness/report/__init__.py` exports `FsCpsatSolverReport` which
also pulls `mbls.cpsat`. Any module that touches these import paths will fail
to load on a machine without CP-SAT/CPLEX/docplex installed (or just slow
down imports significantly).

**Solver-free zone** (already established):
`flowshop_tardiness/report/dashboards/` — every submodule uses only pandas /
numpy / yaml / plotly / xlsxwriter. `scripts/generate_dashboards.py:104`
already imports from this subpackage path (not the parent `flowshop_tardiness.report`),
proving the isolation works.

**Rule for this work**: all new aggregator helpers live under
`flowshop_tardiness/report/dashboards/` (or a new sibling sub-package, e.g.
`flowshop_tardiness/report/aggregate/`). The CLI imports **only** from
`flowshop_tardiness.report.dashboards.*` / `flowshop_tardiness.report.aggregate.*`,
never from `fs_multi_scenario_runner`, never from `flowshop_tardiness.report`
(parent package). The `FsMultiScenarioRunner` shims (4.1 below) import from
the new location too, so the helper code has a single home.

### 4.1 Extract Excel/dashboard helpers into the solver-free zone

**New file**: `flowshop_tardiness/report/dashboards/multi_scenario_report.py`
(or split into `excel_writer.py` + `dashboard_builder.py` + `info_sheet.py` if it grows past ~300 LOC).

Currently:

- `write_excel_report` (`fs_multi_scenario_runner.py:382`) — depends on
  `self` only for `self.relative_percentage_*_col_prefix` constants
  (lines 413-449).
- `create_dashboard` (`fs_multi_scenario_runner.py:186`) — depends on
  `self.baseline_df`, `self.baseline_*_col`, `self.base_output_metadata`,
  `self.stat_name_func_pairs`, and the three `*_col_format` class attrs.
- `create_info_sheet` (`fs_multi_scenario_runner.py:367`) — depends on
  `self.scenario_configs`.

**Refactor**:

1. Move bodies to new module-level functions in
   `flowshop_tardiness/report/dashboards/multi_scenario_report.py`:
   - `write_multi_scenario_excel_report(path, *, dashboard_df, raw_summary_df, info_df, baseline_df, rpd_col_prefixes=DEFAULT_RPD_PREFIXES)`
   - `build_dashboard_df(raw_summary_df, *, baseline_df, baseline_cols, base_output_metadata, stat_pairs, rpd_col_formats)`
   - `build_info_df(scenario_records)` — `scenario_records: list[dict]` of `{Scenario, Subroutine Flow, Stopping Criteria, Description}`
   - Module-level constants for default prefixes / formats / stat pairs.
2. `FsMultiScenarioRunner.write_excel_report` / `create_dashboard` /
   `create_info_sheet` become one-line shims that bind class attrs and call
   the module functions. **Zero behavior change** for existing callers;
   existing tests in `tests/` should pass untouched.
3. The CLI builds `scenario_records` by reading each scenario dir's saved
   `subroutine_flow.yaml` + `stopping_criteria.yaml` (constants from
   `output_filenames.py:OutputFilenames`). If `description` isn't on disk,
   pull from the source-run's `metadata_*.yaml` `dicts_of_i_o_data_path`
   entry by matching `output_dir`. (`output_filenames` is solver-free.)
4. **Verify zero solver imports**: after the refactor, run
   `python -c "import flowshop_tardiness.report.dashboards.multi_scenario_report"`
   in a venv without `mbls`/`docplex` to confirm. Add a smoke test
   `tests/test_dashboard_import_isolation.py` that asserts neither
   `mbls.cpsat` nor `docplex` appears in `sys.modules` after importing the
   dashboards subpackage.

### 4.2 Extract aggregation into a reusable function

**File**: new `flowshop_tardiness/report/dashboards/aggregation.py`
(co-located with the dashboards subpackage so import remains solver-free —
**do not** place under `flowshop_tardiness/report/` directly, since that
package's `__init__.py` pulls in `mbls.cpsat`).

```python
def aggregate_scenario_summaries(
    scenario_dirs: Sequence[Path],
    *,
    out_dir: Path,
) -> pd.DataFrame:
    """Read each scenario's multi_instance_summary.csv, add `scenario` col
    (= source basename), concat, save *_endpoint.csv, apply timelimit trim,
    save all_scenarios_summary.csv. Returns the trimmed concat frame."""
```

This mirrors `FsMultiScenarioRunner.post_run_process` lines 100-132.
Could be invoked from `post_run_process` itself in a later cleanup, but **not
in scope here** — keep the diff minimal.

### 4.3 New CLI

**File**: new `scripts/aggregate_analysis.py`

Structure:

```python
def main():
    args = parse_args()                           # --scenario (≥1), --baseline, ...
    scenario_paths = [resolve_and_validate(p) for p in args.scenario]
    check_basename_uniqueness(scenario_paths)     # error early

    ts = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    analysis_dir = args.out_root / f"{ts}_{args.label or default_label(scenario_paths)}"
    analysis_dir.mkdir(parents=True)

    materialize_scenarios(scenario_paths, analysis_dir, mode=args.link)
                                                   # symlink or copytree each dir
                                                   # under analysis_dir/<basename>

    summary_df = aggregate_scenario_summaries(scenario_paths, out_dir=analysis_dir)

    baseline_df, baseline_cols = resolve_baseline(args.baseline, scenario_paths)
    info_df = build_info_df_from_disk(scenario_paths)   # uses 4.1 (4) helper

    dashboard_df = build_dashboard_df(
        summary_df,
        baseline_df=baseline_df,
        baseline_cols=baseline_cols,
        base_output_metadata={},
        stat_pairs=DEFAULT_STAT_PAIRS,           # solver-free constant from dashboards.multi_scenario_report
        rpd_col_formats=DEFAULT_RPD_FORMATS,
    )

    write_multi_scenario_excel_report(
        analysis_dir / "multi_scenario_report.xlsx",
        dashboard_df=dashboard_df,
        raw_summary_df=summary_df,
        info_df=info_df,
        baseline_df=baseline_df,
    )

    write_post_run_dashboard_artifacts(
        analysis_dir,
        baseline_df=baseline_df,
        baseline_instance_col=baseline_cols.instance,
        baseline_obj_val_col=baseline_cols.obj_val,
        baseline_obj_bound_col=baseline_cols.obj_bound,
        run_id=ts,
    )

    write_manifest(analysis_dir, scenario_paths, baseline_path, args.label, ts)
```

`write_manifest` dumps a small YAML (sources, baseline, timestamp, link
mode, CLI argv) so the analysis dir is self-describing for future re-runs.

### 4.4 Baseline resolution

Order of precedence:

1. `--baseline` CLI flag (absolute or repo-relative).
2. Each scenario's parent dir → glob `metadata_*.yaml` (single match expected
   — same heuristic as `scripts/generate_dashboards.py:31`). Read
   `baseline_csv_path`. Collect unique values across the 5 scenarios.
3. If all 5 agree → use it. If they disagree → log all candidates and exit
   non-zero with a clear message ("pass `--baseline` explicitly").
4. If no metadata found → proceed without baseline (matches the existing
   "no baseline" path in `write_post_run_dashboard_artifacts`).

`baseline_column_mapping` is read from the same metadata YAML (default to
`BaselineColumnMapping()` if absent — `fs_config.py:59`).

---

## 5. Edge cases / decisions

- **Symlinks vs copy**: default symlink. Pro: instant; the dashboard
  generator only reads files, never writes inside the scenario dirs **except**
  for `summary_method_*_scatter.html` (per-scenario scatter HTML — see
  `post_run.py:336`). That write would land back in the original run's
  scenario folder if we symlink. **Decision**: write per-scenario scatter
  HTMLs to `analysis_dir/<basename>/` instead. Requires a small change in
  `post_run.py:336` — pass an override base dir, or change the symlink to a
  per-dir bind that lets us redirect just that file. Cleanest fix: add a
  `scenario_output_root: Path | None = None` kwarg to
  `write_post_run_dashboard_artifacts` that defaults to `run_dir` (= current
  behavior). When set, per-scenario scatter HTMLs are written under that root
  using basename. **This is the only behavioral knob added to existing
  code.**

- **Basename collisions**: hard error with a message listing both source
  paths. The user must rename their experiments or exclude one.

- **Missing `multi_instance_summary.csv`**: hard error per scenario (the
  scenario is unusable downstream anyway).

- **Mixed `timelimit`**: `apply_timelimit_trim` already handles per-row
  timelimits, so heterogeneous timelimits across scenarios are fine — but
  the dashboard reader assumes the summary frame is internally consistent
  per `(scenario, insName)`, which it is.

- **`analysis/` and `.gitignore`**: add `analysis/` to `.gitignore` so
  generated artifacts don't pollute the repo.

---

## 6. Testing

- **Unit**: keep existing tests passing (the method shims preserve the
  current public surface). Add a small test for
  `aggregate_scenario_summaries` over a synthetic 2-scenario fixture.
- **Smoke**: run the CLI against the 5 ablation scenarios from the prompt.
  Verify: (a) xlsx opens, BestObjDashboard has 5 scenario columns, Scenario_Info has 5 rows with correct descriptions; (b) all 3 run-level HTMLs render; (c) 5 per-scenario scatter HTMLs land **inside `analysis_dir/<basename>/`**, not in the original `Outputs_scenarios/...` tree.
- **Regression**: run `scripts/generate_dashboards.py` on `Outputs_scenarios/20260513T142520_492897/` afterwards; output bytes should be byte-identical to a pre-refactor run (the method shims should not change behavior).

---

## 7. Order of work

1. **Verify dependency isolation baseline**: write the smoke test
   `tests/test_dashboard_import_isolation.py` first (red — confirm it fails
   if any solver lib leaks in). This locks in the invariant before we add
   new code under `dashboards/`.
2. Refactor: extract `write_multi_scenario_excel_report`,
   `build_dashboard_df`, `build_info_df` to
   `flowshop_tardiness/report/dashboards/multi_scenario_report.py`. Keep
   method shims on `FsMultiScenarioRunner`. Re-run isolation test (still
   green) + existing tests.
3. Add `scenario_output_root` kwarg to `write_post_run_dashboard_artifacts`
   (`post_run.py:190`). Default preserves behavior.
4. Add `aggregate_scenario_summaries` helper in
   `flowshop_tardiness/report/dashboards/aggregation.py`.
5. Write `scripts/aggregate_analysis.py` CLI. Import only from
   `flowshop_tardiness.report.dashboards.*`. Add a CLI-level smoke test
   that runs it in a subprocess with `PYTHONPATH` set so we'd see solver
   import errors immediately if they regress.
6. Add `analysis/` to `.gitignore`.
7. Smoke test against the 5 ablation dirs. Sanity-check xlsx + HTMLs.

Estimated diff size: ~250 LOC new (mostly the CLI), ~50 LOC of low-risk
refactor with no behavior change.
