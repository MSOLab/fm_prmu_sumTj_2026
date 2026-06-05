# Fix plan: `KeyError: 'instanceName'` in multi-scenario dashboard

## 1. Symptom

End of the 540×2 GAPR / GAPRsu run (`Outputs_scenarios/20260605T104300_928058`):

```
2026-06-05 14:23:31,727 - ERROR - Failed to create dashboard: 'instanceName'
...
  File ".../ga_ctrlr_multi_scenario.py", line 165, in create_dashboard
    best_obj_value_df = raw_summary_df.pivot_table(
        index="instanceName", columns="scenario", values="bestObj")
...
KeyError: 'instanceName'
```

**Impact is cosmetic / post-processing only.** The compute is finished and intact:

- Per-instance results, `multi_instance_summary.csv`, and `all_scenarios_summary.csv` were all written.
- `create_dashboard` is wrapped in `try/except` (`ga_ctrlr_multi_scenario.py:335`) that logs and returns an empty `DataFrame`, so the run continued and the Excel report was still produced (`multi_scenario_report.xlsx`).
- Only the **`BestObjDashboard`** sheet (the pivoted GAPR-vs-GAPRsu comparison) is missing/empty. No need to re-run the 3h40m experiment — see §5.

## 2. Root cause

The dashboard pivots on a column name that the summary writer no longer emits.

Actual aggregated summary schema (`all_scenarios_summary.csv` header from this run):

```
name,job_count,stage_count,timelimit,insName,foundFeasibleSol,totalElapsedTime,
firstBound,bestObj,bestBound,improvementRatio,methodCallCounts,reportCount,
initObj,status,scenario
```

- The instance identifier column is **`insName`** (emitted by the report statistics;
  `name` is the duplicate from `FsInputSummary`, `flowshop_tardiness/fs_input_summary.py:18`
  → header `"name,job_count,stage_count,timelimit"`).
- There is **no `instanceName` column**. That key is **legacy** — it was renamed to
  `insName` everywhere else in the reporting code but `ga_ctrlr_multi_scenario.py`
  (and its twin `tbb_2018_mhx1_multi_scenario.py`) were missed.

Evidence that `insName` is the current canonical key (already used consistently elsewhere):

- `flowshop_tardiness/report/dashboards/rpdf_pivot.py:27,87,103,108-109,129` — pivots/merges on `insName`.
- `flowshop_tardiness/report/dashboards/post_run.py:99,102,148` — keys on `insName`.

So `raw_summary_df.pivot_table(index="instanceName", ...)` raises `KeyError` because the
DataFrame's instance column is `insName`.

## 3. Affected code (stale `"instanceName"` literals)

All occurrences are inside `create_dashboard` (+ one in `write_excel_report`) of
`ga_ctrlr_multi_scenario.py`; they all refer to the same single instance-id column:

| Line | Context |
|---|---|
| 166 | `pivot_table(index="instanceName", ...)` — **the throw site** |
| 172 | baseline rename map target: `self.baseline_instance_col: "instanceName"` |
| 187-188 | `best_obj_value_df["instanceName"].astype(str)` |
| 202 | `pd.merge(..., on="instanceName", ...)` |
| 221 | `scenarios = [col ... if col != "instanceName"]` |
| 279 | `ordered_columns = ["instanceName"]` |
| 321 | summary row key: `{"instanceName": stat_name}` |
| 323 | `if col != "instanceName":` |
| 414 | Excel header match: `elif col == "instanceName": header.append(("", "insId"))` |

Twin file with the identical bug (not exercised by this run, but will fail the same way):
`tbb_2018_mhx1_multi_scenario.py:166,172,187-188,202,221,279,321,323,414`.

## 4. Fix — replace `"instanceName"` with `SubroutineReportStatisticsKeys.INSTANCE_NAME`

Use the canonical key constant from routix — the single source of truth that the summary
writer itself uses — instead of a hardcoded string literal.

**Step 1.** Add the import at the top of `ga_ctrlr_multi_scenario.py`:

```python
from routix.report.subroutine_report_statistics import (
    SubroutineReportStatisticsKeys,
)
```

**Step 2.** Replace every `"instanceName"` string literal in `create_dashboard` /
`write_excel_report` (the 9 sites in §3) with
`SubroutineReportStatisticsKeys.INSTANCE_NAME`.

- **Verified:** `SubroutineReportStatisticsKeys.INSTANCE_NAME == "insName"` (a plain
  `str` constant — no `.value`). Because it *is* a `str`, it works unchanged as the
  `pivot_table(index=...)` key, the `merge(on=...)` key, a dict key
  (`{SubroutineReportStatisticsKeys.INSTANCE_NAME: stat_name}`), and in
  `col == SubroutineReportStatisticsKeys.INSTANCE_NAME` comparisons. After
  `pivot_table(...).reset_index()` the resulting column label is `"insName"`, so *all*
  references must use the same constant — replace every site, not just the throw site
  at line 166.
- Line 414 branch becomes `elif col == SubroutineReportStatisticsKeys.INSTANCE_NAME:`.
  The displayed header label `("", "insId")` is cosmetic and may stay as-is.
- Apply the identical change (import + 9 replacements) to the twin file
  `tbb_2018_mhx1_multi_scenario.py` so the Taillard-2018 scenario runner doesn't hit
  the same `KeyError` later.

Why the constant rather than a bare `"insName"` literal: it binds the dashboard to the
exact key the writer emits (`SubroutineReportStatisticsKeys`), so any future column
rename propagates automatically instead of silently re-breaking the pivot.

> Do **not** substitute the `name` column instead: although `name` carries the same
> value, `insName` (= `SubroutineReportStatisticsKeys.INSTANCE_NAME`) is the key the rest
> of the reporting pipeline (`rpdf_pivot.py`, `post_run.py`) standardizes on.

## 5. Verification (no recompute needed)

The expensive run is already on disk; regenerate only the report:

1. **Re-run in POST_PROCESS_ONLY mode** against the existing output, so it rebuilds the
   report from the per-instance summary files without re-solving. In the metadata file
   that `ga_ctrlr_main.MAIN_METADATA_FILENAME` points to, set
   `analysis_timestamp: "20260605T104300_928058"` (or `analysis_dir_path` to the same
   dir), then `uv run python ga_ctrlr_main.py`. POST_PROCESS_ONLY re-loads the metadata
   that was dumped *inside* that run directory (`ga_ctrlr_main.py:95`), so only the
   timestamp/dir pointer matters here.
   (`fs_config.MainMetadata.analysis_timestamp` / `analysis_dir_path` drive
   `RunMode.POST_PROCESS_ONLY` in `ga_ctrlr_main.py:247-263`.)
   - Confirm log no longer shows `Failed to create dashboard`.
   - Open `multi_scenario_report.xlsx` → `BestObjDashboard` sheet is populated with one
     row per instance (1..540), one column per scenario (`...gapr_none/20260120`,
     `...gapr_vr2010/20260120`), plus baseline/RPD/Gap columns and the stat rows.
   - Revert the temporary `analysis_timestamp` afterwards.
2. **(Optional) regression test** mirroring `tests/test_dashboard_import_isolation.py`:
   build a tiny `raw_summary_df` with columns `insName, scenario, bestObj` and assert
   `create_dashboard(df)` returns a non-empty frame whose index/first column is the
   instance id — would have caught this rename drift.

## 6. Scope & risk

- Pure string-literal rename inside reporting code; no change to solver, evaluators,
  configs, or the summary-writing schema. Low risk.
- The two scenario-runner files are near-duplicates — fixing both keeps them in sync and
  avoids the same error surfacing in the Taillard-2018 pipeline.
- Existing run artifacts are untouched and reusable for the POST_PROCESS_ONLY rebuild.
