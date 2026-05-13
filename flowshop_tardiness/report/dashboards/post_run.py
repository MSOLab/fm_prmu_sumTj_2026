"""End-to-end driver: turn a run directory into the dashboard artifacts.

Reads ``all_scenarios_summary.csv`` and the per-instance
``<ins>/results/<ins>_obj_log.yaml`` files, then writes three artifacts at
the run root (filenames prefixed with the run timestamp = run directory name):

* ``<run_id>_rpdf_comparison.csv`` — long-format scenario × instance frame
  with ``RPDf``, ``RPDv``, ``Gap``, ``time%``, etc.
* ``<run_id>_rpdf_dashboard.html`` — self-contained PivotTable.js dashboard
  built from the comparison CSV. Default view: rows = ``scenarioName``,
  cols = ``insName``, values = ``RPDf`` (heatmap).
* ``<run_id>_multi_scenario_subroutine_flow_comparison.html`` — Plotly chart
  overlaying each scenario's mean RPDf-over-normalized-time trajectory with
  subroutine-end guide markers.

Invoked at the tail of ``FsMultiScenarioRunner.post_run_process`` so the
artifacts ship alongside the existing Excel report.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .multi_scenario_method_chart import (
    export_multi_scenario_method_rpdf_comparison_html,
)
from .obj_log_loader import (
    build_endpoint_df,
    build_raw_progression_df,
    load_instance_progression,
)
from .rpdf_pivot import (
    PERCENT_AGGREGATORS_JS,
    build_rpdf_comparison_df,
    write_pivot_html,
)

logger = logging.getLogger(__name__)

_OBJ_LOG_FN_FORMAT = "{}_obj_log.yaml"
_RESULT_DIR_NAME = "results"


def _scenario_short_name(scenario_path: str) -> str:
    return scenario_path.rsplit("/", 1)[-1] if scenario_path else scenario_path


def _attach_rpdf(
    df: pd.DataFrame, baseline_obj_by_instance: dict[str, float]
) -> pd.DataFrame:
    """Add ``rpd_f`` column (NaN-dropped on unmatched instances) to ``df``.

    Matches the ``ffc_ddw_sum_et`` baseline-join semantics: unmatched instances
    get logged + dropped; matched rows compute symmetric percentage diff.
    """
    if df.empty:
        return df.assign(rpd_f=pd.Series(dtype=float))

    work = df.copy()
    work["instance_id"] = work["instance_id"].astype(str)
    ref = work["instance_id"].map(baseline_obj_by_instance)
    unmatched = ref.isna()
    if unmatched.any():
        logger.warning(
            "Dropping %d chart rows missing baseline obj (instances=%s)",
            int(unmatched.sum()),
            sorted(set(work.loc[unmatched, "instance_id"])),
        )
        work = work.loc[~unmatched].copy()
        ref = ref.loc[~unmatched]

    obj = work["obj_value"].astype(float).to_numpy()
    ref_arr = ref.astype(float).to_numpy()
    denom = obj + ref_arr
    # Avoid divide-by-zero: when both are zero treat as 0% diff; when sum is
    # zero with opposite signs we end up with NaN which gets dropped later.
    import numpy as np

    with np.errstate(divide="ignore", invalid="ignore"):
        rpdf = np.where(denom == 0, 0.0, 2 * (obj - ref_arr) / denom)
    work["rpd_f"] = rpdf
    return work


def _instance_metadata_lookup(
    summary_df: pd.DataFrame,
) -> dict[tuple[str, str], dict[str, Any]]:
    """``(scenario_path, insName) -> {job_count, stage_count, timelimit}``."""
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in summary_df.iterrows():
        key = (str(row["scenario"]), str(row["insName"]))
        lookup[key] = {
            "job_count": int(row["job_count"]),
            "stage_count": int(row["stage_count"]),
            "timelimit": float(row["timelimit"]),
        }
    return lookup


def _build_baseline_obj_map(
    baseline_df: pd.DataFrame | None,
    *,
    instance_col: str,
    obj_val_col: str,
) -> dict[str, float]:
    if baseline_df is None or baseline_df.empty:
        return {}
    if instance_col not in baseline_df.columns or obj_val_col not in baseline_df.columns:
        logger.warning(
            "Baseline df missing required columns %r / %r; skipping baseline join",
            instance_col,
            obj_val_col,
        )
        return {}
    keep = baseline_df[[instance_col, obj_val_col]].dropna()
    keep[instance_col] = keep[instance_col].astype(str)
    keep = keep.drop_duplicates(subset=[instance_col], keep="first")
    return {
        str(k): float(v)
        for k, v in zip(keep[instance_col], keep[obj_val_col])
        if pd.notna(v)
    }


def _load_scenario_progressions(
    run_dir: Path,
    scenario_path: str,
    summary_df: pd.DataFrame,
    instance_meta: dict[tuple[str, str], dict[str, Any]],
) -> list:
    scenario_df = summary_df[summary_df["scenario"] == scenario_path]
    progressions = []
    for _, row in scenario_df.iterrows():
        ins_name = str(row["insName"])
        meta = instance_meta.get((scenario_path, ins_name))
        if meta is None:
            continue
        ins_dir = run_dir / scenario_path / ins_name
        obj_log_path = (
            ins_dir / _RESULT_DIR_NAME / _OBJ_LOG_FN_FORMAT.format(ins_name)
        )
        if not obj_log_path.exists():
            obj_log_path = ins_dir / _OBJ_LOG_FN_FORMAT.format(ins_name)
        if not obj_log_path.exists():
            logger.info(
                "No obj_log for %s/%s; skipping in flow chart", scenario_path, ins_name
            )
            continue
        try:
            progressions.append(
                load_instance_progression(
                    obj_log_path,
                    instance_id=ins_name,
                    job_cnt=meta["job_count"],
                    stage_cnt=meta["stage_count"],
                    timelimit_sec=meta["timelimit"],
                )
            )
        except Exception as e:
            logger.warning("Failed to load obj_log %s: %s", obj_log_path, e)
    return progressions


def write_post_run_dashboard_artifacts(
    run_dir: Path,
    *,
    summary_csv: Path | None = None,
    baseline_df: pd.DataFrame | None = None,
    baseline_instance_col: str = "Instance",
    baseline_obj_val_col: str = "BKS",
    baseline_obj_bound_col: str = "LB",
    run_id: str | None = None,
) -> dict[str, Path]:
    """Write the comparison CSV + two HTML dashboards under ``run_dir``.

    Returns a mapping of artifact name -> written path. Empty mapping when
    the run summary is missing/empty (no warning beyond a debug log — the
    Excel report path emits its own warning in that case).
    """
    summary_csv = summary_csv or (run_dir / "all_scenarios_summary.csv")
    if not summary_csv.exists():
        logger.info("Dashboards skipped: %s not found", summary_csv)
        return {}

    summary_df = pd.read_csv(summary_csv)
    if summary_df.empty:
        logger.info("Dashboards skipped: %s is empty", summary_csv)
        return {}

    run_id = run_id or run_dir.name
    written: dict[str, Path] = {}

    # ------------------------------------------------------------------
    # 1. RPDf comparison CSV + pivot HTML
    # ------------------------------------------------------------------
    comparison_df = build_rpdf_comparison_df(
        summary_df,
        baseline_df,
        baseline_instance_col=baseline_instance_col,
        baseline_obj_val_col=baseline_obj_val_col,
        baseline_obj_bound_col=baseline_obj_bound_col,
    )

    comparison_csv_path = run_dir / f"{run_id}_rpdf_comparison.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)
    written["rpdf_comparison_csv"] = comparison_csv_path
    logger.info("Wrote %s (%d rows)", comparison_csv_path, len(comparison_df))

    pivot_path = run_dir / f"{run_id}_rpdf_dashboard.html"
    write_pivot_html(
        comparison_df,
        pivot_path,
        initial_state={
            "rows": ["scenarioName"],
            "cols": ["insName"],
            "vals": ["RPDf"],
            "aggregatorName": "Average",
            "rendererName": "Heatmap",
        },
        aggregators_js=PERCENT_AGGREGATORS_JS,
        title=f"RPDf Pivot — {run_id}",
    )
    written["rpdf_dashboard_html"] = pivot_path
    logger.info("Wrote %s", pivot_path)

    # ------------------------------------------------------------------
    # 2. Multi-scenario subroutine flow comparison HTML
    # ------------------------------------------------------------------
    baseline_map = _build_baseline_obj_map(
        baseline_df,
        instance_col=baseline_instance_col,
        obj_val_col=baseline_obj_val_col,
    )
    if not baseline_map:
        # Fall back to using the best obj observed across all scenarios for
        # each instance — keeps the chart useful even without a published BKS.
        logger.info("No baseline obj available; using per-instance best across scenarios")
        baseline_map = {
            str(ins): float(grp["bestObj"].min())
            for ins, grp in summary_df.groupby("insName")
            if grp["bestObj"].notna().any()
        }

    instance_meta = _instance_metadata_lookup(summary_df)
    scenario_paths: list[str] = []
    for sc in summary_df["scenario"].astype(str):
        if sc not in scenario_paths:
            scenario_paths.append(sc)

    scenario_metrics: list[dict[str, Any]] = []
    for scenario_path in scenario_paths:
        progressions = _load_scenario_progressions(
            run_dir, scenario_path, summary_df, instance_meta
        )
        if not progressions:
            continue
        endpoint_df = build_endpoint_df(progressions)
        raw_progression_df = build_raw_progression_df(progressions)
        endpoint_df = _attach_rpdf(endpoint_df, baseline_map)
        raw_progression_df = _attach_rpdf(raw_progression_df, baseline_map)
        if endpoint_df.empty:
            continue
        scenario_metrics.append(
            {
                "label": _scenario_short_name(scenario_path),
                "endpoint_df": endpoint_df,
                "raw_progression_df": raw_progression_df,
            }
        )

    if scenario_metrics:
        flow_path = run_dir / f"{run_id}_multi_scenario_subroutine_flow_comparison.html"
        ok = export_multi_scenario_method_rpdf_comparison_html(
            scenario_metrics=scenario_metrics,
            output_path=flow_path,
        )
        if ok:
            written["multi_scenario_subroutine_flow_comparison_html"] = flow_path
            logger.info("Wrote %s", flow_path)
    else:
        logger.info(
            "Multi-scenario flow chart skipped: no scenarios with usable obj_logs"
        )

    return written
