"""End-to-end driver: turn a run directory into the dashboard artifacts.

Reads ``all_scenarios_summary.csv`` and the per-instance
``<ins>/results/<ins>_obj_log.yaml`` files, then writes:

* ``<run_id>_rpdf_comparison.csv`` — long-format scenario × instance frame
  with ``RPDf``, ``RPDv``, ``Gap``, ``time%``, etc.
* ``<run_id>_rpdf_dashboard.html`` — PivotTable.js dashboard built from the
  comparison CSV. Default view: rows = ``(scenarioName, c)``, cols = ``n``,
  vals = ``RPDf`` (heatmap).
* ``<run_id>_multi_scenario_subroutine_flow_comparison.html`` — Plotly chart
  overlaying each scenario's mean RPDf-over-normalized-time trajectory with
  subroutine-end guide markers.
* ``<scenario_dir>/summary_method_rpdf_and_norm_time_scatter.html`` — one
  per scenario, interactive per-instance / per-(n,c)-mean detail chart.

Invoked at the tail of ``FsMultiScenarioRunner.post_run_process``. All inputs
are read from disk so POST_PROCESS_ONLY runs (and the standalone
``scripts/generate_dashboards.py`` CLI) reproduce every artifact.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from routix.report.subroutine_report_statistics import (
    SubroutineReportStatisticsKeys,
)

from flowshop_tardiness.io_solution import OBJ_LOG_FN_FORMAT, RESULT_DIR_NAME

from .method_mean_scatter import (
    export_method_mean_scatter_html,
    load_method_mean_metrics,
)
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
from .rpdf_scatter_chart import export_method_rpdf_scatter_html

logger = logging.getLogger(__name__)

_SCENARIO_SCATTER_FN = "summary_method_rpdf_and_norm_time_scatter.html"
_SCENARIO_METHOD_MEAN_FN = "summary_method_mean_rpdf_and_mean_norm_time_scatter.html"
_METHOD_END_SUMMARY_FN = "summary_method_end_time_and_obj_value.csv"


def _scenario_short_name(scenario_path: str) -> str:
    return scenario_path.rsplit("/", 1)[-1] if scenario_path else scenario_path


def _attach_rpdf_and_dims(
    df: pd.DataFrame,
    baseline_obj_by_instance: dict[str, float],
) -> pd.DataFrame:
    """Add ``rpd_f`` column (and keep ``n``, ``c`` already attached upstream).

    Drops rows whose ``instance_id`` is missing from the baseline map.
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
        key = (
            str(row["scenario"]),
            str(row[SubroutineReportStatisticsKeys.INSTANCE_NAME]),
        )
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
    if (
        instance_col not in baseline_df.columns
        or obj_val_col not in baseline_df.columns
    ):
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
        ins_name = str(row[SubroutineReportStatisticsKeys.INSTANCE_NAME])
        meta = instance_meta.get((scenario_path, ins_name))
        if meta is None:
            continue
        ins_dir = run_dir / scenario_path / ins_name
        obj_log_path = ins_dir / RESULT_DIR_NAME / OBJ_LOG_FN_FORMAT.format(ins_name)
        if not obj_log_path.exists():
            obj_log_path = ins_dir / OBJ_LOG_FN_FORMAT.format(ins_name)
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


def _resolve_baseline_df(
    baseline_df: pd.DataFrame | None,
    baseline_csv_path: Path | None,
) -> pd.DataFrame | None:
    if baseline_df is not None:
        return baseline_df
    if baseline_csv_path is None:
        return None
    if not Path(baseline_csv_path).exists():
        logger.warning("Baseline CSV not found at %s", baseline_csv_path)
        return None
    return pd.read_csv(baseline_csv_path)


def write_post_run_dashboard_artifacts(
    run_dir: Path,
    *,
    summary_csv: Path | None = None,
    baseline_df: pd.DataFrame | None = None,
    baseline_csv_path: Path | None = None,
    baseline_instance_col: str = "Instance",
    baseline_obj_val_col: str = "BKS",
    baseline_obj_bound_col: str = "LB",
    run_id: str | None = None,
    scenario_output_root: Path | None = None,
) -> dict[str, Path]:
    """Write the comparison CSV + run-level HTML dashboards + per-scenario
    scatter HTMLs under ``run_dir``.

    Pass either ``baseline_df`` (already loaded) or ``baseline_csv_path``
    (loaded lazily). Both omitted is fine — the flow chart then uses
    per-instance best-across-scenarios as the reference, and the comparison
    CSV's BKS / RPDf columns come out NaN.

    ``scenario_output_root`` defaults to ``run_dir`` (current behaviour).
    When set, per-scenario scatter HTMLs are written under that root
    using the scenario basename, which prevents writes from following
    symlinks back into the original run directory.

    Returns a mapping of artifact name -> written path.
    """
    summary_csv = summary_csv or (run_dir / "all_scenarios_summary.csv")
    if not summary_csv.exists():
        logger.info("Dashboards skipped: %s not found", summary_csv)
        return {}

    summary_df = pd.read_csv(summary_csv)
    if summary_df.empty:
        logger.info("Dashboards skipped: %s is empty", summary_csv)
        return {}

    baseline_df = _resolve_baseline_df(baseline_df, baseline_csv_path)
    run_id = run_id or run_dir.name
    written: dict[str, Path] = {}

    # ------------------------------------------------------------------
    # 1. RPDf comparison CSV + pivot dashboard
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
            "rows": ["scenarioName", "c"],
            "cols": ["n"],
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
    # 2. Per-scenario obj_log → endpoint/progression frames
    # ------------------------------------------------------------------
    baseline_map = _build_baseline_obj_map(
        baseline_df,
        instance_col=baseline_instance_col,
        obj_val_col=baseline_obj_val_col,
    )
    if not baseline_map:
        logger.info(
            "No baseline obj available; using per-instance best across scenarios"
        )
        baseline_map = {
            str(ins): float(grp["bestObj"].min())
            for ins, grp in summary_df.groupby(
                SubroutineReportStatisticsKeys.INSTANCE_NAME
            )
            if grp["bestObj"].notna().any()
        }

    instance_meta = _instance_metadata_lookup(summary_df)
    scenario_paths: list[str] = []
    for sc in summary_df["scenario"].astype(str):
        if sc not in scenario_paths:
            scenario_paths.append(sc)

    _scenario_write_root = scenario_output_root or run_dir

    scenario_frames: list[dict[str, Any]] = []
    for scenario_path in scenario_paths:
        progressions = _load_scenario_progressions(
            run_dir, scenario_path, summary_df, instance_meta
        )
        if not progressions:
            continue
        endpoint_df = build_endpoint_df(progressions)
        raw_progression_df = build_raw_progression_df(progressions)

        # Attach n / c onto chart frames so the scatter writer can group by
        # problem size. job_cnt/stage_cnt come straight from the loader.
        for df in (endpoint_df, raw_progression_df):
            if not df.empty:
                df["n"] = df["job_cnt"].astype(int)
                df["c"] = df["stage_cnt"].astype(int)

        endpoint_df = _attach_rpdf_and_dims(endpoint_df, baseline_map)
        raw_progression_df = _attach_rpdf_and_dims(raw_progression_df, baseline_map)
        if endpoint_df.empty:
            continue
        scenario_frames.append(
            {
                "scenario_path": scenario_path,
                "label": _scenario_short_name(scenario_path),
                "endpoint_df": endpoint_df,
                "raw_progression_df": raw_progression_df,
            }
        )

    # ------------------------------------------------------------------
    # 3. Run-level multi-scenario flow comparison HTML
    # ------------------------------------------------------------------
    if scenario_frames:
        flow_path = run_dir / f"{run_id}_multi_scenario_subroutine_flow_comparison.html"
        ok = export_multi_scenario_method_rpdf_comparison_html(
            scenario_metrics=[
                {
                    "label": f["label"],
                    "endpoint_df": f["endpoint_df"],
                    "raw_progression_df": f["raw_progression_df"],
                }
                for f in scenario_frames
            ],
            output_path=flow_path,
        )
        if ok:
            written["multi_scenario_subroutine_flow_comparison_html"] = flow_path
            logger.info("Wrote %s", flow_path)

        # ------------------------------------------------------------------
        # 4. Per-scenario scatter HTMLs
        # ------------------------------------------------------------------
        for frame in scenario_frames:
            scatter_path = (
                _scenario_write_root / frame["scenario_path"] / _SCENARIO_SCATTER_FN
            )
            ok = export_method_rpdf_scatter_html(
                endpoint_df=frame["endpoint_df"],
                raw_progression_df=frame["raw_progression_df"],
                output_path=scatter_path,
            )
            if ok:
                written[f"scatter:{frame['label']}"] = scatter_path
                logger.info("Wrote %s", scatter_path)
    else:
        logger.info("Subroutine charts skipped: no scenarios with usable obj_logs")

    # ------------------------------------------------------------------
    # 5. Per-method mean (time%, RPDf) scatter — per scenario + run level
    # ------------------------------------------------------------------
    method_mean_scenarios: list[dict[str, Any]] = []
    for scenario_path in scenario_paths:
        csv_path = run_dir / scenario_path / _METHOD_END_SUMMARY_FN
        if not csv_path.exists():
            logger.info(
                "Skipping method-mean chart for %s: %s not found",
                scenario_path,
                csv_path.name,
            )
            continue
        timelimit_by_instance = {
            ins: float(meta["timelimit"])
            for (sc, ins), meta in instance_meta.items()
            if sc == scenario_path
        }
        method_points = load_method_mean_metrics(
            csv_path,
            timelimit_by_instance=timelimit_by_instance,
            baseline_obj_by_instance=baseline_map,
        )
        if not method_points:
            logger.info(
                "No method-mean points for %s (no instances with both obj and BKS)",
                scenario_path,
            )
            continue
        label = _scenario_short_name(scenario_path)
        scenario_entry = {"label": label, "method_points": method_points}
        per_scenario_path = (
            _scenario_write_root / scenario_path / _SCENARIO_METHOD_MEAN_FN
        )
        if export_method_mean_scatter_html(
            [scenario_entry],
            per_scenario_path,
            title=f"Method mean RPDf vs mean Time% — {label}",
        ):
            written[f"method_mean_scatter:{label}"] = per_scenario_path
            logger.info("Wrote %s", per_scenario_path)
        method_mean_scenarios.append(scenario_entry)

    if method_mean_scenarios:
        run_level_path = (
            run_dir
            / f"{run_id}_multi_scenario_method_mean_rpdf_and_mean_norm_time_scatter.html"
        )
        if export_method_mean_scatter_html(
            method_mean_scenarios,
            run_level_path,
            title=f"Method mean RPDf vs mean Time% — {run_id}",
        ):
            written["multi_scenario_method_mean_scatter_html"] = run_level_path
            logger.info("Wrote %s", run_level_path)

    return written
