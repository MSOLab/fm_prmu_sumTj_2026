"""RPDf comparison CSV + PivotTable.js HTML dashboards.

Joins ``all_scenarios_summary.csv`` (one row per scenario × instance) with the
baseline CSV (``naderi2023CPOCodeResult.csv``-style: ``Instance``, ``BKS``,
``LB``) to produce a long-format frame with one row per scenario × instance
carrying ``RPDf``, ``RPDv``, ``Gap``, ``time%`` etc. The frame is then
rendered as a self-contained pivot dashboard.

HTML template adapted from pivottablejs (Nicolas Kruchten, MIT license,
https://github.com/nicolaskruchten/jupyter_pivottablejs). Inlined so we
don't pull in the IPython runtime that pivottablejs imports on load.
Mirrors ``ffc_ddw_sum_et/src/ffc_ddw_sum_et/orchestration/post_run_pivot.py``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from routix.report.subroutine_report_statistics import (
    SubroutineReportStatisticsKeys,
)

logger = logging.getLogger(__name__)


COMPARISON_COLUMNS: tuple[str, ...] = (
    SubroutineReportStatisticsKeys.INSTANCE_NAME,
    "scenarioName",
    "n",
    "c",
    "BKS",
    "LB",
    "bestObj",
    "bestBound",
    "RPDf",
    "RPDv",
    "Gap",
    "elapsedTime",
    "timelimit",
    "time%",
    "foundFeasibleSol",
    "status",
)


def _rpdf(best_obj: float, bks: float) -> float:
    denom = best_obj + bks
    return 0.0 if denom == 0 else 2 * (best_obj - bks) / denom


def _rpdv(best_obj: float, bks: float) -> float:
    if best_obj == 0:
        return 0.0 if bks == 0 else float("nan")
    return (best_obj - bks) / best_obj


def _gap(best_obj: float, lb: float) -> float:
    if best_obj == 0:
        return 0.0 if lb == 0 else float("nan")
    return (best_obj - lb) / best_obj


def _scenario_short_name(scenario_path: str) -> str:
    """``output_cp_lns/20260512_ablation_c1`` -> ``20260512_ablation_c1``."""
    return scenario_path.rsplit("/", 1)[-1] if scenario_path else scenario_path


def build_rpdf_comparison_df(
    summary_df: pd.DataFrame,
    baseline_df: pd.DataFrame | None,
    *,
    baseline_instance_col: str = "Instance",
    baseline_obj_val_col: str = "BKS",
    baseline_obj_bound_col: str = "LB",
) -> pd.DataFrame:
    """Join run summary with the BKS baseline; compute RPDf / RPDv / Gap / time%.

    Rows whose ``bestObj`` is NaN are kept so ``time%`` still shows; their
    RPDf / RPDv / Gap come out NaN. Rows with no matching baseline keep BKS /
    LB as NaN (the resulting RPDf / RPDv / Gap are NaN). ``insName`` and the
    baseline instance column are normalized to string for the join.
    """
    if summary_df.empty:
        return pd.DataFrame(columns=list(COMPARISON_COLUMNS))

    df = summary_df.copy()
    df[SubroutineReportStatisticsKeys.INSTANCE_NAME] = df[
        SubroutineReportStatisticsKeys.INSTANCE_NAME
    ].astype(str)
    df["scenarioName"] = df["scenario"].astype(str).map(_scenario_short_name)
    df["n"] = df["job_count"]
    df["c"] = df["stage_count"]
    df["elapsedTime"] = df["totalElapsedTime"]

    if baseline_df is not None and not baseline_df.empty:
        bdf = baseline_df.copy()
        bdf[baseline_instance_col] = bdf[baseline_instance_col].astype(str)
        keep_cols = [
            baseline_instance_col,
            baseline_obj_val_col,
            baseline_obj_bound_col,
        ]
        bdf = bdf[keep_cols].rename(
            columns={
                baseline_instance_col: SubroutineReportStatisticsKeys.INSTANCE_NAME,
                baseline_obj_val_col: "BKS",
                baseline_obj_bound_col: "LB",
            }
        )
        bdf = bdf.drop_duplicates(
            subset=[SubroutineReportStatisticsKeys.INSTANCE_NAME], keep="first"
        )
        df = df.merge(bdf, on=SubroutineReportStatisticsKeys.INSTANCE_NAME, how="left")
    else:
        df["BKS"] = float("nan")
        df["LB"] = float("nan")

    df["RPDf"] = [
        _rpdf(float(o), float(k)) if pd.notna(o) and pd.notna(k) else float("nan")
        for o, k in zip(df["bestObj"], df["BKS"])
    ]
    df["RPDv"] = [
        _rpdv(float(o), float(k)) if pd.notna(o) and pd.notna(k) else float("nan")
        for o, k in zip(df["bestObj"], df["BKS"])
    ]
    df["Gap"] = [
        _gap(float(o), float(lb)) if pd.notna(o) and pd.notna(lb) else float("nan")
        for o, lb in zip(df["bestObj"], df["LB"])
    ]
    df["time%"] = df["elapsedTime"] / df["timelimit"]

    out = df[list(COMPARISON_COLUMNS)].sort_values(
        [SubroutineReportStatisticsKeys.INSTANCE_NAME, "scenarioName"]
    )
    return out.reset_index(drop=True)


_PIVOT_TEMPLATE = """\
<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>%(title)s</title>

        <link rel="stylesheet" type="text/css" href="https://cdnjs.cloudflare.com/ajax/libs/c3/0.4.11/c3.min.css">
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js"></script>
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/c3/0.4.11/c3.min.js"></script>
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/jquery/1.11.2/jquery.min.js"></script>
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/jqueryui/1.11.4/jquery-ui.min.js"></script>
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/jquery-csv/0.71/jquery.csv-0.71.min.js"></script>

        <link rel="stylesheet" type="text/css" href="https://cdnjs.cloudflare.com/ajax/libs/pivottable/2.19.0/pivot.min.css">
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/pivottable/2.19.0/pivot.min.js"></script>
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/pivottable/2.19.0/d3_renderers.min.js"></script>
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/pivottable/2.19.0/c3_renderers.min.js"></script>
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/pivottable/2.19.0/export_renderers.min.js"></script>

        <style>body {font-family: Verdana;}</style>
    </head>
    <body>
        <script type="text/javascript">
            $(function(){
                %(aggregators_js)s
                $("#output").pivotUI(
                    $.csv.toArrays($("#output").text()),
                    $.extend({
                        renderers: $.extend(
                            $.pivotUtilities.renderers,
                            $.pivotUtilities.c3_renderers,
                            $.pivotUtilities.d3_renderers,
                            $.pivotUtilities.export_renderers
                        ),
                        aggregators: aggregators,
                        hiddenAttributes: [""]
                    }, %(initial_state)s)
                ).show();
            });
        </script>
        <div id="output" style="display: none;">%(csv)s</div>
    </body>
</html>
"""

DEFAULT_AGGREGATORS_JS = "var aggregators = $.pivotUtilities.aggregators;"

PERCENT_AGGREGATORS_JS = """\
var pctFmt = $.pivotUtilities.numberFormat(
                    {digitsAfterDecimal: 1, scaler: 100, suffix: "%"}
                );
                var tpl = $.pivotUtilities.aggregatorTemplates;
                var aggregators = $.extend({}, $.pivotUtilities.aggregators, {
                    "Sum":     tpl.sum(pctFmt),
                    "Average": tpl.average(pctFmt),
                    "Median":  tpl.median(pctFmt),
                    "Minimum": tpl.min(pctFmt),
                    "Maximum": tpl.max(pctFmt),
                    "First":   tpl.first(pctFmt),
                    "Last":    tpl.last(pctFmt)
                });"""


def write_pivot_html(
    df: pd.DataFrame,
    outfile: Path,
    *,
    initial_state: dict,
    aggregators_js: str = DEFAULT_AGGREGATORS_JS,
    title: str = "Pivot",
) -> None:
    """Render ``df`` as a self-contained PivotTable.js HTML at ``outfile``."""
    outfile.parent.mkdir(parents=True, exist_ok=True)
    payload = _PIVOT_TEMPLATE % {
        "title": title,
        "aggregators_js": aggregators_js,
        "initial_state": json.dumps(initial_state),
        "csv": df.to_csv(index=False),
    }
    outfile.write_text(payload, encoding="utf8")
