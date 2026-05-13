"""Per-controller-method mean (time%, RPDf) scatter charts.

Each top-level method in ``subroutine_flow.yaml`` becomes a single point per
scenario: arithmetic mean of ``method_end_sec / timelimit`` (x) and
``2 * (obj - BKS) / (obj + BKS)`` (y) across the instances that actually
recorded an ``objective_value`` for that method. Methods with no
``objective_value`` recorded for any instance (e.g. ``set_random_seed``)
are dropped.

Distinct from ``multi_scenario_method_chart`` (best-so-far step-function
trajectory from ``<ins>_obj_log.yaml``) and ``rpdf_scatter_chart``
(per-instance / per-(n, c) subroutine markers) — neither of those shows the
very first method endpoints because they wait for all instances to report
before drawing the mean line.

Data source is ``<scenario>/summary_method_end_time_and_obj_value.csv``,
written by ``scripts/process_logs.py::process_scenario`` at the tail of
``FsMultiInstanceRunner.post_run_process`` in every run mode.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from string import Template
from typing import Any

import pandas as pd

from ._chart_internals import series_colors_json, symbol_map_json

logger = logging.getLogger(__name__)

_END_SEC_SUFFIX = "_end_sec"
_OBJ_VALUE_SUFFIX = "_obj_value"

_EMPTY_POSITIVE_AXIS_UPPER = 0.01
_POSITIVE_AXIS_PADDING = 1.05
_MIN_NORMALIZED_TIME_X_UPPER = 1.0


def _discover_methods(columns: list[str]) -> list[str]:
    """Return method names in CSV column order. A method is recognized only
    when both ``<name>_end_sec`` and ``<name>_obj_value`` columns are present.
    """
    column_set = set(columns)
    methods: list[str] = []
    seen: set[str] = set()
    for col in columns:
        if not col.endswith(_END_SEC_SUFFIX):
            continue
        method = col[: -len(_END_SEC_SUFFIX)]
        if method in seen:
            continue
        if f"{method}{_OBJ_VALUE_SUFFIX}" not in column_set:
            continue
        methods.append(method)
        seen.add(method)
    return methods


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    return v


def _rpd_f(obj: float, bks: float) -> float:
    denom = obj + bks
    if denom == 0:
        return 0.0
    return 2.0 * (obj - bks) / denom


def load_method_mean_metrics(
    summary_csv_path: Path,
    *,
    timelimit_by_instance: dict[str, float],
    baseline_obj_by_instance: dict[str, float],
    drop_non_improving_methods: bool = True,
) -> list[dict[str, Any]]:
    """Read the scenario's ``summary_method_end_time_and_obj_value.csv`` and
    aggregate one ``(mean time%, mean RPDf)`` point per top-level method.

    Returns a method-ordered list of dicts with keys ``method``,
    ``mean_time_pct``, ``mean_rpdf``, ``instance_count``. Methods that have
    no instance with a finite ``obj_value`` *and* baseline *and* timelimit
    are omitted.

    When ``drop_non_improving_methods`` is ``True`` (default), also drop any
    method whose ``obj_value`` matches the prior recorded ``obj_value`` for
    *every* instance — i.e. methods like ``set_cp_model_as_base_cp_model``
    that snapshot the current solution without changing it. The first
    method with a recorded ``obj_value`` per instance is always kept (no
    prior to compare against), and the last method that recorded any
    ``obj_value`` is also always kept so the chart still shows where the
    flow terminated.
    """
    df = pd.read_csv(summary_csv_path)
    if df.empty or "instance_id" not in df.columns:
        return []

    methods = _discover_methods(list(df.columns))
    if not methods:
        return []

    # Stringify per column to avoid iterrows() promoting an int instance_id
    # to "1.0" when the row Series contains any NaN.
    ins_ids = df["instance_id"].astype(str).tolist()
    prev_obj_by_instance: dict[str, float] = {}
    candidates: list[dict[str, Any]] = []
    for method in methods:
        end_col = f"{method}{_END_SEC_SUFFIX}"
        obj_col = f"{method}{_OBJ_VALUE_SUFFIX}"
        end_values = df[end_col].tolist()
        obj_values = df[obj_col].tolist()
        contributions: list[
            tuple[str, float, float, float]
        ] = []  # (ins, time_pct, rpdf, obj)
        improves = False
        for ins_id, end_raw, obj_raw in zip(ins_ids, end_values, obj_values):
            end_sec = _safe_float(end_raw)
            obj = _safe_float(obj_raw)
            if end_sec is None or obj is None:
                continue
            timelimit = timelimit_by_instance.get(ins_id)
            bks = baseline_obj_by_instance.get(ins_id)
            if timelimit is None or timelimit <= 0 or bks is None:
                continue
            prior = prev_obj_by_instance.get(ins_id)
            if prior is None or obj < prior:
                improves = True
            contributions.append(
                (ins_id, end_sec / timelimit, _rpd_f(obj, float(bks)), obj)
            )
        if not contributions:
            continue
        # Update prev_obj regardless of whether we keep the point — the next
        # method should compare against the latest recorded obj, not against
        # the last *kept* one (otherwise dropping a non-improver would let
        # the next equal-valued method look like an improvement).
        for ins_id, _, _, obj in contributions:
            prev_obj_by_instance[ins_id] = obj
        time_pcts = [t for _, t, _, _ in contributions]
        rpdfs = [r for _, _, r, _ in contributions]
        candidates.append(
            {
                "method": method,
                "improves": improves,
                "mean_time_pct": sum(time_pcts) / len(time_pcts),
                "mean_rpdf": sum(rpdfs) / len(rpdfs),
                "instance_count": len(time_pcts),
            }
        )

    if drop_non_improving_methods and candidates:
        last_idx = len(candidates) - 1
        kept: list[dict[str, Any]] = []
        for idx, cand in enumerate(candidates):
            if cand["improves"] or idx == last_idx:
                kept.append(cand)
            else:
                logger.info(
                    "Dropping non-improving method %r from %s (every instance equals the prior obj)",
                    cand["method"],
                    summary_csv_path.parent.name,
                )
        candidates = kept

    return [{k: v for k, v in c.items() if k != "improves"} for c in candidates]


def _positive_axis_upper(values: list[float]) -> float:
    if not values:
        return _EMPTY_POSITIVE_AXIS_UPPER
    max_value = max(values)
    if max_value <= 0:
        return _EMPTY_POSITIVE_AXIS_UPPER
    return max_value * _POSITIVE_AXIS_PADDING


def _x_axis_upper(values: list[float]) -> float:
    if not values:
        return _MIN_NORMALIZED_TIME_X_UPPER
    return max(_MIN_NORMALIZED_TIME_X_UPPER, max(values))


def _y_axis_lower(values: list[float]) -> float:
    if not values:
        return 0.0
    return min(0.0, min(values))


def _build_payload(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    traces: list[dict[str, Any]] = []
    all_x: list[float] = []
    all_y: list[float] = []
    for scenario in scenarios:
        method_points = scenario.get("method_points") or []
        if not method_points:
            continue
        xs = [float(p["mean_time_pct"]) for p in method_points]
        ys = [float(p["mean_rpdf"]) for p in method_points]
        names = [str(p["method"]) for p in method_points]
        counts = [int(p["instance_count"]) for p in method_points]
        traces.append(
            {
                "scenario": str(scenario["label"]),
                "x": xs,
                "y": ys,
                "method": names,
                "instance_count": counts,
            }
        )
        all_x.extend(xs)
        all_y.extend(ys)
    return {
        "traces": traces,
        "x_max": _x_axis_upper(all_x),
        "y_min": _y_axis_lower(all_y),
        "y_max": _positive_axis_upper(all_y),
    }


_HTML_TEMPLATE = Template("""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>$title</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 18px; color: #1b1b1b; }
    h1 { font-size: 20px; margin: 0 0 8px 0; }
    p { margin: 0 0 16px 0; color: #444; }
  </style>
</head>
<body>
  <h1>$title</h1>
  <p>Per-method mean (Time%, RPDf) across instances. Methods without recorded obj_value are omitted.</p>
  <div id="method-mean-scatter" style="width: 100%; height: 760px;"></div>
  <script>
    const payload = $payload_json;
    const SERIES_COLORS = $series_colors_json;
    const SYMBOL_MAP = $symbol_map_json;

    const traces = payload.traces.map((trace, idx) => {
      const seriesColor = SERIES_COLORS[idx % SERIES_COLORS.length];
      const customdata = trace.method.map((name, i) => [trace.scenario, name, trace.instance_count[i]]);
      return {
        type: "scatter",
        mode: "lines+markers",
        name: trace.scenario,
        x: trace.x,
        y: trace.y,
        customdata: customdata,
        line: { width: 2, color: seriesColor },
        marker: {
          size: 11,
          color: seriesColor,
          symbol: trace.method.map((name) => SYMBOL_MAP[name] || "circle"),
          line: { width: 1, color: "#1b1b1b" }
        },
        hovertemplate:
          "scenario=%{customdata[0]}<br>" +
          "method=%{customdata[1]}<br>" +
          "instance_cnt=%{customdata[2]}<br>" +
          "mean Time%=%{x:.$x_percent_decimals%}<br>" +
          "mean RPDf=%{y:.$y_percent_decimals%}<extra></extra>"
      };
    });

    const layout = {
      title: { text: "$title" },
      xaxis: { title: { text: "Mean normalized time" }, tickformat: ".$x_percent_decimals%", range: [0, payload.x_max] },
      yaxis: { title: { text: "Mean RPDf" }, tickformat: ".$y_percent_decimals%", range: [payload.y_min, payload.y_max] },
      template: "plotly_white",
      hovermode: "closest",
      legend: { orientation: "h" },
      margin: { l: 70, r: 20, t: 70, b: 70 }
    };

    Plotly.newPlot("method-mean-scatter", traces, layout, { responsive: true });
  </script>
</body>
</html>
""")


def _render_html(
    payload: dict[str, Any], title: str, x_decimals: int, y_decimals: int
) -> str:
    return _HTML_TEMPLATE.substitute(
        payload_json=json.dumps(payload, separators=(",", ":")),
        title=title,
        x_percent_decimals=x_decimals,
        y_percent_decimals=y_decimals,
        series_colors_json=series_colors_json(),
        symbol_map_json=symbol_map_json(),
    )


def export_method_mean_scatter_html(
    scenarios: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str = "Method mean RPDf vs mean Time%",
    x_percent_decimals: int = 1,
    y_percent_decimals: int = 1,
) -> bool:
    """Write the Plotly HTML. ``scenarios`` is a list of
    ``{label: str, method_points: list[dict]}`` where ``method_points`` is
    the output of :func:`load_method_mean_metrics`.

    Returns ``False`` when no scenario produced any method point.
    """
    payload = _build_payload(scenarios)
    if not payload["traces"]:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _render_html(payload, title, x_percent_decimals, y_percent_decimals),
        encoding="utf-8",
    )
    logger.info("Method-mean scatter HTML saved to %s", output_path)
    return True
