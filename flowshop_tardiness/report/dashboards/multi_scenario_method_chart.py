"""Multi-scenario subroutine-flow comparison Plotly HTML.

For each scenario, the writer builds a per-instance "best-so-far" RPDf
trajectory, takes the mean step function across instances, and overlays the
results in one chart. Vertical dotted guides mark each scenario's average
subroutine-call endpoint times so the reader can see "where the time went"
per stage.

Ported from ``ffc_ddw_sum_et/src/ffc_ddw_sum_et/report/multi_scenario_method_chart.py``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Fallback axis-upper when no positive RPDf values are present.
_EMPTY_POSITIVE_AXIS_UPPER = 0.01
# Headroom multiplier above the largest RPDf so markers aren't clipped.
_POSITIVE_AXIS_PADDING = 1.05
# Minimum x-axis upper so the chart doesn't squeeze horizontally when every
# scenario finishes well before t=1.
_MIN_NORMALIZED_TIME_X_UPPER = 1.0


_SERIES_COLORS: tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

# Stable marker shapes for the subroutines exercised by the CP-LNS flow.
# Subroutines not listed here fall back to "circle" at render time.
_SUBROUTINE_SYMBOL_MAP: dict[str, str] = {
    "set_random_seed": "x",
    "compute_preemptive_last_stage_lb": "diamond",
    "initialize_by_edd": "triangle-up",
    "initialize_by_nehms": "triangle-down",
    "set_cp_model_as_base_cp_model": "square",
    "improve_by_insertion": "cross",
    "repeat_while_improvement": "star",
    "pw_cp": "circle",
    "solve_base_cp_model": "hexagon",
}


@dataclass(frozen=True)
class _ProgressionPoint:
    time: float
    rpd_f: float


def _series_colors_json() -> str:
    return json.dumps(list(_SERIES_COLORS), separators=(",", ":"))


def _symbol_map_json() -> str:
    return json.dumps(_SUBROUTINE_SYMBOL_MAP, separators=(",", ":"))


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


def _normalize_scenario_input(
    scenario_input: tuple[str, pd.DataFrame] | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(scenario_input, tuple):
        label, endpoint_df = scenario_input
        return {
            "label": str(label),
            "endpoint_df": endpoint_df,
            "raw_progression_df": None,
        }
    if not isinstance(scenario_input, dict):
        raise TypeError("Scenario input must be a tuple or dict.")
    return {
        "label": str(scenario_input["label"]),
        "endpoint_df": scenario_input["endpoint_df"],
        "raw_progression_df": scenario_input.get("raw_progression_df"),
    }


def _prepare_endpoint_df(endpoint_df: pd.DataFrame) -> pd.DataFrame:
    work_df = endpoint_df.copy()
    order_map = {
        name: idx
        for idx, name in enumerate(pd.unique(work_df["subroutine_name"]), start=1)
    }
    work_df["subroutine_order"] = work_df["subroutine_name"].map(order_map)
    return work_df


def _prepare_progression_df(
    raw_progression_df: pd.DataFrame | None,
    order_source_df: pd.DataFrame,
) -> pd.DataFrame | None:
    if raw_progression_df is None or raw_progression_df.empty:
        return None
    work_df = raw_progression_df.copy()
    order_map = {
        name: idx
        for idx, name in enumerate(
            pd.unique(order_source_df["subroutine_name"]), start=1
        )
    }
    work_df["subroutine_order"] = work_df["subroutine_name"].map(order_map)
    return work_df.dropna(subset=["subroutine_order"]).copy()


def _best_so_far_points(grp: pd.DataFrame) -> list[_ProgressionPoint]:
    if grp.empty:
        return []
    times = grp["norm_time"].tolist()
    raw = grp["rpd_f"].tolist()
    best: list[float] = []
    cur: float | None = None
    for y in raw:
        cur = y if cur is None else min(cur, y)
        best.append(cur)
    deduped: dict[float, _ProgressionPoint] = {}
    for t, y in zip(times, best):
        deduped[float(t)] = _ProgressionPoint(time=float(t), rpd_f=float(y))
    return [deduped[k] for k in sorted(deduped)]


def _keep_strict_global_improvements_or_endpoints(
    progression_grp: pd.DataFrame,
) -> pd.DataFrame:
    """Keep rows whose ``rpd_f`` strictly improves the global running min,
    plus each ``call_index`` group's last row. Mirrors source-repo logic to
    keep HTML size bounded when the raw progression is dense.
    """
    if progression_grp.empty:
        return progression_grp
    sort_cols = [c for c in ["norm_time", "global_sec"] if c in progression_grp.columns]
    ordered = progression_grp.sort_values(sort_cols)
    endpoint_indices: set = set()
    for _, sub_grp in ordered.groupby("call_index", sort=False):
        endpoint_indices.add(sub_grp.index[-1])
    keep_indices: list = []
    running_min = float("inf")
    for idx, rpdf in zip(ordered.index, ordered["rpd_f"].tolist()):
        is_strict = rpdf < running_min
        is_endpoint = idx in endpoint_indices
        if is_strict or is_endpoint:
            keep_indices.append(idx)
        if is_strict:
            running_min = rpdf
    return progression_grp.loc[keep_indices].sort_values(sort_cols)


def _progression_points_to_arrays(
    points: list[_ProgressionPoint],
) -> tuple[np.ndarray, np.ndarray]:
    n = len(points)
    times = np.fromiter((p.time for p in points), dtype=np.float64, count=n)
    values = np.fromiter((p.rpd_f for p in points), dtype=np.float64, count=n)
    return times, values


def _step_function_mean_over_union(
    model_arrays: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[list[float], list[float]]:
    start_time = max(float(times[0]) for times, _ in model_arrays)
    end_time = max(float(times[-1]) for times, _ in model_arrays)

    in_range = [t[(t >= start_time) & (t <= end_time)] for t, _ in model_arrays]
    event_times = np.unique(np.concatenate(in_range)) if in_range else np.array([])
    if event_times.size == 0:
        event_times = (
            np.array([start_time, end_time], dtype=np.float64)
            if end_time > start_time
            else np.array([start_time], dtype=np.float64)
        )
    elif event_times[-1] < end_time:
        event_times = np.append(event_times, end_time)

    sum_y = np.zeros(event_times.shape, dtype=np.float64)
    for times, values in model_arrays:
        idx = np.searchsorted(times, event_times, side="right") - 1
        sum_y += values[idx]
    mean_y_arr = sum_y / len(model_arrays)

    return event_times.tolist(), mean_y_arr.tolist()


def _build_step_path(
    x_values: list[float], y_values: list[float]
) -> tuple[list[float], list[float]]:
    step_x: list[float] = []
    step_y: list[float] = []
    for idx, (x, y) in enumerate(zip(x_values, y_values)):
        if idx == 0:
            step_x.append(x)
            step_y.append(y)
            continue
        prev_y = y_values[idx - 1]
        step_x.append(x)
        step_y.append(prev_y)
        if y < prev_y:
            step_x.append(x)
            step_y.append(y)
    return step_x, step_y


def _fill_missing_subroutine_endpoints(endpoint_df: pd.DataFrame) -> pd.DataFrame:
    """For each instance, add a synthetic endpoint row for every scenario-level
    subroutine the instance never reached. Without this, the guide-marker
    average for a step only a subset of instances reached sits at that
    subset's mean, which misleads when most instances never got there.
    """
    if endpoint_df.empty:
        return endpoint_df
    all_subroutines = list(pd.unique(endpoint_df["subroutine_name"]))
    order_by_name = (
        endpoint_df[["subroutine_name", "subroutine_order"]]
        .drop_duplicates()
        .set_index("subroutine_name")["subroutine_order"]
        .to_dict()
    )
    synth_rows: list[dict[str, Any]] = []
    for _ins, grp in endpoint_df.groupby("instance_id", sort=False):
        present = set(grp["subroutine_name"])
        missing = [s for s in all_subroutines if s not in present]
        if not missing:
            continue
        last = grp.sort_values("norm_time").iloc[-1].to_dict()
        for s in missing:
            row = dict(last)
            row["subroutine_name"] = s
            row["subroutine_order"] = order_by_name[s]
            synth_rows.append(row)
    if not synth_rows:
        return endpoint_df
    return pd.concat([endpoint_df, pd.DataFrame(synth_rows)], ignore_index=True)


def _build_scenario_progression_models(
    endpoint_df: pd.DataFrame,
    raw_progression_df: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    progression_by_instance: dict[str, pd.DataFrame] = {}
    if raw_progression_df is not None and not raw_progression_df.empty:
        sort_cols = [
            c
            for c in ["norm_time", "global_sec", "call_index"]
            if c in raw_progression_df.columns
        ]
        progression_by_instance = {
            str(ins): _keep_strict_global_improvements_or_endpoints(
                grp.sort_values(sort_cols)
            )
            for ins, grp in raw_progression_df.groupby("instance_id", sort=True)
        }

    models: list[dict[str, Any]] = []
    for ins, ep_grp in endpoint_df.groupby("instance_id", sort=True):
        ep_grp = ep_grp.sort_values(
            ["norm_time", "subroutine_order", "subroutine_name"]
        )
        prog_grp = progression_by_instance.get(str(ins))
        source_grp = ep_grp if prog_grp is None or prog_grp.empty else prog_grp
        models.append(
            {
                "instance_id": str(ins),
                "progression_points": _best_so_far_points(source_grp),
            }
        )
    return models


def _build_scenario_mean_series(
    scenario_label: str,
    endpoint_df: pd.DataFrame,
    raw_progression_df: pd.DataFrame | None,
) -> dict[str, Any] | None:
    endpoint_df = _fill_missing_subroutine_endpoints(endpoint_df)
    models = _build_scenario_progression_models(endpoint_df, raw_progression_df)
    models = [m for m in models if m["progression_points"]]
    if not models:
        return None

    model_arrays = [
        _progression_points_to_arrays(m["progression_points"]) for m in models
    ]
    mean_x, mean_y = _step_function_mean_over_union(model_arrays)
    step_x, step_y = _build_step_path(mean_x, mean_y)

    guide_df = (
        endpoint_df.sort_values(["subroutine_order", "subroutine_name", "norm_time"])
        .groupby("subroutine_name", as_index=False, sort=False)
        .agg(avg_norm_time=("norm_time", "mean"))
    )
    guide_x = guide_df["avg_norm_time"].astype(float).tolist()
    guide_text = guide_df["subroutine_name"].astype(str).tolist()
    return {
        "scenario": scenario_label,
        "step_x": step_x,
        "step_y": step_y,
        "meta": [scenario_label, len(models)],
        "vertical_guides": [
            {"subroutine_name": name, "x": x}
            for name, x in zip(guide_text, guide_x)
        ],
        "guide_marker_x": guide_x,
        "guide_marker_text": guide_text,
        "guide_marker_customdata": [[scenario_label, str(n)] for n in guide_text],
    }


def _drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["norm_time", "rpd_f", "subroutine_name"]).copy()


def _build_payload(
    scenario_metrics: list[tuple[str, pd.DataFrame] | dict[str, Any]],
) -> dict:
    traces: list[dict[str, Any]] = []
    all_x: list[float] = []
    all_y: list[float] = []
    for raw_input in scenario_metrics:
        scenario = _normalize_scenario_input(raw_input)
        endpoint_df = scenario["endpoint_df"]
        raw_progression_df = scenario["raw_progression_df"]
        if endpoint_df is None or endpoint_df.empty:
            continue
        endpoint_clean = _drop_invalid_rows(endpoint_df)
        if endpoint_clean.empty:
            continue
        progression_clean: pd.DataFrame | None = None
        if raw_progression_df is not None and not raw_progression_df.empty:
            progression_clean = _drop_invalid_rows(raw_progression_df)

        endpoint_work = _prepare_endpoint_df(endpoint_clean)
        progression_work = _prepare_progression_df(progression_clean, endpoint_work)
        mean_series = _build_scenario_mean_series(
            str(scenario["label"]), endpoint_work, progression_work
        )
        if mean_series is None:
            continue
        traces.append(mean_series)
        all_x.extend(float(x) for x in mean_series["step_x"])
        all_y.extend(float(y) for y in mean_series["step_y"])
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
  <title>Subroutine Flow Comparison</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 18px; color: #1b1b1b; }
    h1 { font-size: 20px; margin: 0 0 8px 0; }
    p { margin: 0 0 16px 0; color: #444; }
  </style>
</head>
<body>
  <h1>Subroutine Flow Comparison</h1>
  <p>Mean over-time RPDf progression by scenario.</p>
  <div id="multi-scenario-method-chart" style="width: 100%; height: 760px;"></div>
  <script>
    const payload = $payload_json;
    const SERIES_COLORS = $series_colors_json;
    const SYMBOL_MAP = $symbol_map_json;

    function buildVisibleGuideShapes(plotData) {
      return payload.traces.flatMap((trace, idx) => {
        const lineTrace = plotData?.[idx * 2];
        const isVisible = lineTrace && lineTrace.visible !== "legendonly";
        if (!isVisible) return [];
        const seriesColor = SERIES_COLORS[idx % SERIES_COLORS.length];
        return (trace.vertical_guides || []).map((guide) => ({
          type: "line", xref: "x", yref: "paper",
          x0: guide.x, x1: guide.x, y0: 0, y1: 1,
          line: { color: seriesColor, width: 1, dash: "dot" }
        }));
      });
    }

    const traces = payload.traces.flatMap((trace, idx) => {
      const seriesColor = SERIES_COLORS[idx % SERIES_COLORS.length];
      return [
        { type: "scatter", mode: "lines",
          name: trace.scenario, legendgroup: trace.scenario,
          x: trace.step_x, y: trace.step_y,
          meta: trace.meta,
          line: { width: 2, color: seriesColor },
          hovertemplate:
            "scenario=%{meta[0]}<br>" +
            "instance_cnt=%{meta[1]}<br>" +
            "Time%=%{x:.4%}<br>" +
            "Mean RPDf=%{y:.4%}<extra></extra>",
          showlegend: true },
        { type: "scatter", mode: "markers",
          name: trace.scenario, legendgroup: trace.scenario,
          x: trace.guide_marker_x,
          y: trace.guide_marker_x.map(() => 0),
          text: trace.guide_marker_text,
          customdata: trace.guide_marker_customdata,
          marker: {
            size: 9, color: seriesColor,
            symbol: trace.guide_marker_text.map((name) => SYMBOL_MAP[name] || "circle")
          },
          hovertemplate:
            "scenario=%{customdata[0]}<br>" +
            "subroutine=%{customdata[1]}<br>" +
            "avg end Time%=%{x:.4%}<extra></extra>",
          showlegend: false }
      ];
    });

    const layout = {
      title: { text: "Subroutine flow mean over-time RPDf by scenario" },
      xaxis: { title: { text: "Normalized time" }, tickformat: ".$x_percent_decimals%", range: [0, payload.x_max] },
      yaxis: { title: { text: "Mean RPDf" }, tickformat: ".$y_percent_decimals%", range: [payload.y_min, payload.y_max] },
      template: "plotly_white",
      hovermode: "closest",
      legend: { orientation: "h", groupclick: "togglegroup" },
      margin: { l: 70, r: 20, t: 70, b: 70 },
      shapes: buildVisibleGuideShapes(traces)
    };

    Plotly.newPlot("multi-scenario-method-chart", traces, layout, { responsive: true })
      .then((gd) => {
        const sync = () => Plotly.relayout(gd, { shapes: buildVisibleGuideShapes(gd.data) });
        gd.on("plotly_restyle", sync);
      });
  </script>
</body>
</html>
""")


def _render_html(payload: dict, x_decimals: int, y_decimals: int) -> str:
    return _HTML_TEMPLATE.substitute(
        payload_json=json.dumps(payload, separators=(",", ":")),
        x_percent_decimals=x_decimals,
        y_percent_decimals=y_decimals,
        series_colors_json=_series_colors_json(),
        symbol_map_json=_symbol_map_json(),
    )


def export_multi_scenario_method_rpdf_comparison_html(
    scenario_metrics: list[tuple[str, pd.DataFrame] | dict[str, Any]],
    output_path: Path,
    *,
    x_percent_decimals: int = 1,
    y_percent_decimals: int = 1,
) -> bool:
    """Render the run-level scenario-comparison chart.

    Returns ``False`` when no scenario yielded usable data.
    """
    payload = _build_payload(scenario_metrics)
    if not payload["traces"]:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _render_html(payload, x_percent_decimals, y_percent_decimals),
        encoding="utf-8",
    )
    logger.info("Multi-scenario method comparison HTML saved to %s", output_path)
    return True
