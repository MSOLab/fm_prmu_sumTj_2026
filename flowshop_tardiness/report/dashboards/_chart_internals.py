"""Shared internals for the dashboard chart writers.

Both ``multi_scenario_method_chart`` (run-level overlay) and
``rpdf_scatter_chart`` (per-scenario detail) emit Plotly HTML over the same
``ProgressionPoint`` model and the same color / symbol palette. This module
is the single home for that shared surface so the two chart writers can't
drift apart silently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd


SERIES_COLORS: tuple[str, ...] = (
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
SUBROUTINE_SYMBOL_MAP: dict[str, str] = {
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


def series_colors_json() -> str:
    return json.dumps(list(SERIES_COLORS), separators=(",", ":"))


def symbol_map_json() -> str:
    return json.dumps(SUBROUTINE_SYMBOL_MAP, separators=(",", ":"))


@dataclass(frozen=True)
class ProgressionPoint:
    """One sample on an instance's RPDf-vs-norm-time trajectory."""

    time: float
    rpd_f: float


def build_best_so_far_points(grp: pd.DataFrame) -> list[ProgressionPoint]:
    """Convert a per-instance trajectory frame to a sorted, deduped list of
    points where ``rpd_f`` is replaced by the running min (best-so-far).
    """
    if grp.empty:
        return []
    times = grp["norm_time"].tolist()
    raw = grp["rpd_f"].tolist()
    best: list[float] = []
    cur: float | None = None
    for y in raw:
        cur = y if cur is None else min(cur, y)
        best.append(cur)
    deduped: dict[float, ProgressionPoint] = {}
    for t, y in zip(times, best):
        deduped[float(t)] = ProgressionPoint(time=float(t), rpd_f=float(y))
    return [deduped[k] for k in sorted(deduped)]


def keep_strict_global_improvements_or_endpoints(
    progression_grp: pd.DataFrame,
) -> pd.DataFrame:
    """Keep rows whose ``rpd_f`` strictly improves the global running min
    plus each ``call_index`` group's last row. Bounds HTML payload size when
    the raw progression is dense.
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


def progression_points_to_arrays(
    points: list[ProgressionPoint],
) -> tuple[np.ndarray, np.ndarray]:
    n = len(points)
    times = np.fromiter((p.time for p in points), dtype=np.float64, count=n)
    values = np.fromiter((p.rpd_f for p in points), dtype=np.float64, count=n)
    return times, values


def step_function_mean_over_union(
    model_arrays: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[list[float], list[float]]:
    """Mean step function across per-instance piecewise-constant trajectories,
    sampled at the union of all change times within
    ``[max(first_times), max(last_times)]``.
    """
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


def build_step_path(
    x_values: list[float], y_values: list[float]
) -> tuple[list[float], list[float]]:
    """Expand ``(x_values, y_values)`` into staircase coordinates suitable
    for a ``mode: lines`` Plotly trace. At each downward transition, emit
    both the held-previous-y and the new-y points at the same x so Plotly
    draws the vertical drop.
    """
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
