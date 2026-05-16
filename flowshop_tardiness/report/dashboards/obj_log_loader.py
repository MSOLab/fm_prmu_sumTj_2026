"""Decode per-instance ``<ins>_obj_log.yaml`` into chart-friendly DataFrames.

The on-disk layout written by ``FsSingleInstanceRunner`` is::

    obj_value:
      data:  { "<global_sec>": <obj>, ... }   # every recorded transition
      notes: { "<global_sec>": "<step_idx>-<subroutine_name>", ... }
                                              # only at controller-step endpoints

Notes mark the END of a controller step. A data point at time ``t`` belongs to
the segment whose ``(prev_end, end_sec]`` window contains ``t``. The label
format from routix is ``"<step_idx>-<subroutine_name>"`` for top-level calls
and ``"5-repeat_while_improvement.1-reps.1-pw_cp"`` for nested ones; for the
chart we keep the innermost method as ``subroutine_name`` so e.g. ``pw_cp``
gets a stable symbol/legend regardless of which iteration produced it.

Failure policy mirrors the upstream ``ffc_ddw_sum_et`` chart: raise loudly on
shape drift rather than emit best-effort rows that mislead downstream means.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_STEP_LABEL_RE = re.compile(r"^(\d+)-(.+)$")


@dataclass(frozen=True)
class ProgPoint:
    """One point on a series trajectory in controller-frame seconds."""

    global_sec: float
    value: float


@dataclass(frozen=True)
class CallSegment:
    """One controller-step's contribution to a single series."""

    call_index: int  # synthetic 1-based sequence index across all notes
    subroutine_name: str  # innermost method name (e.g. "pw_cp")
    prefixed_subroutine_name: str  # raw note label (may include nested chain)
    global_start_sec: float
    global_end_sec: float
    points: tuple[ProgPoint, ...]

    @property
    def elapsed_sec(self) -> float:
        return self.global_end_sec - self.global_start_sec


@dataclass(frozen=True)
class InstanceProgression:
    """Decoded trajectory for one instance."""

    instance_id: str
    job_cnt: int
    stage_cnt: int
    timelimit_sec: float
    obj_value_calls: tuple[CallSegment, ...]


def _innermost_method_name(label: str) -> str:
    # "5-repeat_while_improvement.1-reps.1-pw_cp" -> "pw_cp"
    # "2-initialize_by_edd" -> "initialize_by_edd"
    last = label.rsplit(".", 1)[-1]
    m = _STEP_LABEL_RE.match(last)
    if m is None:
        return last
    return m.group(2)


def _truncate_calls_to_timelimit(
    calls: tuple[CallSegment, ...], timelimit_sec: float
) -> tuple[CallSegment, ...]:
    """Truncate the call trajectory to ``timelimit_sec``.

    Models the deadline-truncated view (matches ``apply_timelimit_trim`` in
    :mod:`obj_log_trim`):

    * Calls entirely within the budget are kept unchanged.
    * A call straddling the deadline is truncated: ``global_end_sec`` becomes
      ``timelimit_sec`` and only points with ``global_sec <= timelimit_sec``
      are kept. If no original point falls within the truncated window, a
      single carry-forward point is synthesized at ``timelimit_sec`` using
      the last known value, so downstream charts still emit an endpoint
      marker at ``norm_time = 1``.
    * Calls starting after the deadline are dropped.
    """
    if timelimit_sec is None or timelimit_sec <= 0 or not calls:
        return calls

    out: list[CallSegment] = []
    last_value: float | None = None
    for call in calls:
        if call.global_start_sec >= timelimit_sec:
            break
        if call.global_end_sec <= timelimit_sec:
            out.append(call)
            if call.points:
                last_value = call.points[-1].value
            continue
        kept_points = tuple(p for p in call.points if p.global_sec <= timelimit_sec)
        if not kept_points and last_value is not None:
            kept_points = (ProgPoint(global_sec=timelimit_sec, value=last_value),)
        if not kept_points:
            break
        out.append(
            CallSegment(
                call_index=call.call_index,
                subroutine_name=call.subroutine_name,
                prefixed_subroutine_name=call.prefixed_subroutine_name,
                global_start_sec=call.global_start_sec,
                global_end_sec=timelimit_sec,
                points=kept_points,
            )
        )
        last_value = kept_points[-1].value
        break
    return tuple(out)


def _build_calls_for_series(
    data: dict[str, float],
    notes: dict[str, str],
) -> tuple[CallSegment, ...]:
    if not notes:
        return ()

    sorted_data = sorted(
        ((float(k), float(v)) for k, v in data.items()), key=lambda x: x[0]
    )
    sorted_endpoints = sorted(
        ((float(k), str(v)) for k, v in notes.items()), key=lambda x: x[0]
    )

    calls: list[CallSegment] = []
    prev_end = 0.0
    cursor = 0
    for synthetic_idx, (end_sec, label) in enumerate(sorted_endpoints, start=1):
        seg_points: list[ProgPoint] = []
        while cursor < len(sorted_data) and sorted_data[cursor][0] <= end_sec:
            t, v = sorted_data[cursor]
            if t > prev_end:
                seg_points.append(ProgPoint(global_sec=t, value=v))
            cursor += 1

        calls.append(
            CallSegment(
                call_index=synthetic_idx,
                subroutine_name=_innermost_method_name(label),
                prefixed_subroutine_name=label,
                global_start_sec=prev_end,
                global_end_sec=end_sec,
                points=tuple(seg_points),
            )
        )
        prev_end = end_sec

    return tuple(calls)


def _extract_obj_value_block(
    payload: dict[str, Any], source: Path
) -> tuple[dict[str, float], dict[str, str]]:
    block = payload.get("obj_value")
    if block is None:
        return {}, {}
    if not isinstance(block, dict):
        raise ValueError(
            f"obj_log['obj_value'] in {source} is {type(block).__name__}, "
            "expected mapping"
        )
    data = block.get("data", {})
    notes = block.get("notes", {})
    if not isinstance(data, dict) or not isinstance(notes, dict):
        raise ValueError(
            f"obj_log['obj_value'].(data|notes) in {source} is not a mapping"
        )
    return data, notes


def load_instance_progression(
    obj_log_path: Path,
    *,
    instance_id: str,
    job_cnt: int,
    stage_cnt: int,
    timelimit_sec: float,
) -> InstanceProgression:
    """Read ``<ins>_obj_log.yaml`` and decode it into an :class:`InstanceProgression`.

    Instance metadata (``job_cnt``, ``stage_cnt``, ``timelimit_sec``) is passed
    in by the caller — typically pulled from the run-level
    ``all_scenarios_summary.csv`` so we don't re-parse per-instance summaries.
    """
    with open(obj_log_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"obj_log payload is not a mapping: {obj_log_path}")

    data, notes = _extract_obj_value_block(payload, obj_log_path)
    obj_value_calls = _build_calls_for_series(data, notes)
    obj_value_calls = _truncate_calls_to_timelimit(
        obj_value_calls, float(timelimit_sec)
    )

    return InstanceProgression(
        instance_id=instance_id,
        job_cnt=int(job_cnt),
        stage_cnt=int(stage_cnt),
        timelimit_sec=float(timelimit_sec),
        obj_value_calls=obj_value_calls,
    )


def build_endpoint_df(progressions: list[InstanceProgression]) -> pd.DataFrame:
    """One row per (instance, controller-step endpoint).

    ``rpd_f`` is left for the caller to fill via a baseline join.
    """
    rows: list[dict[str, Any]] = []
    for prog in progressions:
        if prog.timelimit_sec <= 0:
            raise ValueError(
                f"non-positive timelimit_sec for instance {prog.instance_id}: "
                f"{prog.timelimit_sec}"
            )
        for call in prog.obj_value_calls:
            if not call.points:
                continue
            endpoint_value = call.points[-1].value
            rows.append(
                {
                    "instance_id": prog.instance_id,
                    "job_cnt": prog.job_cnt,
                    "stage_cnt": prog.stage_cnt,
                    "subroutine_name": call.subroutine_name,
                    "prefixed_subroutine_name": call.prefixed_subroutine_name,
                    "call_index": call.call_index,
                    "global_end_sec": call.global_end_sec,
                    "norm_time": call.global_end_sec / prog.timelimit_sec,
                    "obj_value": endpoint_value,
                }
            )
    return pd.DataFrame(rows)


def build_raw_progression_df(progressions: list[InstanceProgression]) -> pd.DataFrame:
    """One row per (instance, controller-step, data point).

    Used by the multi-scenario chart to draw the inter-step trajectory.
    """
    rows: list[dict[str, Any]] = []
    for prog in progressions:
        if prog.timelimit_sec <= 0:
            raise ValueError(
                f"non-positive timelimit_sec for instance {prog.instance_id}: "
                f"{prog.timelimit_sec}"
            )
        for call in prog.obj_value_calls:
            for point in call.points:
                rows.append(
                    {
                        "instance_id": prog.instance_id,
                        "job_cnt": prog.job_cnt,
                        "stage_cnt": prog.stage_cnt,
                        "subroutine_name": call.subroutine_name,
                        "prefixed_subroutine_name": call.prefixed_subroutine_name,
                        "call_index": call.call_index,
                        "global_sec": point.global_sec,
                        "norm_time": point.global_sec / prog.timelimit_sec,
                        "obj_value": point.value,
                    }
                )
    return pd.DataFrame(rows)


# Surfaced so callers can dump intermediate frames for debugging.
def _debug_dump_endpoint_df(df: pd.DataFrame, out_path: Path) -> None:
    out_path.write_text(json.dumps(df.to_dict(orient="records"), indent=2))
