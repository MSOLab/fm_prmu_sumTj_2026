"""Trim per-instance obj_log time series to the configured timelimit.

Models the run as if the solver had actually halted at ``timelimit`` seconds —
useful when OR-Tools CP-SAT overruns its wall-clock deadline under heavy CPU
contention. The on-disk per-instance ``<ins>_obj_log.yaml`` already records
``(elapsed_time, obj_value/obj_bound)`` pairs, so we just take the latest
entry with ``elapsed_time <= timelimit``.

Overwrites ``bestObj`` / ``bestBound`` / ``totalElapsedTime`` in the
aggregated summary frame; preserves the originals as ``*_endpoint`` columns
for traceability and recomputes ``improvementRatio`` from the trimmed
``bestObj`` so downstream RPDf / Gap reflect the deadline-truncated view.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_OBJ_LOG_FN_FORMAT = "{}_obj_log.yaml"
_RESULT_DIR_NAME = "results"


def _last_value_at_or_before(
    data: dict, threshold_sec: float
) -> float | None:
    """Latest numeric value whose timestamp key is ``<= threshold_sec``."""
    best_t = -math.inf
    best_v: float | None = None
    for k, v in data.items():
        try:
            t = float(k)
        except (TypeError, ValueError):
            continue
        if t > threshold_sec:
            continue
        if t > best_t:
            best_t = t
            try:
                best_v = float(v)
            except (TypeError, ValueError):
                continue
    return best_v


def _load_obj_log_series(obj_log_path: Path) -> tuple[dict, dict]:
    with open(obj_log_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    obj_value_block = payload.get("obj_value") or {}
    obj_bound_block = payload.get("obj_bound") or {}
    return (
        dict(obj_value_block.get("data") or {}),
        dict(obj_bound_block.get("data") or {}),
    )


def _resolve_obj_log_path(
    run_dir: Path, scenario: str, ins_name: str
) -> Path | None:
    base = run_dir / scenario / ins_name
    in_results = base / _RESULT_DIR_NAME / _OBJ_LOG_FN_FORMAT.format(ins_name)
    if in_results.exists():
        return in_results
    flat = base / _OBJ_LOG_FN_FORMAT.format(ins_name)
    if flat.exists():
        return flat
    return None


def apply_timelimit_trim(
    summary_df: pd.DataFrame, run_dir: Path
) -> pd.DataFrame:
    """Return a copy of ``summary_df`` with values trimmed to ``timelimit``.

    For each row, reads the matching ``<ins>_obj_log.yaml`` and replaces:

    * ``bestObj`` ← obj_value recorded at or before ``timelimit``
    * ``bestBound`` ← obj_bound recorded at or before ``timelimit``
    * ``totalElapsedTime`` ← ``min(originalElapsed, timelimit)``
    * ``improvementRatio`` ← recomputed from ``(initObj - trimmedBest) / initObj``

    Originals are preserved in ``bestObj_endpoint``, ``bestBound_endpoint``,
    ``totalElapsedTime_endpoint``. Rows without a readable obj_log keep their
    original values unchanged.
    """
    if summary_df.empty:
        return summary_df
    required = {"scenario", "insName", "timelimit", "bestObj", "bestBound"}
    missing = required - set(summary_df.columns)
    if missing:
        logger.warning(
            "Timelimit trim skipped: summary df missing columns %s",
            sorted(missing),
        )
        return summary_df

    out = summary_df.copy()
    out["bestObj_endpoint"] = out["bestObj"]
    out["bestBound_endpoint"] = out["bestBound"]
    has_elapsed = "totalElapsedTime" in out.columns
    if has_elapsed:
        out["totalElapsedTime_endpoint"] = out["totalElapsedTime"]
    has_init = "initObj" in out.columns

    trimmed_obj: list = []
    trimmed_bnd: list = []
    trimmed_time: list = []
    trimmed_improve: list = []
    trimmed_count = 0
    missing_log = 0

    for _, row in out.iterrows():
        scenario = str(row["scenario"])
        ins_name = str(row["insName"])
        try:
            timelimit = float(row["timelimit"])
        except (TypeError, ValueError):
            timelimit = math.nan

        orig_obj = row["bestObj"]
        orig_bnd = row["bestBound"]
        orig_time = row["totalElapsedTime"] if has_elapsed else None
        orig_improve = row.get("improvementRatio") if has_init else None

        new_obj = orig_obj
        new_bnd = orig_bnd
        new_time = orig_time
        new_improve = orig_improve

        path = _resolve_obj_log_path(run_dir, scenario, ins_name)
        if path is None or math.isnan(timelimit):
            missing_log += 1
        else:
            try:
                data_v, data_b = _load_obj_log_series(path)
            except Exception as e:
                logger.warning("Failed to read %s: %s", path, e)
                missing_log += 1
            else:
                v_at = _last_value_at_or_before(data_v, timelimit)
                b_at = _last_value_at_or_before(data_b, timelimit)
                if v_at is not None:
                    new_obj = v_at
                if b_at is not None:
                    new_bnd = b_at
                if has_elapsed and pd.notna(orig_time):
                    new_time = min(float(orig_time), timelimit)
                if has_init:
                    init = row.get("initObj")
                    if (
                        pd.notna(init)
                        and pd.notna(new_obj)
                        and float(init) != 0
                    ):
                        new_improve = (float(init) - float(new_obj)) / float(init)
                if v_at is not None or b_at is not None:
                    trimmed_count += 1

        trimmed_obj.append(new_obj)
        trimmed_bnd.append(new_bnd)
        trimmed_time.append(new_time)
        trimmed_improve.append(new_improve)

    out["bestObj"] = trimmed_obj
    out["bestBound"] = trimmed_bnd
    if has_elapsed:
        out["totalElapsedTime"] = trimmed_time
    if has_init:
        out["improvementRatio"] = trimmed_improve

    logger.info(
        "Timelimit trim applied: %d/%d rows updated, %d rows without obj_log",
        trimmed_count,
        len(out),
        missing_log,
    )
    return out
