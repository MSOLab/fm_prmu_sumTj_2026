#!/usr/bin/env python3
import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# phase -> method_name row in method_end_time_and_obj_value.csv (end_sec source)
PHASE_MAP = [
    ("edd", "initialize_by_edd"),
    ("neh_ms", "initialize_by_nehms"),
    ("local_search", "repeat_while_improvement"),
    ("base_cp", "solve_base_cp_model"),
]
PHASE_ORDER = {phase: i for i, (phase, _) in enumerate(PHASE_MAP)}

# token in obj_log.yaml notes -> phase (obj_value source)
NOTE_TOKENS = [
    ("initialize_by_edd", "edd"),
    ("initialize_by_nehms", "neh_ms"),
    ("repeat_while_improvement", "local_search"),
    ("solve_base_cp_model", "base_cp"),
]


def _is_missing(v) -> bool:
    if v is None:
        return True
    try:
        return pd.isna(float(v))
    except (TypeError, ValueError):
        return str(v).strip() == ""


def _note_phase(note: str) -> str | None:
    for token, phase in NOTE_TOKENS:
        if token in note:
            return phase
    return None


def load_end_sec(instance_dir: Path) -> dict[str, float]:
    """phase -> end_sec from method_end_time_and_obj_value.csv."""
    csv_path = instance_dir / "method_end_time_and_obj_value.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    by_method = {
        str(row.get("method_name", "")).strip(): row.get("method_end_sec")
        for _, row in df.iterrows()
    }
    result: dict[str, float] = {}
    for phase, method_name in PHASE_MAP:
        end_sec = by_method.get(method_name)
        if not _is_missing(end_sec):
            result[phase] = float(end_sec)
    return result


def load_obj_trace(instance_dir: Path) -> list[tuple[float, float, int]]:
    """Sorted (timestamp, obj, phase_rank) points from results/<ins>_obj_log.yaml."""
    yaml_path = instance_dir / "results" / f"{instance_dir.name}_obj_log.yaml"
    if not yaml_path.exists():
        return []
    with open(yaml_path) as f:
        content = yaml.safe_load(f)
    obj_value = (content or {}).get("obj_value") or {}
    data = obj_value.get("data") or {}
    notes = obj_value.get("notes") or {}

    points: list[tuple[float, float, int]] = []
    for ts_key, obj in data.items():
        phase = _note_phase(str(notes.get(ts_key, "")))
        if phase is None:
            continue
        points.append((float(ts_key), float(obj), PHASE_ORDER[phase]))
    points.sort(key=lambda p: p[0])
    return points


def process_instance(instance_dir: Path) -> list[dict]:
    end_sec_by_phase = load_end_sec(instance_dir)
    if not end_sec_by_phase:
        return []
    trace = load_obj_trace(instance_dir)
    if not trace:
        print(f"[warn] {instance_dir.name}: empty obj trace", file=sys.stderr)
        return []

    rows: list[dict] = []
    for phase, _ in PHASE_MAP:
        end_sec = end_sec_by_phase.get(phase)
        if end_sec is None:
            continue
        rank = PHASE_ORDER[phase]
        # obj = best incumbent at phase end = last trace point attributed to
        # this phase or an earlier one (trace is monotone non-increasing).
        eligible = [obj for _, obj, r in trace if r <= rank]
        if not eligible:
            continue
        rows.append(
            {
                "insName": instance_dir.name,
                "phase": phase,
                "end_sec": end_sec,
                "obj_value": eligible[-1],
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Extract phase-level (end_sec, obj_value) from C4 ablation results."
    )
    ap.add_argument(
        "--root",
        default="Outputs_scenarios/20260513T142520_492897/20260512_ablation_c4",
        help="C4 ablation scenario directory",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: analysis/<timestamp>_phase_obj_c4/)",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"[error] Root not found: {root}", file=sys.stderr)
        sys.exit(1)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")[:22]
        out_dir = Path("analysis") / f"{timestamp}_phase_obj_c4"
    out_dir.mkdir(parents=True, exist_ok=True)

    instance_dirs = sorted(
        (d for d in root.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda p: int(p.name),
    )

    all_rows: list[dict] = []
    for inst_dir in instance_dirs:
        try:
            all_rows.extend(process_instance(inst_dir))
        except Exception as e:
            print(f"[warn] Failed to process {inst_dir.name}: {e}", file=sys.stderr)

    out_path = out_dir / "phase_obj_c4.csv"
    df = pd.DataFrame(all_rows, columns=["insName", "phase", "end_sec", "obj_value"])
    df["_order"] = df["phase"].map(PHASE_ORDER)
    df = df.sort_values(["insName", "_order"]).drop(columns="_order")
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"  Phases: {sorted(df['phase'].unique())}")
    print(f"  Instances: {df['insName'].nunique()}")


if __name__ == "__main__":
    main()
