#!/usr/bin/env python3
import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Optional

PAT_START = re.compile(r"-\s*by start time sequence:\s*([-+]?\d+(?:\.\d+)?)")
PAT_COMP = re.compile(r"-\s*by completion time sequence:\s*([-+]?\d+(?:\.\d+)?)")
PAT_AVG = re.compile(r"-\s*by average time sequence:\s*([-+]?\d+(?:\.\d+)?)")


def _parse_value(s: str) -> float:
    try:
        v = float(s)
        return int(v) if v.is_integer() else v
    except Exception:
        return float("nan")


def parse_log(fp: Path) -> Optional[tuple[float, float, float]]:
    by_start = by_comp = by_avg = None
    try:
        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if by_start is None:
                    m = PAT_START.search(line)
                    if m:
                        by_start = _parse_value(m.group(1))
                        continue
                if by_comp is None:
                    m = PAT_COMP.search(line)
                    if m:
                        by_comp = _parse_value(m.group(1))
                        continue
                if by_avg is None:
                    m = PAT_AVG.search(line)
                    if m:
                        by_avg = _parse_value(m.group(1))
                        continue
                if by_start is not None and by_comp is not None and by_avg is not None:
                    break
    except Exception as e:
        print(f"[warn] Failed to read {fp}: {e}", file=sys.stderr)
        return None
    if by_start is None or by_comp is None or by_avg is None:
        return None
    return by_start, by_comp, by_avg


def main():
    ap = argparse.ArgumentParser(
        description="Collect dispatched schedule totals from subroutine_controller.log files under a subroutine_flow_lb directory (one level of insId subdirs)."
    )
    ap.add_argument(
        "root",
        nargs="?",
        default="../Outputs_scenarios/20251102T023701_187983/output_600s/subroutine_flow_lb",
        help="Path to subroutine_flow_lb directory (default: Outputs_scenarios)",
    )
    ap.add_argument(
        "-o",
        "--output",
        default="dispatched_totals_by_instance.csv",
        help="Output CSV path (default: dispatched_totals_by_instance.csv)",
    )
    ap.add_argument(
        "--filename",
        default="subroutine_controller.log",
        help="Log filename to read inside each insId dir (default: subroutine_controller.log)",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"[error] Root not found: {root}", file=sys.stderr)
        sys.exit(1)

    rows: list[dict[str, object]] = []
    seen: dict[str, dict[str, object]] = {}

    # Scan only one level of insId directories under subroutine_flow_lb
    ins_dirs = [d for d in root.iterdir() if d.is_dir()]

    # Sort by numeric name when possible, else lexicographically
    def _dir_key(p: Path):
        s = p.name
        try:
            return (0, int(s))
        except ValueError:
            return (1, s)

    ins_dirs.sort(key=_dir_key)

    for ins_dir in ins_dirs:
        fp = ins_dir / args.filename
        if not fp.is_file():
            continue
        vals = parse_log(fp)
        if not vals:
            continue

        ins_id = ins_dir.name
        row = {
            "insId": ins_id,
            "byStartTime": vals[0],
            "byCompTime": vals[1],
            "byAvgTime": vals[2],
            "logPath": str(fp),
        }

        if ins_id in seen:
            prev = seen[ins_id]
            if (prev["byStartTime"], prev["byCompTime"], prev["byAvgTime"]) != (
                row["byStartTime"],
                row["byCompTime"],
                row["byAvgTime"],
            ):
                print(
                    f"[warn] Duplicate insId {ins_id} with differing values; keeping first from {prev['logPath']}, ignoring {fp}",
                    file=sys.stderr,
                )
            continue

        seen[ins_id] = row
        rows.append(row)

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["insId", "byStartTime", "byCompTime", "byAvgTime"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "insId": r["insId"],
                    "byStartTime": r["byStartTime"],
                    "byCompTime": r["byCompTime"],
                    "byAvgTime": r["byAvgTime"],
                }
            )

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
