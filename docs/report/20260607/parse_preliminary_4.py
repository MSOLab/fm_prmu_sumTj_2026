"""Parse per-instance subroutine_controller.log files from the
LBinit(NEH-MS) scenario run and build preliminary_4.csv.

Run from the repo root:
    uv run python docs/report/20260607/parse_preliminary_4.py

For each instance it extracts the four NEH-MS objective values produced by the
`compute_preemptive_last_stage_lb(init_method=neh-ms)` flow:
  1. NEH-MS(EDD)        -- baseline from `initialize_by_nehms`
  2. LBinit_start       -- LB-init NEH-MS, by start time sequence
  3. LBinit_completion  -- LB-init NEH-MS, by completion time sequence
  4. LBinit_average     -- LB-init NEH-MS, by average time sequence
The best LB-init schedule is compared against the NEH-MS(EDD) baseline and
against the VR2010 Best Solution (Total Tardiness).
"""

import csv
import glob
import os
import re
import statistics as st
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SCENARIO_DIR = os.path.join(
    HERE, "..", "..", "..",
    "Outputs_scenarios", "20260608T011058_692458", "20260607_00",
)
EVA_CSV = os.path.join(HERE, "Eva_Instances_EarlinessTardiness.csv")
OUT_CSV = os.path.join(HERE, "20260607_preliminary_4.csv")

RE_INIT = re.compile(r"Initialized by makespan with total tardiness (\d+)")
RE_LB = re.compile(r"MCF LB = (\d+)")
RE_START = re.compile(r"by start time sequence: (\d+)")
RE_COMP = re.compile(r"by completion time sequence: (\d+)")
RE_AVG = re.compile(r"by average time sequence: (\d+)")
RE_BEST = re.compile(r"best total tardiness is (\d+) by (\w[\w ]*?) sequence")

COLS = [
    "insName", "n", "m", "T", "R",
    "NEH-MS(EDD)", "LBinit_start", "LBinit_completion", "LBinit_average",
    "best_LBinit", "best_seq", "improve_vs_NEHMS", "improve_pct",
    "MCF_LB", "Best_VR2010", "delta_best_VR2010",
]


def load_eva():
    with open(EVA_CSV, encoding="utf-8-sig") as f:
        return {r["insName"]: r for r in csv.DictReader(f)}


def parse_instance(log_path, eva_row):
    log = open(log_path).read()
    nehms = int(RE_INIT.search(log).group(1))
    lb = int(RE_LB.search(log).group(1))
    s = int(RE_START.search(log).group(1))
    c = int(RE_COMP.search(log).group(1))
    a = int(RE_AVG.search(log).group(1))
    bm = RE_BEST.search(log)
    best_lbinit, best_seq = int(bm.group(1)), bm.group(2).strip()
    vr = int(eva_row["Best Solution (Total Tardiness)"])
    imp = nehms - best_lbinit
    return {
        "insName": eva_row["insName"],
        "n": eva_row["n"], "m": eva_row["m"],
        "T": eva_row["T"], "R": eva_row["R"],
        "NEH-MS(EDD)": nehms,
        "LBinit_start": s,
        "LBinit_completion": c,
        "LBinit_average": a,
        "best_LBinit": best_lbinit,
        "best_seq": best_seq,
        "improve_vs_NEHMS": imp,
        "improve_pct": round(imp / nehms * 100, 3),
        "MCF_LB": lb,
        "Best_VR2010": vr,
        "delta_best_VR2010": best_lbinit - vr,
    }


def main():
    eva = load_eva()
    rows = []
    for d in sorted(glob.glob(os.path.join(SCENARIO_DIR, "[0-9]*", ""))):
        ins = os.path.basename(d.rstrip(os.sep))
        rows.append(parse_instance(
            os.path.join(d, "subroutine_controller.log"), eva[ins]))

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV} ({len(rows)} instances)")

    print(f"mean improve vs NEH-MS(EDD): {round(st.mean(r['improve_vs_NEHMS'] for r in rows))}")
    print(f"mean improve %: {round(st.mean(r['improve_pct'] for r in rows), 3)}")
    print(f"best_seq counts: {dict(Counter(r['best_seq'] for r in rows))}")
    print(f"cases best_LBinit beats NEH-MS(EDD): "
          f"{sum(1 for r in rows if r['best_LBinit'] < r['NEH-MS(EDD)'])}/{len(rows)}")


if __name__ == "__main__":
    main()
