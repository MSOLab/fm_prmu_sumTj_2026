"""Tests for the asymmetric averaging in load_method_mean_metrics.

The key invariant: instances whose flow short-circuited before a given
method must still contribute to that method's RPDf via carry-forward,
even though they don't contribute to the time%.
"""

import tempfile
from pathlib import Path

import pandas as pd

from flowshop_tardiness.report.dashboards.method_mean_scatter import (
    load_method_mean_metrics,
)


def _write_summary_csv(path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_all_instances_ran_every_method_returns_symmetric_counts(tmp_path: Path):
    csv = tmp_path / "summary.csv"
    _write_summary_csv(
        csv,
        [
            {
                "instance_id": 1,
                "edd_end_sec": 1.0,
                "edd_obj_value": 100.0,
                "cp_end_sec": 50.0,
                "cp_obj_value": 80.0,
            },
            {
                "instance_id": 2,
                "edd_end_sec": 0.5,
                "edd_obj_value": 200.0,
                "cp_end_sec": 40.0,
                "cp_obj_value": 150.0,
            },
        ],
    )
    points = load_method_mean_metrics(
        csv,
        timelimit_by_instance={"1": 100.0, "2": 100.0},
        baseline_obj_by_instance={"1": 80.0, "2": 150.0},
        drop_non_improving_methods=False,
    )
    by_method = {p["method"]: p for p in points}
    # Both methods: every instance contributed to both axes
    for m in ("edd", "cp"):
        assert by_method[m]["time_instance_count"] == 2
        assert by_method[m]["rpdf_instance_count"] == 2


def test_skipped_method_carries_forward_obj_for_rpdf(tmp_path: Path):
    """Instance 2 hit a stopping condition after ``edd`` and never ran
    ``cp``. Its RPDf at the ``cp`` tick must still be the EDD RPDf — not
    excluded — so the mean_rpdf for ``cp`` reflects all instances."""
    csv = tmp_path / "summary.csv"
    _write_summary_csv(
        csv,
        [
            {
                "instance_id": 1,
                "edd_end_sec": 1.0,
                "edd_obj_value": 100.0,
                "cp_end_sec": 50.0,
                "cp_obj_value": 80.0,
            },
            {
                "instance_id": 2,
                "edd_end_sec": 0.5,
                "edd_obj_value": 200.0,
                "cp_end_sec": None,  # short-circuit before cp
                "cp_obj_value": None,
            },
        ],
    )
    points = load_method_mean_metrics(
        csv,
        timelimit_by_instance={"1": 100.0, "2": 100.0},
        baseline_obj_by_instance={"1": 80.0, "2": 200.0},
        drop_non_improving_methods=False,
    )
    by_method = {p["method"]: p for p in points}

    cp = by_method["cp"]
    # x-axis: only instance 1 ran cp
    assert cp["time_instance_count"] == 1
    # y-axis: both instances contribute (2 via carry-forward)
    assert cp["rpdf_instance_count"] == 2

    # Instance 2's carried-forward RPDf is (200 - 200) / ((200+200)/2) = 0
    # Instance 1's RPDf at cp is (80 - 80) / ((80+80)/2) = 0
    # Mean is 0
    assert abs(cp["mean_rpdf"]) < 1e-9
    # x mean is 50/100 = 0.5 (only instance 1 contributes)
    assert abs(cp["mean_time_pct"] - 0.5) < 1e-9


def test_skipped_method_carry_forward_lowers_mean_rpdf(tmp_path: Path):
    """The 'last point rises' case from ablation_c5: the hard instances
    that ran the late method had high RPDf, while the easy ones already
    short-circuited. Carry-forward must bring the y down."""
    csv = tmp_path / "summary.csv"
    # 3 easy instances short-circuit after edd (RPDf 0.05 each), 1 hard
    # instance runs cp and ends at RPDf 0.20.
    rows = []
    for i in range(1, 4):
        rows.append(
            {
                "instance_id": i,
                "edd_end_sec": 1.0,
                "edd_obj_value": 105.0,  # vs BKS 100 → RPDf ~0.0488
                "cp_end_sec": None,
                "cp_obj_value": None,
            }
        )
    rows.append(
        {
            "instance_id": 4,
            "edd_end_sec": 1.0,
            "edd_obj_value": 130.0,
            "cp_end_sec": 80.0,
            "cp_obj_value": 122.0,  # vs BKS 100 → RPDf ~0.1982
        }
    )
    _write_summary_csv(csv, rows)
    timelimits = {str(i): 100.0 for i in range(1, 5)}
    bkss = {str(i): 100.0 for i in range(1, 5)}
    points = load_method_mean_metrics(
        csv,
        timelimit_by_instance=timelimits,
        baseline_obj_by_instance=bkss,
        drop_non_improving_methods=False,
    )
    by_method = {p["method"]: p for p in points}
    cp = by_method["cp"]
    # x-axis: only instance 4 ran cp
    assert cp["time_instance_count"] == 1
    # y-axis: all 4 contribute
    assert cp["rpdf_instance_count"] == 4
    # mean_rpdf < 0.1 because 3 easy instances pulled it down
    assert cp["mean_rpdf"] < 0.1
    # Sanity: it's still positive (instance 4's contribution)
    assert cp["mean_rpdf"] > 0


def test_method_dropped_when_no_instance_recorded_end_sec(tmp_path: Path):
    """No instance ran the method → cannot place an x value → method
    is dropped entirely (even if carry-forward y would be defined)."""
    csv = tmp_path / "summary.csv"
    _write_summary_csv(
        csv,
        [
            {
                "instance_id": 1,
                "edd_end_sec": 1.0,
                "edd_obj_value": 100.0,
                "cp_end_sec": None,
                "cp_obj_value": None,
            },
            {
                "instance_id": 2,
                "edd_end_sec": 0.5,
                "edd_obj_value": 200.0,
                "cp_end_sec": None,
                "cp_obj_value": None,
            },
        ],
    )
    points = load_method_mean_metrics(
        csv,
        timelimit_by_instance={"1": 100.0, "2": 100.0},
        baseline_obj_by_instance={"1": 100.0, "2": 200.0},
        drop_non_improving_methods=False,
    )
    method_names = [p["method"] for p in points]
    assert "edd" in method_names
    assert "cp" not in method_names


def test_carry_forward_uses_prior_method_not_sibling_row(tmp_path: Path):
    """The carry-forward map must reflect the PREVIOUS method only —
    don't let earlier rows in this method's loop poison later rows."""
    csv = tmp_path / "summary.csv"
    _write_summary_csv(
        csv,
        [
            {
                "instance_id": 1,
                "edd_end_sec": 1.0,
                "edd_obj_value": 100.0,
                "cp_end_sec": 50.0,
                "cp_obj_value": 90.0,
            },
            {
                "instance_id": 2,
                "edd_end_sec": 1.0,
                "edd_obj_value": 200.0,
                "cp_end_sec": None,
                "cp_obj_value": None,
            },
        ],
    )
    points = load_method_mean_metrics(
        csv,
        timelimit_by_instance={"1": 100.0, "2": 100.0},
        baseline_obj_by_instance={"1": 90.0, "2": 200.0},
        drop_non_improving_methods=False,
    )
    by_method = {p["method"]: p for p in points}
    cp = by_method["cp"]
    # Instance 2's carry-forward must be its OWN EDD obj (200), not
    # instance 1's cp obj (90). RPDf at BKS=200 is 0.
    # Instance 1's RPDf at cp obj=90, BKS=90: 0.
    # So mean_rpdf should be 0, not affected by sibling row.
    assert abs(cp["mean_rpdf"]) < 1e-9


def test_drop_non_improving_methods_still_works(tmp_path: Path):
    """A snapshot method whose obj == prior for every instance is dropped
    when drop_non_improving_methods is on."""
    csv = tmp_path / "summary.csv"
    _write_summary_csv(
        csv,
        [
            {
                "instance_id": 1,
                "edd_end_sec": 1.0,
                "edd_obj_value": 100.0,
                "snapshot_end_sec": 1.5,
                "snapshot_obj_value": 100.0,  # equal to prior
                "cp_end_sec": 50.0,
                "cp_obj_value": 80.0,
            }
        ],
    )
    points = load_method_mean_metrics(
        csv,
        timelimit_by_instance={"1": 100.0},
        baseline_obj_by_instance={"1": 80.0},
        drop_non_improving_methods=True,
    )
    methods_kept = [p["method"] for p in points]
    assert "snapshot" not in methods_kept
    assert "edd" in methods_kept
    assert "cp" in methods_kept
