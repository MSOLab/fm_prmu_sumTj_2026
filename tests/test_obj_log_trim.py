import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from flowshop_tardiness.report.dashboards.obj_log_loader import (
    CallSegment,
    ProgPoint,
    _build_calls_for_series,
    _truncate_calls_to_timelimit,
    load_instance_progression,
)
from flowshop_tardiness.report.dashboards.obj_log_trim import (
    _last_value_at_or_before,
    _resolve_obj_log_path,
    apply_timelimit_trim,
)


# ---------- _last_value_at_or_before ----------


def test_last_value_picks_latest_within_threshold():
    data = {
        "0.5": 100.0,
        "2.0": 80.0,
        "5.0": 50.0,
    }
    assert _last_value_at_or_before(data, 3.0) == 80.0


def test_last_value_skips_entries_after_threshold():
    data = {"1.0": 10.0, "5.0": 5.0, "9.0": 1.0}
    assert _last_value_at_or_before(data, 5.0) == 5.0
    assert _last_value_at_or_before(data, 4.999) == 10.0


def test_last_value_returns_none_when_all_after_threshold():
    data = {"10.0": 5.0, "20.0": 1.0}
    assert _last_value_at_or_before(data, 9.999) is None


def test_last_value_handles_empty_data():
    assert _last_value_at_or_before({}, 100.0) is None


def test_last_value_ignores_non_numeric_keys():
    data = {"1.0": 10.0, "not_a_number": 99.0, "3.0": 7.0}
    assert _last_value_at_or_before(data, 5.0) == 7.0


def test_last_value_handles_unordered_dict():
    data = {"5.0": 50.0, "1.0": 100.0, "3.0": 70.0}
    assert _last_value_at_or_before(data, 4.0) == 70.0


# ---------- _resolve_obj_log_path ----------


def test_resolve_path_prefers_results_subdir(tmp_path: Path):
    scenario = "scn"
    ins = "42"
    base = tmp_path / scenario / ins
    results = base / "results"
    results.mkdir(parents=True)
    (results / "42_obj_log.yaml").write_text("obj_value: {}")
    (base / "42_obj_log.yaml").write_text("ignored: true")

    resolved = _resolve_obj_log_path(tmp_path, scenario, ins)
    assert resolved == results / "42_obj_log.yaml"


def test_resolve_path_falls_back_to_flat(tmp_path: Path):
    base = tmp_path / "scn" / "5"
    base.mkdir(parents=True)
    (base / "5_obj_log.yaml").write_text("obj_value: {}")

    resolved = _resolve_obj_log_path(tmp_path, "scn", "5")
    assert resolved == base / "5_obj_log.yaml"


def test_resolve_path_returns_none_when_missing(tmp_path: Path):
    assert _resolve_obj_log_path(tmp_path, "scn", "999") is None


# ---------- apply_timelimit_trim ----------


def _write_obj_log(
    run_dir: Path,
    scenario: str,
    ins_name: str,
    obj_value_data: dict,
    obj_bound_data: dict | None = None,
) -> Path:
    results = run_dir / scenario / ins_name / "results"
    results.mkdir(parents=True, exist_ok=True)
    path = results / f"{ins_name}_obj_log.yaml"
    payload = {
        "obj_value": {"data": obj_value_data},
        "obj_bound": {"data": obj_bound_data or {}},
    }
    with path.open("w") as f:
        yaml.safe_dump(payload, f)
    return path


def test_trim_overwrites_when_solver_overran(tmp_path: Path):
    """Solver exceeded timelimit; trim picks the value at the deadline."""
    _write_obj_log(
        tmp_path,
        "scn",
        "1",
        obj_value_data={
            "10.0": 1000.0,
            "100.0": 800.0,
            "500.0": 700.0,  # past timelimit
            "1000.0": 650.0,  # past timelimit
        },
        obj_bound_data={"10.0": 0.0, "200.0": 50.0, "1000.0": 100.0},
    )

    df = pd.DataFrame(
        [
            {
                "scenario": "scn",
                "insName": "1",
                "timelimit": 300.0,
                "bestObj": 650.0,
                "bestBound": 100.0,
                "totalElapsedTime": 1000.0,
                "initObj": 1000.0,
                "improvementRatio": 0.35,
            }
        ]
    )

    out = apply_timelimit_trim(df, tmp_path)

    assert out.loc[0, "bestObj"] == 800.0  # last value at t<=300
    assert out.loc[0, "bestBound"] == 50.0  # last bound at t<=300
    assert out.loc[0, "totalElapsedTime"] == 300.0  # capped
    # Originals preserved
    assert out.loc[0, "bestObj_endpoint"] == 650.0
    assert out.loc[0, "bestBound_endpoint"] == 100.0
    assert out.loc[0, "totalElapsedTime_endpoint"] == 1000.0
    # improvementRatio recomputed: (1000 - 800) / 1000 = 0.2
    assert out.loc[0, "improvementRatio"] == pytest.approx(0.2)


def test_trim_no_op_when_solver_finished_within_limit(tmp_path: Path):
    """If the run finished inside the budget, all logged values stay; bestObj
    becomes the final logged value, totalElapsedTime is below cap."""
    _write_obj_log(
        tmp_path,
        "scn",
        "2",
        obj_value_data={"1.0": 100.0, "5.0": 80.0, "9.0": 60.0},
        obj_bound_data={"1.0": 0.0, "9.0": 30.0},
    )

    df = pd.DataFrame(
        [
            {
                "scenario": "scn",
                "insName": "2",
                "timelimit": 60.0,
                "bestObj": 60.0,
                "bestBound": 30.0,
                "totalElapsedTime": 9.0,
                "initObj": 100.0,
                "improvementRatio": 0.4,
            }
        ]
    )

    out = apply_timelimit_trim(df, tmp_path)

    assert out.loc[0, "bestObj"] == 60.0
    assert out.loc[0, "bestBound"] == 30.0
    assert out.loc[0, "totalElapsedTime"] == 9.0  # min(9.0, 60.0)
    assert out.loc[0, "totalElapsedTime_endpoint"] == 9.0


def test_trim_leaves_row_untouched_when_obj_log_missing(tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "scenario": "scn",
                "insName": "missing",
                "timelimit": 100.0,
                "bestObj": 42.0,
                "bestBound": 0.0,
                "totalElapsedTime": 200.0,
                "initObj": 100.0,
                "improvementRatio": 0.58,
            }
        ]
    )

    out = apply_timelimit_trim(df, tmp_path)

    assert out.loc[0, "bestObj"] == 42.0
    assert out.loc[0, "bestBound"] == 0.0
    assert out.loc[0, "totalElapsedTime"] == 200.0  # not capped, no log
    assert out.loc[0, "bestObj_endpoint"] == 42.0


def test_trim_picks_final_logged_value_when_recorded_just_after_limit(
    tmp_path: Path,
):
    """Real case from instance 1: final value recorded at t > timelimit must
    be excluded; the predecessor value at t <= timelimit is the trimmed one."""
    _write_obj_log(
        tmp_path,
        "scn",
        "1",
        obj_value_data={
            "21.0": 5427.0,
            "22.0": 5427.0,
            "22.55": 5419.0,  # past timelimit=22.5
        },
    )

    df = pd.DataFrame(
        [
            {
                "scenario": "scn",
                "insName": "1",
                "timelimit": 22.5,
                "bestObj": 5419.0,
                "bestBound": 0.0,
                "totalElapsedTime": 22.55,
                "initObj": 11985.0,
                "improvementRatio": 0.547,
            }
        ]
    )

    out = apply_timelimit_trim(df, tmp_path)
    assert out.loc[0, "bestObj"] == 5427.0
    assert out.loc[0, "bestObj_endpoint"] == 5419.0


def test_trim_handles_multiple_rows_and_scenarios(tmp_path: Path):
    _write_obj_log(tmp_path, "a", "1", {"5.0": 100.0, "50.0": 80.0})
    _write_obj_log(tmp_path, "a", "2", {"5.0": 200.0, "50.0": 150.0})
    _write_obj_log(tmp_path, "b", "1", {"5.0": 300.0, "9.0": 250.0})

    df = pd.DataFrame(
        [
            {
                "scenario": "a",
                "insName": "1",
                "timelimit": 10.0,
                "bestObj": 80.0,
                "bestBound": 0.0,
                "totalElapsedTime": 50.0,
            },
            {
                "scenario": "a",
                "insName": "2",
                "timelimit": 100.0,
                "bestObj": 150.0,
                "bestBound": 0.0,
                "totalElapsedTime": 50.0,
            },
            {
                "scenario": "b",
                "insName": "1",
                "timelimit": 10.0,
                "bestObj": 250.0,
                "bestBound": 0.0,
                "totalElapsedTime": 9.0,
            },
        ]
    )

    out = apply_timelimit_trim(df, tmp_path)

    assert out.loc[0, "bestObj"] == 100.0  # a/1 trimmed (50s > 10s limit)
    assert out.loc[1, "bestObj"] == 150.0  # a/2 within limit
    assert out.loc[2, "bestObj"] == 250.0  # b/1 within limit


def test_trim_skips_when_required_columns_missing(tmp_path: Path):
    df = pd.DataFrame([{"scenario": "x", "insName": "1", "bestObj": 100.0}])
    out = apply_timelimit_trim(df, tmp_path)
    # No mutation, no added columns
    assert "bestObj_endpoint" not in out.columns
    assert out.loc[0, "bestObj"] == 100.0


def test_trim_handles_empty_dataframe(tmp_path: Path):
    df = pd.DataFrame()
    out = apply_timelimit_trim(df, tmp_path)
    assert out.empty


def test_trim_preserves_original_when_no_value_recorded_before_limit(
    tmp_path: Path,
):
    """If all logged points are past timelimit, keep the original bestObj."""
    _write_obj_log(tmp_path, "scn", "1", {"500.0": 100.0, "1000.0": 50.0})

    df = pd.DataFrame(
        [
            {
                "scenario": "scn",
                "insName": "1",
                "timelimit": 100.0,
                "bestObj": 50.0,
                "bestBound": 0.0,
                "totalElapsedTime": 1000.0,
            }
        ]
    )

    out = apply_timelimit_trim(df, tmp_path)
    # No value <= 100s in log → preserve original bestObj
    assert out.loc[0, "bestObj"] == 50.0
    # But elapsed is still capped because we did read the log successfully
    assert out.loc[0, "totalElapsedTime"] == 100.0


# ---------- _truncate_calls_to_timelimit ----------


def _seg(idx, name, start, end, points):
    return CallSegment(
        call_index=idx,
        subroutine_name=name,
        prefixed_subroutine_name=f"{idx}-{name}",
        global_start_sec=start,
        global_end_sec=end,
        points=tuple(ProgPoint(global_sec=t, value=v) for t, v in points),
    )


def test_truncate_no_op_when_all_calls_within_budget():
    calls = (
        _seg(1, "edd", 0.0, 1.0, [(1.0, 100.0)]),
        _seg(2, "cp", 1.0, 50.0, [(50.0, 80.0)]),
    )
    out = _truncate_calls_to_timelimit(calls, 100.0)
    assert out == calls


def test_truncate_drops_calls_starting_after_deadline():
    calls = (
        _seg(1, "edd", 0.0, 1.0, [(0.5, 200.0), (1.0, 100.0)]),
        _seg(2, "cp", 1.0, 200.0, [(150.0, 50.0)]),
        _seg(3, "post", 200.0, 300.0, [(300.0, 25.0)]),
    )
    out = _truncate_calls_to_timelimit(calls, 100.0)
    # Call 1: ends at 1.0, fully within budget → kept
    # Call 2: 1.0→200.0 straddles deadline; no point at t<=100, carry call 1's
    #         last value (100.0) forward as synthetic endpoint at t=100.
    # Call 3: starts at 200 >= 100 → dropped (and break terminates loop early)
    assert len(out) == 2
    assert out[0].global_end_sec == 1.0
    assert out[1].global_end_sec == 100.0
    assert out[1].points[-1].value == 100.0


def test_truncate_straddling_call_keeps_pre_deadline_points():
    """Call with points before and after the deadline keeps only the ones at
    or before, and the call end is moved to the deadline."""
    calls = (
        _seg(
            1,
            "cp",
            0.0,
            100.0,
            [(10.0, 1000.0), (30.0, 800.0), (60.0, 700.0), (90.0, 650.0)],
        ),
    )
    out = _truncate_calls_to_timelimit(calls, 50.0)
    assert len(out) == 1
    call = out[0]
    assert call.global_start_sec == 0.0
    assert call.global_end_sec == 50.0
    assert [(p.global_sec, p.value) for p in call.points] == [
        (10.0, 1000.0),
        (30.0, 800.0),
    ]


def test_truncate_synthesizes_endpoint_when_no_point_in_window():
    """The 297/c1 case: call straddles deadline but has no recorded point
    within the truncated window (only an endpoint past the deadline).
    Should carry the last value forward and place it at timelimit."""
    calls = (
        _seg(1, "edd", 0.0, 0.5, [(0.5, 1159047.0)]),
        # solve_base_cp_model: starts at 264.83 (after prev_end), ends at 4155
        _seg(2, "solve", 264.83, 4155.59, [(4155.59, 1159047.0)]),
    )
    out = _truncate_calls_to_timelimit(calls, 787.5)
    assert len(out) == 2
    truncated = out[1]
    assert truncated.global_end_sec == 787.5
    assert truncated.global_start_sec == 264.83
    # Carry-forward from the previous call's last value
    assert [(p.global_sec, p.value) for p in truncated.points] == [
        (787.5, 1159047.0)
    ]


def test_truncate_returns_calls_unchanged_for_invalid_timelimit():
    calls = (_seg(1, "edd", 0.0, 1.0, [(1.0, 100.0)]),)
    assert _truncate_calls_to_timelimit(calls, 0.0) == calls
    assert _truncate_calls_to_timelimit(calls, -1.0) == calls
    assert _truncate_calls_to_timelimit(calls, None) == calls  # type: ignore[arg-type]
    assert _truncate_calls_to_timelimit((), 100.0) == ()


def test_truncate_drops_call_with_no_data_and_no_prior_value():
    """If the first call has no points falling within (0, timelimit] and
    there's no prior value to carry forward, drop the call entirely."""
    calls = (_seg(1, "cp", 0.0, 100.0, [(100.0, 50.0)]),)
    # timelimit=50 means the only point (at t=100) is past deadline
    out = _truncate_calls_to_timelimit(calls, 50.0)
    assert out == ()


# ---------- load_instance_progression integration ----------


def test_load_instance_progression_applies_truncation(tmp_path: Path):
    """Real obj_log + timelimit → progression frames are bounded."""
    obj_log = tmp_path / "297_obj_log.yaml"
    obj_log.write_text(
        "obj_value:\n"
        "  data:\n"
        "    '0.5': 1159047\n"
        "    '264.83': 1159047\n"
        "    '4155.59': 1159047\n"
        "  notes:\n"
        "    '0.5': 2-initialize_by_edd\n"
        "    '264.83': 4-solve_base_cp_model\n"
        "    '4155.59': 4-solve_base_cp_model\n"
    )
    prog = load_instance_progression(
        obj_log,
        instance_id="297",
        job_cnt=350,
        stage_cnt=50,
        timelimit_sec=787.5,
    )
    # Every kept call must end at or before the deadline.
    for call in prog.obj_value_calls:
        assert call.global_end_sec <= 787.5, (
            f"call {call.call_index} ended at {call.global_end_sec}"
        )
        for pt in call.points:
            assert pt.global_sec <= 787.5
