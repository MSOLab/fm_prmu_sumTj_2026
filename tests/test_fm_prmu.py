import copy

import pytest

from flowshop_tardiness.fm_prmu import PermutationFlowshopScheduleLite


def start_time(sch: PermutationFlowshopScheduleLite, stage: str, job: str) -> int:
    return sch._stage_2_job_2_end_map[stage][job] - sch._job_2_stage_2_p_map[job][stage]


def assert_feasible_flowshop(sch: PermutationFlowshopScheduleLite) -> None:
    """Check feasibility w.r.t. precedence (within job) and machine capacity (within stage)."""
    stages = sch._stage_name_list
    jobs = sch._job_seq

    # (A) within each job: end(prev_stage) <= start(next_stage)
    for j in jobs:
        for s_prev, s_next in zip(stages[:-1], stages[1:]):
            end_prev = sch._stage_2_job_2_end_map[s_prev][j]
            st_next = start_time(sch, s_next, j)
            assert end_prev <= st_next, (
                f"Job precedence violated for job={j}: {s_prev}->{s_next}"
            )

    # (B) within each stage: end(prev_job) <= start(next_job)
    for s in stages:
        for j_prev, j_next in zip(jobs[:-1], jobs[1:]):
            end_prev = sch._stage_2_job_2_end_map[s][j_prev]
            st_next = start_time(sch, s, j_next)
            assert end_prev <= st_next, (
                f"Machine capacity violated at stage={s}: {j_prev}->{j_next}"
            )


def build_sample_instance() -> PermutationFlowshopScheduleLite:
    stages = ["M1", "M2", "M3"]
    jobs = [f"J{k}" for k in range(1, 7)]

    # processing times (job -> stage -> p)
    p = {
        "J1": {"M1": 3, "M2": 2, "M3": 4},
        "J2": {"M1": 2, "M2": 5, "M3": 1},
        "J3": {"M1": 4, "M2": 1, "M3": 3},
        "J4": {"M1": 2, "M2": 2, "M3": 2},
        "J5": {"M1": 5, "M2": 3, "M3": 2},
        "J6": {"M1": 1, "M2": 4, "M3": 3},
    }

    # due dates: designed to have both early and tardy jobs
    due = {
        "J1": 20,
        "J2": 18,
        "J3": 19,
        "J4": 17,
        "J5": 30,
        "J6": 22,
    }

    sch = PermutationFlowshopScheduleLite(stages, p, due)
    for j in jobs:
        sch.append_job(j)

    return sch


def snapshot_end_map(sch: PermutationFlowshopScheduleLite):
    # deep copy nested dict: stage -> (job -> end)
    return {s: copy.deepcopy(m) for s, m in sch._stage_2_job_2_end_map.items()}


def test_push_back_keeps_total_tardiness_and_feasible():
    sch = build_sample_instance()
    before_tt = sch.get_total_tardiness()
    before_map = snapshot_end_map(sch)

    # push back last 3 jobs: J4,J5,J6
    sch.push_back_tail_jobs_keep_tardiness(tail_job_cnt=3)

    after_tt = sch.get_total_tardiness()
    assert after_tt == before_tt, "Total tardiness must remain unchanged."

    # feasibility should still hold
    assert_feasible_flowshop(sch)

    # prefix jobs (J1..J3) must remain unchanged (method should only affect tail)
    prefix = sch._job_seq[:-3]
    for s in sch._stage_name_list:
        for j in prefix:
            assert sch._stage_2_job_2_end_map[s][j] == before_map[s][j], (
                f"Prefix job end time changed unexpectedly: stage={s}, job={j}"
            )


def test_push_back_invalid_tail_job_cnt_raises():
    sch = build_sample_instance()
    with pytest.raises(ValueError):
        sch.push_back_tail_jobs_keep_tardiness(0)
    with pytest.raises(ValueError):
        sch.push_back_tail_jobs_keep_tardiness(999)


def test_push_back_does_not_decrease_any_end_time_on_tail():
    """
    Ensure that pushing back tail jobs does not result in any of their end times decreasing.
    """
    sch = build_sample_instance()
    before_map = snapshot_end_map(sch)

    tail_cnt = 3
    tail_jobs = sch._job_seq[-tail_cnt:]

    sch.push_back_tail_jobs_keep_tardiness(tail_cnt)

    for s in sch._stage_name_list:
        for j in tail_jobs:
            assert sch._stage_2_job_2_end_map[s][j] >= before_map[s][j], (
                f"End time decreased for tail job: stage={s}, job={j}"
            )


def test_simulate_append_with_est_map():
    """Test simulate_append with stage_2_est_map parameter."""
    stages = ["M1", "M2", "M3"]
    p = {
        "J1": {"M1": 3, "M2": 2, "M3": 4},
        "J2": {"M1": 2, "M2": 5, "M3": 1},
    }
    sch = PermutationFlowshopScheduleLite(stages, p)

    # Append first job without est_map
    sch.append_job("J1")

    # Simulate appending second job with no est_map
    end_times_no_est = sch.simulate_append("J2")
    assert len(end_times_no_est) == 3
    assert end_times_no_est["M1"] == 5  # M1: 3 + 2
    assert end_times_no_est["M2"] == 10  # M2: max(5, 5) + 5 = 10
    assert end_times_no_est["M3"] == 11  # M3: max(10, 10) + 1 = 11

    # Simulate appending with est_map (earliest start times)
    est_map = {"M1": 10, "M2": 15, "M3": 21}
    end_times_with_est = sch.simulate_append("J2", stage_2_est_map=est_map)
    assert len(end_times_with_est) == 3
    assert end_times_with_est["M1"] == 12  # M1: max(5, 10) + 2 = 12
    assert end_times_with_est["M2"] == 20  # M2: max(12, 15) + 5 = 20
    assert end_times_with_est["M3"] == 22  # M3: max(20, 21) + 1 = 22


def test_get_total_tardiness_basic():
    """Test get_total_tardiness calculation."""
    stages = ["M1", "M2"]
    p = {
        "J1": {"M1": 5, "M2": 3},
        "J2": {"M1": 4, "M2": 2},
        "J3": {"M1": 2, "M2": 6},
    }
    due = {
        "J1": 10,
        "J2": 15,
        "J3": 12,
    }

    sch = PermutationFlowshopScheduleLite(stages, p, due)
    for j in ["J1", "J2", "J3"]:
        sch.append_job(j)

    # J1: end at M2 = 8, due = 10, tardiness = 0
    # J2: end at M2 = 11, due = 15, tardiness = 0
    # J3: end at M2 = 17, due = 12, tardiness = 5
    # total = 5
    tt = sch.get_total_tardiness()
    assert tt == 5


def test_get_total_tardiness_all_on_time():
    """Test get_total_tardiness when all jobs are on time."""
    stages = ["M1", "M2"]
    p = {
        "J1": {"M1": 2, "M2": 2},
        "J2": {"M1": 2, "M2": 2},
    }
    due = {
        "J1": 10,
        "J2": 20,
    }

    sch = PermutationFlowshopScheduleLite(stages, p, due)
    for j in ["J1", "J2"]:
        sch.append_job(j)

    # J1: end = 4, due = 10, tardiness = 0
    # J2: end = 8, due = 20, tardiness = 0
    tt = sch.get_total_tardiness()
    assert tt == 0


def test_get_total_tardiness_all_tardy():
    """Test get_total_tardiness when all jobs are tardy."""
    stages = ["M1", "M2"]
    p = {
        "J1": {"M1": 10, "M2": 10},
        "J2": {"M1": 10, "M2": 10},
    }
    due = {
        "J1": 5,
        "J2": 15,
    }

    sch = PermutationFlowshopScheduleLite(stages, p, due)
    for j in ["J1", "J2"]:
        sch.append_job(j)

    # J1: end = 20, due = 5, tardiness = 15
    # J2: end = 30, due = 15, tardiness = 15
    tt = sch.get_total_tardiness()
    assert tt == 30


def test_get_total_tardiness_no_due_dates():
    """Test get_total_tardiness when no due dates are specified."""
    stages = ["M1", "M2"]
    p = {
        "J1": {"M1": 5, "M2": 3},
        "J2": {"M1": 4, "M2": 2},
    }

    sch = PermutationFlowshopScheduleLite(stages, p)
    for j in ["J1", "J2"]:
        sch.append_job(j)

    # No due dates specified, so all tardiness should be 0
    tt = sch.get_total_tardiness()
    assert tt == 0


def test_get_total_tardiness_partial_due_dates():
    """Test get_total_tardiness when only some jobs have due dates."""
    stages = ["M1", "M2"]
    p = {
        "J1": {"M1": 5, "M2": 3},
        "J2": {"M1": 4, "M2": 2},
        "J3": {"M1": 2, "M2": 6},
    }
    due = {
        "J1": 5,
        "J3": 10,
    }

    sch = PermutationFlowshopScheduleLite(stages, p, due)
    for j in ["J1", "J2", "J3"]:
        sch.append_job(j)

    # J1: end = 8, due = 5, tardiness = 3
    # J2: no due date, treated as 0, tardiness = max(0, 11 - 0) = 11
    # J3: end = 17, due = 10, tardiness = 7
    # Total = 3 + 11 + 7 = 21
    tt = sch.get_total_tardiness()
    assert tt == 21


def test_simulate_append_empty_schedule():
    """Test simulate_append on empty schedule (no previous jobs)."""
    stages = ["M1", "M2", "M3"]
    p = {
        "J1": {"M1": 3, "M2": 4, "M3": 2},
    }

    sch = PermutationFlowshopScheduleLite(stages, p)

    # Simulate appending to empty schedule with no est_map
    end_times = sch.simulate_append("J1")
    assert end_times["M1"] == 3  # 3
    assert end_times["M2"] == 7  # 3 + 4
    assert end_times["M3"] == 9  # 7 + 2


def test_simulate_append_empty_schedule_with_est_map():
    """Test simulate_append on empty schedule with est_map."""
    stages = ["M1", "M2", "M3"]
    p = {
        "J1": {"M1": 3, "M2": 4, "M3": 2},
    }

    sch = PermutationFlowshopScheduleLite(stages, p)

    # est_map with high values on empty schedule
    est_map = {"M1": 10, "M2": 15, "M3": 20}
    end_times = sch.simulate_append("J1", stage_2_est_map=est_map)
    assert end_times["M1"] == 13  # 10 + 3
    assert end_times["M2"] == 19  # max(13, 15)+4
    assert end_times["M3"] == 22  # max(19, 20)+2
