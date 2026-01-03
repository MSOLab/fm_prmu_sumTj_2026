import copy

import pytest

from flowshop_tardiness.fm_prmu import PermutationFlowshopScheduleLight


def total_tardiness(sch: PermutationFlowshopScheduleLight) -> int:
    """Compute total tardiness from the schedule's current end-time map."""
    last_stage = sch._stage_name_list[-1]
    tot = 0
    for j in sch._job_seq:
        due = sch._job_2_due_map.get(j, None)
        if due is None:
            continue
        c = sch._stage_2_job_2_end_map[last_stage][j]
        tot += max(0, c - due)
    return tot


def start_time(sch: PermutationFlowshopScheduleLight, stage: str, job: str) -> int:
    return sch._stage_2_job_2_end_map[stage][job] - sch._job_2_stage_2_p_map[job][stage]


def assert_feasible_flowshop(sch: PermutationFlowshopScheduleLight) -> None:
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


def build_sample_instance():
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

    sch = PermutationFlowshopScheduleLight(stages, p, due)
    for j in jobs:
        sch.append_job(j)

    return sch


def snapshot_end_map(sch: PermutationFlowshopScheduleLight):
    # deep copy nested dict: stage -> (job -> end)
    return {s: copy.deepcopy(m) for s, m in sch._stage_2_job_2_end_map.items()}


def test_push_back_keeps_total_tardiness_and_feasible():
    sch = build_sample_instance()
    before_tt = total_tardiness(sch)
    before_map = snapshot_end_map(sch)

    # push back last 3 jobs: J4,J5,J6
    sch.push_back_tail_jobs_keep_total_tardiness(tail_job_cnt=3)

    after_tt = total_tardiness(sch)
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
        sch.push_back_tail_jobs_keep_total_tardiness(0)
    with pytest.raises(ValueError):
        sch.push_back_tail_jobs_keep_total_tardiness(999)


def test_push_back_does_not_decrease_any_end_time_on_tail():
    """
    Ensure that pushing back tail jobs does not result in any of their end times decreasing.
    """
    sch = build_sample_instance()
    before_map = snapshot_end_map(sch)

    tail_cnt = 3
    tail_jobs = sch._job_seq[-tail_cnt:]

    sch.push_back_tail_jobs_keep_total_tardiness(tail_cnt)

    for s in sch._stage_name_list:
        for j in tail_jobs:
            assert sch._stage_2_job_2_end_map[s][j] >= before_map[s][j], (
                f"End time decreased for tail job: stage={s}, job={j}"
            )
