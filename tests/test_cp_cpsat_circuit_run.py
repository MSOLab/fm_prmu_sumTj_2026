from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from schore.parameters_examples.shop.flow.flowshop_duedate import (
    FlowshopDuedateParameters,
)
from schore.schedule_examples.shop.flow import FlowshopOperation, FlowshopSchedule

from flowshop_tardiness.cp_cpsat_circuit import CpCpsatCircuit
from flowshop_tardiness.painter.gantt import GanttPlotter


def test_cp_cpsat_circuit_runs_and_builds_schedule():
    # Ensure repository root is on sys.path so imports using package layout work
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Load VRM instance from resources/vrm/0.txt
    vrm_path = repo_root / "resources" / "vrm" / "0.txt"
    assert vrm_path.exists(), f"VRM file not found: {vrm_path}"

    with vrm_path.open("r") as f:
        instance = FlowshopDuedateParameters.from_vrm_data("test", f)

    # Build CP model with a large horizon
    horizon = 100000
    model = CpCpsatCircuit.from_instance(instance, horizon)

    # Solve the model
    timelimit = 60  # seconds
    num_workers = 8
    (solver_status, elapsed_time, obj_value, obj_bound) = model.solve_with_callbacks(
        timelimit, num_workers
    )

    # Log and persist solver report for inspection
    solver_status_name = (
        solver_status.name if hasattr(solver_status, "name") else str(solver_status)
    )
    report = {
        "solver_status": solver_status_name,
        "elapsed_time": float(elapsed_time) if elapsed_time is not None else None,
        "obj_value": float(obj_value) if obj_value is not None else None,
        "obj_bound": float(obj_bound) if obj_bound is not None else None,
    }
    logging.basicConfig(level=logging.INFO)
    logging.info("CP solve result: %s", report)

    # save to file in repo root for easy access after test
    with (repo_root / "cp_solve_report.json").open("w") as rf:
        json.dump(report, rf, indent=2)

    # extract start/end maps
    start_map, end_map = model.extract_start_end_time_map()

    # Plot
    GanttPlotter().export_flowshop_plot(Path("test_output.png"), start_map, end_map)

    # Build FlowshopSchedule and populate operations
    i_list = instance.stage_id_list
    schedule = FlowshopSchedule.from_stage_name_list(i_list)

    for j in instance.job_id_list:
        for i in i_list:
            s = int(start_map[j, i])
            e = int(end_map[j, i])
            op = FlowshopOperation(job_name=j, stage_name=i, start=s, end=e)
            added = schedule.schedule_operation(op)
            assert added is not None, f"Failed to add operation {j},{i} to schedule"

    # Basic sanity checks
    assert schedule.makespan <= horizon

    total_ops = sum(len(stage.operations) for stage in schedule._stages.values())
    assert total_ops == len(instance.job_id_list) * len(instance.stage_id_list)

    # Check objective value
    total_tardiness = schedule.get_total_tardiness(instance.job_2_duedate_map)
    assert total_tardiness == obj_value

    # Check permutation schedule: each stage should have the same job sequence
    i_2_j_list_map = schedule.get_stage_2_job_list_map()
    reference_sequence = i_2_j_list_map[i_list[0]]
    for stage_name in i_list[1:]:
        if i_2_j_list_map[stage_name] != reference_sequence:
            raise ValueError(
                f"Job sequence mismatch between stage {i_list[0]} & {stage_name}."
            )
