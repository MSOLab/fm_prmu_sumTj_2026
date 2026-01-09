from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from routix import DynamicDataObject, ElapsedTimer, StoppingCriteria
from schore.parameters_examples.shop.flow import FlowshopDuedateParameters
from schore.schedule_examples.shop.flow import FlowshopOperation, FlowshopSchedule

from ..cplex_model.model import TBB2018Data, TBB2018MilpModelBuilder
from ..report import FsSubroutineReport
from .base_flowshop_controller import BaseFlowshopController


class FlowshopTardinessCplexMatheuristicController(BaseFlowshopController):
    """Matheuristic controller using CPLEX via docplex.

    Current scope (minimal, runner-compatible):
    - builds TaEtAl2018 MILP (assignment/position-based permutation flow shop)
    - solves it with a time limit
    - decodes the permutation, dispatches a schedule, registers it

    Notes:
    - This controller is solver-agnostic at the base level, but this subclass
      assumes docplex is installed and CPLEX runtime is available.
    """

    def __init__(
        self,
        instance: FlowshopDuedateParameters,
        shared_param_dict: dict,
        subroutine_flow: Sequence[DynamicDataObject] | DynamicDataObject,
        stopping_criteria: StoppingCriteria,
    ):
        super().__init__(
            instance=instance,
            shared_param_dict=shared_param_dict,
            subroutine_flow=subroutine_flow,
            stopping_criteria=stopping_criteria,
        )
        self.method_names_to_run_before_resume = {"set_random_seed"}

    # -----------------
    # Core subroutines
    # -----------------

    def solve_base_milp(
        self,
        computational_time: float,
        solver_thread_cnt: int = 1,
        mip_gap: float | None = None,
        use_mip_start_from_incumbent: bool = True,
        error_if_infeasible: bool = False,
    ) -> None:
        """Build and solve the baseline MILP model.

        Intended to be used as a subroutine-flow step.

        Args:
            computational_time: time limit for this call (seconds).
            solver_thread_cnt: CPLEX threads.
            mip_gap: optional MIP gap.
            use_mip_start_from_incumbent: add MIP start if incumbent exists.
            error_if_infeasible: if True, raise when decoded schedule is infeasible.
        """
        sub_timer = ElapsedTimer()

        incumbent_schedule = self.solution_manager.get_incumbent()
        is_initial_solution = incumbent_schedule is None

        data = self._build_tbb2018_data()
        builder = TBB2018MilpModelBuilder(data, model_name=f"milp_{self.instance.name}")

        incumbent_perm = None
        if use_mip_start_from_incumbent and isinstance(incumbent_schedule, FlowshopSchedule):
            incumbent_perm = self._perm_indices_from_schedule(incumbent_schedule)

        timelimit = self.get_remaining_time_limit(computational_time)

        mdl, variables = builder.build(
            time_limit_s=timelimit,
            threads=solver_thread_cnt,
            mip_gap=mip_gap,
            incumbent_perm=incumbent_perm,
        )

        sol = mdl.solve()
        if sol is None:
            logging.warning(
                f"MILP solve returned no solution for {self.instance.name} (status={mdl.solve_details.status})."
            )
            return

        perm_idx = builder.extract_permutation_from_solution(sol, variables.x, data.n)
        job_seq = [self.instance.job_id_list[j] for j in perm_idx]

        schedule = self._dispatch_permutation(job_seq)
        obj_value = self.get_obj_value(schedule)

        if error_if_infeasible:
            self.check_feasibility(schedule)

        obj_bound = getattr(mdl.solve_details, "best_bound", None)
        if obj_bound is not None:
            try:
                obj_bound = float(obj_bound)
            except Exception:
                obj_bound = None

        report = FsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=float(obj_value),
            obj_bound=float(obj_bound) if obj_bound is not None else None,
            is_init=is_initial_solution,
        )

        was_updated = self.solution_manager.register(report, schedule)

        # Objective time series logs (only record if improving)
        log_time = self.timer.elapsed_sec
        self.obj_store.add_obj_value(log_time, float(obj_value), is_maximize=False)
        if obj_bound is not None:
            # For minimization, higher lower-bound is better
            self.obj_store.add_obj_bound(log_time, float(obj_bound), is_maximize=False)

        self.obj_store.add_last_timestamp_note(
            self._get_call_context_of_current_method(),
            obj_value_is_valid=True,
            obj_bound_is_valid=obj_bound is not None,
        )

        if was_updated:
            logging.info(
                f"New incumbent by MILP: obj={obj_value} (bound={obj_bound}, status={mdl.solve_details.status})"
            )

    # -----------------
    # Data conversion / decoding
    # -----------------

    def _build_tbb2018_data(self) -> TBB2018Data:
        job_ids = list(self.instance.job_id_list)
        stage_ids = list(self.instance.stage_id_list)

        p: list[list[float]] = []
        for stage_id in stage_ids:
            row = [float(self.stage_job_2_p_dict[(stage_id, job_id)]) for job_id in job_ids]
            p.append(row)

        d = [float(self.instance.job_2_duedate_map[j]) for j in job_ids]
        return TBB2018Data(p=p, d=d)

    def _perm_indices_from_schedule(self, schedule: FlowshopSchedule) -> list[int]:
        """Extract permutation (job indices) from a permutation flowshop schedule."""
        stage_2_seq = schedule.get_stage_2_job_list_map()
        first_stage = self.instance.stage_id_list[0]
        job_seq = stage_2_seq[first_stage]
        job_id_to_idx = {jid: idx for idx, jid in enumerate(self.instance.job_id_list)}
        return [job_id_to_idx[j] for j in job_seq]

    def _dispatch_permutation(self, job_sequence: list[str]) -> FlowshopSchedule:
        """Create a permutation flow shop schedule by serial dispatch."""
        schedule = FlowshopSchedule.from_stage_name_list(self.stage_ids)

        # completion frontier per stage
        stage_2_end: dict[str, int] = {i: 0 for i in self.stage_ids}

        for job_id in job_sequence:
            prev_end = 0
            for stage_id in self.stage_ids:
                p = int(self.job_2_stage_2_p_dict[job_id][stage_id])
                start = max(stage_2_end[stage_id], prev_end)
                end = start + p

                stage = schedule.get_stage_by_name(stage_id)
                op = stage.add_operation(
                    FlowshopOperation(
                        job_name=job_id,
                        stage_name=stage_id,
                        start=start,
                        end=end,
                    )
                )
                if op is None:
                    raise RuntimeError(
                        f"Failed to schedule operation job={job_id}, stage={stage_id}"
                    )

                stage_2_end[stage_id] = end
                prev_end = end

        return schedule


def _compute_simple_horizon(instance: FlowshopDuedateParameters) -> int:
    """Return a safe (loose) horizon if none is provided.

    Uses total processing time sum_{i,j} p_{i,j}.
    """
    pmap = instance.get_job_2_stage_2_builtin_int_p_map()
    total = 0
    for j in instance.job_id_list:
        for i in instance.stage_id_list:
            total += int(pmap[j][i])
    return int(total)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Solve one FlowshopDuedateParameters instance using CPLEX/docplex MILP and print the result."
    )
    parser.add_argument(
        "instance_path",
        type=str,
        help="Path to a VRM benchmark instance file (e.g. Inputs/.../123.txt).",
    )
    parser.add_argument(
        "--timelimit",
        type=float,
        default=30.0,
        help="Time limit in seconds for MILP solve.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="CPLEX threads.",
    )
    parser.add_argument(
        "--mip-gap",
        type=float,
        default=None,
        help="Optional MIP gap (e.g., 0.01).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (used for any randomized subroutines / reproducibility).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Optional horizon (shared_param_dict['horizon']). If omitted, uses sum of processing times.",
    )
    parser.add_argument(
        "--no-mip-start",
        action="store_true",
        help="Disable MIP start from incumbent (if any).",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default=None,
        help="Optional working directory for logs (default: none).",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    instance_path = Path(args.instance_path)
    if not instance_path.exists():
        raise FileNotFoundError(f"Instance file not found: {instance_path}")

    ins_name = instance_path.stem
    with instance_path.open("r", encoding="utf-8") as f:
        instance = FlowshopDuedateParameters.from_vrm_data(ins_name, f)

    horizon = int(args.horizon) if args.horizon is not None else _compute_simple_horizon(instance)
    shared_param_dict = {"horizon": horizon}

    stopping_criteria = StoppingCriteria({"timelimit": float(args.timelimit)})
    ctrl = FlowshopTardinessCplexMatheuristicController(
        instance=instance,
        shared_param_dict=shared_param_dict,
        subroutine_flow=DynamicDataObject.from_obj([]),
        stopping_criteria=stopping_criteria,
    )
    if args.workdir:
        ctrl.set_working_dir(Path(args.workdir))

    if args.seed:
        ctrl.set_random_seed(int(args.seed))

    ctrl.solve_base_milp(
        computational_time=float(args.timelimit),
        solver_thread_cnt=int(args.threads),
        mip_gap=args.mip_gap,
        use_mip_start_from_incumbent=not args.no_mip_start,
        error_if_infeasible=True,
    )

    best = ctrl.solution_manager.get_incumbent()
    if best is None:
        print("No solution found.")
        return 2

    best_obj = ctrl.solution_manager.best_obj_value
    best_bound = ctrl.solution_manager.best_obj_bound
    stage_2_seq = best.get_stage_2_job_list_map()
    first_stage = instance.stage_id_list[0]
    perm = stage_2_seq[first_stage]

    print("==== CPLEX MILP Result ====")
    print(f"instance: {instance.name}")
    print(f"jobs: {instance.job_count}, stages: {instance.stage_count}")
    print(f"timelimit_s: {args.timelimit}, threads: {args.threads}, mip_gap: {args.mip_gap}")
    print(f"horizon: {horizon}")
    print(f"objective (sumTj): {best_obj}")
    print(f"bound: {best_bound}")
    print(f"makespan: {best.makespan}")
    print(f"perm(first stage): {perm}")
    print("==========================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
