from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Sequence

from routix import DynamicDataObject, ElapsedTimer, StoppingCriteria
from schore.parameters_examples.shop.flow import FlowshopDuedateParameters
from schore.schedule_examples.shop.flow import FlowshopOperation, FlowshopSchedule

from ..cplex_model.model import TBB2018Data, TBB2018MilpModelBuilder
from ..report import FsSubroutineReport
from ..report.fs_cplex_subroutine_report import FsCplexSolverReport
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

    # Start solution <-> job permutation

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

    # End solution <-> job permutation

    # Start solver building and solution extraction

    def _solve_milp(
        self,
        job_subsequence: list[str] | None = None,
        stage_eat_dict: dict[str, int] | None = None,
        alpha: float | None = None,
        computational_time: float | None = None,
        solver_thread_cnt: int = 1,
        mip_gap: float | None = None,
        is_init: bool = False,
        incumbent_job_sequence: list[str] | None = None,
    ) -> tuple[FsCplexSolverReport, FlowshopSchedule]:
        """Solve MILP on a job subsequence with optional stage earliest available times.

        Args:
            job_subsequence (list[str]): list of job IDs to include in the subsequence.
                If not provided, all jobs are included.
            stage_eat_dict (dict[str, int]): stage name -> earliest available time
                If not provided, no earliest available time constraints are added.
            computational_time (float): time limit for this call (seconds).
            solver_thread_cnt (int): CPLEX threads.
            mip_gap (float | None): optional MIP gap.
            is_init (bool): whether this is an initialization subroutine.

        Returns:
            tuple[FsCplexSolverReport, FlowshopSchedule]: report and resulting schedule
        """
        sub_timer = ElapsedTimer()

        data = self._build_tbb2018_data(job_ids=job_subsequence)
        stage_eat_list = []
        if stage_eat_dict is not None:
            stage_eat_list = [
                stage_eat_dict.get(stage_id, 0)
                for stage_id in self.instance.stage_id_list
            ]
        else:
            stage_eat_list = None
        builder = TBB2018MilpModelBuilder(
            data, stage_eat_list=stage_eat_list, alpha=alpha, model_name="milp_subseq"
        )
        timelimit: float = self.get_remaining_time_limit(computational_time)

        # job pool for decoding & for building incumbent perm
        job_pool = (
            job_subsequence
            if job_subsequence is not None
            else list(self.instance.job_id_list)
        )

        incumbent_perm: list[int] | None = None
        if incumbent_job_sequence is not None:
            pos = {jid: idx for idx, jid in enumerate(job_pool)}
            # incumbent_job_sequence must be compatible with pool
            if all(jid in pos for jid in incumbent_job_sequence) and len(
                incumbent_job_sequence
            ) == len(job_pool):
                incumbent_perm = [pos[jid] for jid in incumbent_job_sequence]
        elif job_subsequence is not None:
            incumbent_perm = list(
                range(data.n)
            )  # 기존 동작 유지 (subseq는 identity start)

        mdl, variables = builder.build(
            time_limit_s=timelimit,
            threads=solver_thread_cnt,
            mip_gap=mip_gap,
            incumbent_perm=incumbent_perm,
        )

        sol = mdl.solve()
        if sol is None:
            status = "UNKNOWN"
            # Create a no solution report
            report = FsCplexSolverReport(
                elapsed_time=sub_timer.elapsed_sec,
                obj_value=None,
                obj_bound=None,
                is_init=is_init,
                obj_value_records=[],
                obj_bound_records=[],
                status=status,
            )
            # Create an empty schedule
            schedule = FlowshopSchedule.from_stage_name_list(self.stage_ids)
            return report, schedule
        status = str(mdl.solve_details.status)
        # TODO: obj_value_record and obj_bound_record extraction

        # Decode permutation & create schedule
        perm_idx: list[int] = builder.extract_permutation_from_solution(
            sol, variables.x, data.n
        )
        new_job_seq: list[str] = [job_pool[j] for j in perm_idx]
        schedule: FlowshopSchedule = self._dispatch_permutation(new_job_seq)

        # Create a report
        obj_value = self.get_obj_value(schedule)
        obj_bound = getattr(mdl.solve_details, "best_bound", None)
        if obj_bound is not None:
            try:
                obj_bound = float(obj_bound)
            except Exception:
                obj_bound = None

        report = FsCplexSolverReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=obj_bound,
            is_init=is_init,
            obj_value_records=[],
            obj_bound_records=[],
            status=status,
        )
        return report, schedule

    def _build_tbb2018_data(self, job_ids: list[str] | None = None) -> TBB2018Data:
        if job_ids is None:
            job_ids = list(self.instance.job_id_list)
        stage_ids = list(self.instance.stage_id_list)

        p: list[list[int]] = []
        for stage_id in stage_ids:
            job_2_p_dict = self.stage_2_job_2_p_dict[stage_id]
            row: list[int] = [int(job_2_p_dict[job_id]) for job_id in job_ids]
            p.append(row)

        d: list[int] = [int(self.instance.job_2_duedate_map[j]) for j in job_ids]
        return TBB2018Data(p=p, d=d)

    # End solver building and solution extraction

    # Start subalgorithms

    def solve_base_milp(
        self,
        computational_time: float | None = None,
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
        incumbent_schedule = self.solution_manager.get_incumbent()
        is_init = incumbent_schedule is None

        if incumbent_schedule is not None and use_mip_start_from_incumbent:
            incumbent_job_seq = incumbent_schedule.get_last_stage_job_list()
        else:
            incumbent_job_seq = None

        report, schedule = self._solve_milp(
            job_subsequence=None,
            stage_eat_dict=None,
            computational_time=computational_time,
            solver_thread_cnt=solver_thread_cnt,
            mip_gap=mip_gap,
            is_init=is_init,
            incumbent_job_sequence=incumbent_job_seq,
        )

        if report.obj_value is None:
            logging.warning("Base MILP: no solution returned.")
            return

        if error_if_infeasible:
            self.check_feasibility(schedule)

        was_updated = self.solution_manager.register(report, schedule)

        # Objective time series logs (only record if improving)
        log_time = self.timer.elapsed_sec
        self.obj_store.add_obj_value(log_time, report.obj_value, is_maximize=False)
        if report.obj_bound is not None:
            # For minimization, higher lower-bound is better
            self.obj_store.add_obj_bound(
                log_time, float(report.obj_bound), is_maximize=False
            )

        self.obj_store.add_last_timestamp_note(
            self._get_call_context_of_current_method(),
            obj_value_is_valid=True,
            obj_bound_is_valid=report.obj_bound is not None,
        )

        if was_updated:
            logging.info(
                f"New incumbent by MILP: obj={report.obj_value} (bound={report.obj_bound})"
            )

    # Initial solution method

    def initialize_by_edd(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        """Initialize by EDD (Earliest Due Date) dispatching."""
        sub_timer = ElapsedTimer()

        job_due_list = [
            (job_id, self.instance.job_2_duedate_map[job_id])
            for job_id in self.instance.job_id_list
        ]
        job_due_list.sort(key=lambda x: x[1])  # sort by due date

        edd_job_sequence = [job_id for job_id, due in job_due_list]
        schedule = self._dispatch_permutation(edd_job_sequence)

        obj_value = self.get_obj_value(schedule)
        report = FsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
        )
        was_updated = self.solution_manager.register(report, schedule)

        if was_updated:
            log_time = self.timer.elapsed_sec
            self.obj_store.add_obj_value(log_time, obj_value, is_maximize=False)
            _last_timestamp_note = self._get_call_context_of_current_method()
            self.obj_store.add_last_timestamp_note(
                _last_timestamp_note, obj_value_is_valid=True
            )
            logging.info(f"New incumbent by EDD initialization: obj={obj_value}")
            if error_if_infeasible:
                self.check_feasibility(schedule)
            if draw_gantt:
                self.export_incumbent_to_yaml()

    # Start MH_X1 method & helpers

    def mh_x1(
        self,
        window_size: int,
        alpha: float,
        swap_trials: int,
        fi_trials: int,
        bi_trials: int,
        solver_thread_cnt: int = 1,
        error_if_infeasible: bool = False,
    ) -> None:
        """Matheuristic X1 method as in TaEtAl2018.

        Args:
            window_size: size of the job subsequence window.
            alpha: fraction of jobs to fix outside the window.
            swap_trials: number of swap trials per iteration.
            bi_trials: number of backward insertion trials per iteration.
            fi_trials: number of forward insertion trials per iteration.
            error_if_infeasible: if True, raise when decoded schedule is infeasible.
        """
        sub_timer = ElapsedTimer()
        rng = random.Random(
            getattr(self, "random_seed", None)
        )  # fallback; seed는 ctrl.set_random_seed로 통제

        incumbent_schedule = self.solution_manager.get_incumbent()
        if incumbent_schedule is None:
            raise RuntimeError(
                "MHX1 requires an incumbent solution. Run initialize_by_edd() or solve_base_milp() first."
            )

        incumbent_seq: list[str] = incumbent_schedule.get_last_stage_job_list()
        n = len(incumbent_seq)
        H = int(window_size)
        if H <= 0 or H > n:
            raise ValueError(f"Invalid window_size={window_size} for n={n}")

        is_timeover = False
        loop_cnt = 0

        # Paper-style outer loop: while TimeLimMH and improved
        # (논문은 improved 플래그로 반복; 여기서는 time limit 안에서 local improvement가 없으면 종료)
        while not is_timeover:
            logging.info(f"MHX1: starting outer loop iteration {loop_cnt + 1}")
            # 논문은 R=1..n-H (1-index) 반복, 그리고 여러 번 R=1로 돌아가는 구조 언급
            # 여기서는 time limit 내에서 한 번 sweep 후, 개선 있으면 다시 sweep
            R = 0
            while R <= n - H:
                # If R > n - H, end sweep
                if self.get_remaining_time_limit(None) <= 0:
                    is_timeover = True
                    break
                # logging.info(f"MHX1: processing window starting at R={R+1} / {n - H + 1}")

                A = incumbent_seq[:R]  # First R jobs
                X = incumbent_seq[R : R + H]  # Next H jobs (window)
                B = incumbent_seq[R + H :]  # Remaining jobs

                # --------------------
                # 1) Improve A with neighborhoods (no EAT)
                # order: SWAP -> FI -> BI (as you requested)
                # --------------------
                # A0 = list(A)
                A = self._try_improve_swap(A, None, alpha, swap_trials, rng)
                A = self._try_improve_fi(A, None, alpha, fi_trials, rng)
                A = self._try_improve_bi(A, None, alpha, bi_trials, rng)

                # stage EAT after A (CA_i in the paper)
                _, _, stage_eat_A = self._simulate_subsequence(A, None)

                # --------------------
                # 2) Re-optimize X by MILP given stage_eat_A
                # --------------------
                X0 = list(X)
                rep_x, sched_x = self._solve_milp(
                    job_subsequence=X,
                    stage_eat_dict=stage_eat_A,
                    alpha=alpha,
                    computational_time=None,
                    solver_thread_cnt=solver_thread_cnt,
                    mip_gap=None,
                    is_init=False,
                    incumbent_job_sequence=X,  # warm-start with current X order
                )

                if rep_x.obj_value is not None:
                    # decoded order is the MILP result for subsequence
                    X = sched_x.get_last_stage_job_list()
                else:
                    # keep old if no solution
                    X = X0

                # stage EAT after A+X
                _, _, stage_eat_AX = self._simulate_subsequence(X, stage_eat_A)

                # --------------------
                # 3) Improve B with neighborhoods given stage_eat_AX
                # order: SWAP -> FI -> BI
                # --------------------
                # B0 = list(B)
                B = self._try_improve_swap(B, stage_eat_AX, alpha, swap_trials, rng)
                B = self._try_improve_fi(B, stage_eat_AX, alpha, fi_trials, rng)
                B = self._try_improve_bi(B, stage_eat_AX, alpha, bi_trials, rng)

                # --------------------
                # 4) Build full sequence and accept if incumbent improves (true objective)
                # --------------------
                cand_seq = A + X + B
                if len(cand_seq) != n:
                    raise RuntimeError("MHX1 internal error: sequence length mismatch.")

                cand_schedule = self._dispatch_permutation(cand_seq)
                cand_obj = self.get_obj_value(cand_schedule)

                inc_obj = self.solution_manager.best_obj_value
                if inc_obj is None:
                    inc_obj = float("inf")

                improved_this_R = False

                # strictly improve incumbent by true objective (sum tardiness)
                if cand_obj < float(inc_obj):
                    report = FsSubroutineReport(
                        elapsed_time=sub_timer.elapsed_sec,
                        obj_value=cand_obj,
                        obj_bound=None,
                        is_init=False,
                    )
                    was_updated = self.solution_manager.register(report, cand_schedule)

                    if was_updated:
                        improved_this_R = True
                        incumbent_seq = cand_seq
                        n = len(incumbent_seq)  # unchanged but safe

                        # logs
                        log_time = self.timer.elapsed_sec
                        self.obj_store.add_obj_value(
                            log_time, cand_obj, is_maximize=False
                        )
                        self.obj_store.add_last_timestamp_note(
                            self._get_call_context_of_current_method(),
                            obj_value_is_valid=True,
                        )

                        if error_if_infeasible:
                            self.check_feasibility(cand_schedule)

                # Paper-style R update rule
                # If improved, jump ahead by (H-1) when allowed, then always R += 1
                if improved_this_R:
                    if (R + H) <= (n - H):
                        R = R + H
                else:
                    R += 1

            # end sweep over R
            loop_cnt += 1
            remaining = self.get_remaining_time_limit(None)
            is_timeover = remaining <= 0

    # -------------------------
    # MHX1 neighborhood helpers
    # -------------------------

    def _weighted_value(
        self, total_tardiness: int, makespan: int, alpha: float
    ) -> float:
        """
        Linear combination score used in MHX1 neighborhoods.

        score = alpha * (sum tardiness) + (1-alpha) * (makespan)
        """
        return alpha * total_tardiness + (1.0 - alpha) * makespan

    def _simulate_subsequence(
        self,
        job_sequence: list[str],
        stage_eat_dict: dict[str, int] | None,
    ) -> tuple[int, int, dict[str, int]]:
        """
        Simulate dispatch of a (sub)sequence starting from stage earliest-available-times.

        Returns:
            sum_Tj, makespan, stage_end_dict
        """
        stage_end_dict: dict[str, int] = {sid: 0 for sid in self.stage_ids}
        if stage_eat_dict is not None:
            for sid, t in stage_eat_dict.items():
                if sid in stage_end_dict:
                    stage_end_dict[sid] = int(t)

        sum_Tj = 0
        last_stage = self.stage_ids[-1]

        for job_id in job_sequence:
            prev_end = 0
            for stage_id in self.stage_ids:
                p = int(self.job_2_stage_2_p_dict[job_id][stage_id])
                start = max(stage_end_dict[stage_id], prev_end)
                end = start + p
                stage_end_dict[stage_id] = end
                prev_end = end

            due = int(self.instance.job_2_duedate_map[job_id])
            sum_Tj += max(0, prev_end - due)

        makespan = int(stage_end_dict[last_stage])
        return sum_Tj, makespan, stage_end_dict

    def _eval_subsequence_score(
        self,
        job_sequence: list[str],
        stage_eat_dict: dict[str, int] | None,
        alpha: float,
    ) -> float:
        sum_t, cmax, _ = self._simulate_subsequence(job_sequence, stage_eat_dict)
        return self._weighted_value(sum_t, cmax, alpha)

    def _try_improve_swap(
        self,
        job_sequence: list[str],
        stage_eat_dict: dict[str, int] | None,
        alpha: float,
        swap_trials: int,
        rng: random.Random,
    ) -> list[str]:
        if swap_trials <= 0 or len(job_sequence) <= 1:
            return job_sequence

        best_seq = list(job_sequence)
        best_val = self._eval_subsequence_score(best_seq, stage_eat_dict, alpha)

        for _ in range(int(swap_trials)):
            cand = list(best_seq)
            i, j = rng.sample(range(len(cand)), 2)
            cand[i], cand[j] = cand[j], cand[i]

            val = self._eval_subsequence_score(cand, stage_eat_dict, alpha)
            if val < best_val:  # strictly improved
                best_seq, best_val = cand, val

        return best_seq

    def _try_improve_fi(
        self,
        job_sequence: list[str],
        stage_eat_dict: dict[str, int] | None,
        alpha: float,
        fi_trials: int,
        rng: random.Random,
    ) -> list[str]:
        """Forward insertion: pick i<j, move job at i to position after j."""
        if fi_trials <= 0 or len(job_sequence) <= 1:
            return job_sequence

        best_seq = list(job_sequence)
        best_val = self._eval_subsequence_score(best_seq, stage_eat_dict, alpha)

        n = len(best_seq)
        for _ in range(int(fi_trials)):
            cand = list(best_seq)
            i = rng.randrange(n - 1)
            j = rng.randrange(i + 1, n)
            job = cand.pop(i)
            cand.insert(j, job)

            val = self._eval_subsequence_score(cand, stage_eat_dict, alpha)
            if val < best_val:
                best_seq, best_val = cand, val

        return best_seq

    def _try_improve_bi(
        self,
        job_sequence: list[str],
        stage_eat_dict: dict[str, int] | None,
        alpha: float,
        bi_trials: int,
        rng: random.Random,
    ) -> list[str]:
        """Backward insertion: pick j<i, move job at i to position before j."""
        if bi_trials <= 0 or len(job_sequence) <= 1:
            return job_sequence

        best_seq = list(job_sequence)
        best_val = self._eval_subsequence_score(best_seq, stage_eat_dict, alpha)

        n = len(best_seq)
        for _ in range(int(bi_trials)):
            cand = list(best_seq)
            i = rng.randrange(1, n)
            j = rng.randrange(0, i)
            job = cand.pop(i)
            cand.insert(j, job)

            val = self._eval_subsequence_score(cand, stage_eat_dict, alpha)
            if val < best_val:
                best_seq, best_val = cand, val

        return best_seq

    # End MH_X1 method & helpers


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

    horizon = (
        int(args.horizon)
        if args.horizon is not None
        else _compute_simple_horizon(instance)
    )
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
    print(
        f"timelimit_s: {args.timelimit}, threads: {args.threads}, mip_gap: {args.mip_gap}"
    )
    print(f"horizon: {horizon}")
    print(f"objective (sumTj): {best_obj}")
    print(f"bound: {best_bound}")
    print(f"makespan: {best.makespan}")
    print(f"perm(first stage): {perm}")
    print("==========================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
