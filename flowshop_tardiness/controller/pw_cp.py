import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from mbls.cpsat import CpsatSolverReport, CpsatStatus, ObjValueBoundStore
from ortools.sat.python.cp_model import CpModel, CpSolver
from routix import ElapsedTimer
from routix.util.comparison import float_a_leq_b
from schore.parameters_examples.shop.flow import FlowshopDuedateParameters
from schore.schedule_examples.shop.flow import FlowshopSchedule

from flowshop_tardiness.cpsat_model_2.indirect_prec import IndirectPrecVars
from flowshop_tardiness.cpsat_model_2.params import Params
from flowshop_tardiness.fm_prmu import PermutationFlowshopScheduleLite


class PwCpContext(Protocol):
    """
    Minimal dependency interface.
    (FlowshopTardinessControllerCore is effectively designed to satisfy this interface.)
    """

    # data
    instance: FlowshopDuedateParameters
    stage_ids: tuple[str, ...]
    job_2_stage_2_p_dict: dict[str, dict[str, int]]
    params: Params  # global/full params
    solver: CpSolver

    # solution manager access
    @property
    def solution_manager(self): ...

    # time & logging
    @property
    def timer(self): ...

    def get_remaining_time_limit(
        self, subroutine_time_limit: float | None
    ) -> float: ...
    def get_obj_value(self, schedule: FlowshopSchedule) -> int: ...
    def check_feasibility(self, schedule: FlowshopSchedule) -> float: ...

    # CP solve & decoding
    def solve_cp_model_2(
        self,
        mdl: CpModel,
        computational_time: float,
        solver_thread_cnt: int,
        e_timer: ElapsedTimer | None = None,
        obj_value_is_valid: bool = False,
        obj_bound_is_valid: bool = False,
        log_level_obj_value: int = logging.INFO,
        log_level_obj_bound: int = logging.INFO,
    ) -> CpsatSolverReport: ...

    def from_job_prec_get_sequence(
        self, params: Params, prec: dict[tuple[int, int], bool]
    ) -> list[int]: ...

    # optional
    def set_sumTj_lower_bound(
        self, mdl, vars: IndirectPrecVars, bound: float | None
    ) -> None: ...
    def export_solution_to_yaml(
        self,
        start_time_map: dict[tuple[str, str], int],
        end_time_map: dict[tuple[str, str], int],
        output_path: Path | None = None,
        encoding="utf-8",
    ) -> None: ...
    def get_file_path_for_subroutine(self, suffix: str): ...


@dataclass
class PwCpRunState:
    timer: ElapsedTimer

    # --- Job partitions ---
    time_fixed_pool: set[str]
    """Jobs already time-fixed"""

    profile_fixed_jobs: list[str]
    """Jobs with fixed relative order (order can be defined by this list)"""

    remaining_jobs: list[str]
    """Remaining jobs (candidates for addition). = All - profile-fixed - time-fixed"""

    # --- Append-only job sequence ---
    committed_time_fixed_jobs: list[str]
    """Sequence of time-fixed jobs; once a job is committed here, it won't be changed."""

    # --- Schedule instances ---
    time_fixed_sol: PermutationFlowshopScheduleLite
    profile_fixed_sol: PermutationFlowshopScheduleLite  # For debugging / logging

    # --- iteration bookkeeping ---
    iter_idx: int
    sub_obj_store: ObjValueBoundStore[int]

    # --- CP solving results ---
    last_cp_subseq: list[str] | None
    """Sequence of (profile-fixed + added batch) jobs (global id)"""

    last_cp_obj: int | None
    """Objective value from last CP solving."""

    last_improved: bool | None
    """Whether the last CP solving improved the objective value."""

    # --- Algorithm parameter cache (referenced repeatedly in loop) ---
    added_batch_size: int

    @property
    def added_job_list(self) -> list[str]:
        """
        Returns:
            list[str]: List of jobs to be considered for addition in this iteration.
        """
        return self.remaining_jobs[: self.added_batch_size]

    @property
    def not_added_first_job(self) -> str | None:
        """
        Returns:
            str | None: The first remaining job not considered for addition in this iteration
                (None if not exists).
        """
        if len(self.remaining_jobs) > self.added_batch_size:
            return self.remaining_jobs[self.added_batch_size]
        else:
            return None

    @property
    def iter_cp_job_list(self) -> list[str]:
        """
        Returns:
            list[str]: List of jobs included in CP for this iteration (profile-fixed + added).
        """
        return self.profile_fixed_jobs + self.added_job_list

    @property
    def last_job_is_included(self) -> bool:
        """
        Returns:
            bool: Whether the last job is included in CP for this iteration.
        """
        return len(self.remaining_jobs) <= self.added_batch_size

    def extend_time_fixed_jobs(self, job_list: list[str]) -> None:
        """Add job_list to time-fixed job list and update time-fixed pool.

        Args:
            job_list (list[str]): List of time-fixed jobs to add.
        """
        self.committed_time_fixed_jobs.extend(job_list)
        self.time_fixed_pool.update(job_list)
        self.time_fixed_sol.extend_jobs(job_list)


@dataclass
class PwCpResult:
    schedule: FlowshopSchedule
    sub_obj_store: ObjValueBoundStore[int]
    last_obj_value: int


class PwCpConstructor:
    # Given solution cache
    job_sequence: list[str]
    job_cnt: int

    # Algorithm parameter cache
    profile_fixed_cnt: int
    step_size_on_improve: int
    step_size_on_no_improve: int

    def __init__(self, ctx: PwCpContext):
        from flowshop_tardiness.cpsat_model_2.indirect_prec import BaseModelBuilder

        self.ctx = ctx
        self.builder = BaseModelBuilder()
        self._st: PwCpRunState | None = None

    def _require_state(self) -> PwCpRunState:
        if self._st is None:
            raise RuntimeError("PwCpConstructor.run() is not active; state is missing.")
        return self._st

    def save_schedule_lite_to_yaml(
        self, schedule: PermutationFlowshopScheduleLite, output_path: Path
    ) -> None:
        """Save a PermutationFlowshopScheduleLite to a YAML file via the context method.

        Args:
            schedule (PermutationFlowshopScheduleLite): The schedule to save.
            output_path (Path): The output file path.
        """
        ctx = self.ctx
        start_time_map = schedule.get_start_time_map()
        end_time_map = schedule.get_end_time_map()

        ctx.export_solution_to_yaml(
            start_time_map=start_time_map,
            end_time_map=end_time_map,
            output_path=output_path,
        )

    def run(
        self,
        job_sequence: list[str],
        added_batch_size: int | None = None,
        profile_fixed_cnt: int | None = None,
        step_size_on_improve: int | None = None,
        step_size_on_no_improve: int | None = None,
        max_time_per_add: float | None = None,
        solver_thread_cnt: int | None = None,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> PwCpResult:
        """
        Run the Prefix-Window CP (PW-CP) algorithm.

        This algorithm incrementally builds a schedule by iteratively solving a CP model
        for a window of jobs. The window slides forward as jobs are fixed.

        Args:
            job_sequence (list[str]): The initial full sequence of jobs.
            added_batch_size (int | None, optional): The number of jobs to add to the window in each iteration.
                If None, defaults to 1.
            profile_fixed_cnt (int | None, optional): The number of jobs at the beginning of the window
                whose relative order is fixed (profile-fixed). These jobs are part of the CP problem
                but their relative positions are constrained. If None, defaults to 0.
            step_size_on_improve (int | None, optional): The number of jobs to finalize (move from window
                to profile-fixed or time-fixed) when the CP solution improves the objective.
                Defaults to `added_batch_size`.
            step_size_on_no_improve (int | None, optional): The number of jobs to finalize when the CP
                solution does not improve the objective. Defaults to `added_batch_size`.
            max_time_per_add (float | None, optional): Time limit for each CP solving iteration.
                Defaults to None.
            solver_thread_cnt (int | None, optional): Number of threads for the CP solver. Defaults to None.
            error_if_infeasible (bool, optional): Whether to raise an error if the final schedule is infeasible.
                Defaults to False.
            draw_gantt (bool, optional): Whether to save Gantt charts for intermediate and final solutions.
                Defaults to False.

        Returns:
            PwCpResult: The result containing the final schedule, objective value, and log store.
        """
        ctx = self.ctx
        timer = ElapsedTimer()
        sub_obj_store = ObjValueBoundStore[int]()
        sub_obj_store.obj_value_series.name = "ObjVal after dispatch"
        sub_obj_store.obj_bound_series.name = "ObjVal before dispatch"

        self.job_sequence = job_sequence
        self.job_cnt = len(job_sequence)
        _added_batch_size = (
            added_batch_size
            if added_batch_size is not None and added_batch_size >= 1
            else 1
        )
        self.profile_fixed_cnt = (
            profile_fixed_cnt
            if profile_fixed_cnt is not None and profile_fixed_cnt >= 0
            else 0
        )
        self.step_size_on_improve = (
            step_size_on_improve
            if step_size_on_improve is not None and step_size_on_improve > 0
            else _added_batch_size
        )
        self.step_size_on_no_improve = (
            step_size_on_no_improve
            if step_size_on_no_improve is not None and step_size_on_no_improve > 0
            else _added_batch_size
        )

        given_sol = PermutationFlowshopScheduleLite(
            ctx.stage_ids,
            job_2_stage_2_p_map=ctx.job_2_stage_2_p_dict,
            job_2_due_map=ctx.instance.job_2_duedate_map,
        )
        given_sol.extend_jobs(self.job_sequence)
        given_sol.push_back_tail_jobs_keep_tardiness(self.job_cnt)
        if draw_gantt:
            given_sol_output_path = ctx.get_file_path_for_subroutine(
                "_0_pushed_back_solution.yaml"
            )
            self.save_schedule_lite_to_yaml(given_sol, given_sol_output_path)
            logging.info("Saved pushed-back solution to: %s", given_sol_output_path)

        # Initialize state
        time_fixed_sol = PermutationFlowshopScheduleLite(
            ctx.stage_ids,
            job_2_stage_2_p_map=ctx.job_2_stage_2_p_dict,
            job_2_due_map=ctx.instance.job_2_duedate_map,
        )
        profile_fixed_sol = PermutationFlowshopScheduleLite(
            ctx.stage_ids,
            job_2_stage_2_p_map=ctx.job_2_stage_2_p_dict,
            job_2_due_map=ctx.instance.job_2_duedate_map,
        )
        self._st = PwCpRunState(
            timer=timer,
            remaining_jobs=self.job_sequence.copy(),
            profile_fixed_jobs=[],
            time_fixed_pool=set(),
            committed_time_fixed_jobs=[],
            time_fixed_sol=time_fixed_sol,
            profile_fixed_sol=profile_fixed_sol,
            iter_idx=0,
            sub_obj_store=sub_obj_store,
            last_cp_subseq=None,
            last_cp_obj=None,
            last_improved=None,
            added_batch_size=_added_batch_size,
        )

        try:
            return self._run_loop(
                given_sol,
                solver_thread_cnt=solver_thread_cnt,
                max_time_per_add=max_time_per_add,
                error_if_infeasible=error_if_infeasible,
                draw_gantt=draw_gantt,
            )
        finally:
            self._st = None

    def _make_all_dispatched(
        self,
        base_sol: PermutationFlowshopScheduleLite,
        already_scheduled_job_set: set[str],
    ) -> PermutationFlowshopScheduleLite:
        """Create a new PermutationFlowshopScheduleLite by dispatching remaining jobs.

        Args:
            base_sol (PermutationFlowshopScheduleLite): The base solution to modify.
            already_scheduled_job_set (set[str]): The set of already scheduled jobs.

        Returns:
            PermutationFlowshopScheduleLite: The modified flow shop schedule.
        """
        _base_sol = base_sol.deepcopy()

        if len(already_scheduled_job_set) == self.job_cnt:
            return _base_sol
        remaining_jobs = [
            j for j in self.job_sequence if j not in already_scheduled_job_set
        ]
        full_sched = _base_sol
        full_sched.extend_jobs(remaining_jobs)
        return full_sched

    def _log_snapshot(
        self,
        picked_obj_val: int,
        note: str,
        timestamp: float,
        bound_val: int | None = None,
    ) -> None:
        """Record one iteration snapshot into the sub objective store.

        Args:
            picked_obj_val (int): Total tardiness of the fully-dispatched schedule
                (time-fixed prefix + remaining jobs). Stored in the
                "ObjVal after dispatch" series.
            note (str): A note to attach to the log entry (number of fixed jobs).
            timestamp (float): The timestamp for the log entry.
            bound_val (int | None, optional): Total tardiness of the time-fixed
                prefix only, i.e. the objective *before* dispatching the remaining
                tail. Stored in the "ObjVal before dispatch" series so the tail
                contribution (= after - before), which the CP subproblem never
                optimizes, is reviewable afterwards. Defaults to None.
        """
        sub_obj_store = self._require_state().sub_obj_store
        sub_obj_store.add_obj_value(timestamp, picked_obj_val, is_maximize=None)
        if bound_val is not None:
            sub_obj_store.add_obj_bound(timestamp, bound_val, is_maximize=None)
        sub_obj_store.add_last_timestamp_note(
            note,
            obj_value_is_valid=True,
            obj_bound_is_valid=bound_val is not None,
        )

    def _solve_cp_model_lexico_for_batch(
        self,
        sub_instance: FlowshopDuedateParameters,
        solver_timelimit: float,
        solver_thread_cnt: int | None,
        last_job_is_included: bool,
        stage_2_est_map: dict[str, int] | None = None,
        stage_2_lct_map: dict[str, int] | None = None,
        sumTj_offset: int | None = None,
        profile_fixed_job_list: list[str] | None = None,
    ) -> tuple[CpsatSolverReport, list[str]]:
        """
        Solve CP model in two phases to optimize lexicographic objectives:

        - Phase 1: Minimize total tardiness
          - If total tardiness is zero with initial sequence, skip CP solving.
          - If no solution is found, return the solver report and an empty job sequence.
        - Phase 2: Minimize sum of latest completion times, subject to optimal total tardiness
          - If no solution is found, return the phase 1 solver report and job sequence.
          - If all jobs are included, skip phase 2.

        Args:
            sub_instance (FlowshopDuedateParameters): Sub-instance to solve CP on.
            stage_2_est_map (dict[str, int] | None): map of earliest start times for stage 2.
            stage_2_lct_map (dict[str, int] | None): map of latest completion times for stage 2.
            sumTj_offset (int | None): offset for total tardiness.
            solver_timelimit (float): time limit for solving each phase.
            solver_thread_cnt (int | None): number of solver threads.
            all_jobs_are_included (bool): whether all jobs are included in the sub-instance.

        Raises:
            RuntimeError: If total_tardiness or sum_latest_completion variable is None after CP building.
            RuntimeError: If obj_value is None despite feasibility.

        Returns:
            tuple[CpsatSolverReport, list[str]]: Solver report and job sequence.
        """
        sub_timer = ElapsedTimer()
        if solver_thread_cnt is None:
            solver_thread_cnt = 1
        st = self._require_state()
        ctx = self.ctx

        subjob_id_list: list[str] = sub_instance.job_id_list
        # init_pi_hint: list[int] = list(range(len(subjob_id_list)))
        prec_hint: dict[tuple[int, int], int] = {}
        for j1_idx in range(len(subjob_id_list)):
            for j2_idx in range(j1_idx + 1, len(subjob_id_list)):
                prec_hint[(j1_idx, j2_idx)] = 1
                prec_hint[(j2_idx, j1_idx)] = 0

        # Phase 1: Minimize total tardiness

        # Simulate before building CP model
        init_pi_sched = PermutationFlowshopScheduleLite(
            self.ctx.stage_ids,
            job_2_stage_2_p_map=self.ctx.job_2_stage_2_p_dict,
            job_2_due_map=self.ctx.instance.job_2_duedate_map,
        )
        init_pi_sched.extend_jobs(subjob_id_list, stage_2_est_map=stage_2_est_map)
        init_pi_total_tardiness = init_pi_sched.get_total_tardiness()

        if init_pi_total_tardiness > 0:
            # If initial total tardiness is nonzero, build & solve phase 1 CP
            logging.info(
                "Initial total tardiness with given subsequence: %d",
                init_pi_total_tardiness,
            )
            mdl1, params1, vars1 = self.builder.build(
                sub_instance,
                stage_2_est_map=stage_2_est_map,
                stage_2_lct_map=stage_2_lct_map,
                sumTj_offset=sumTj_offset,
                profile_fixed_job_list=profile_fixed_job_list,
            )
            if vars1.total_tardiness is None:
                raise RuntimeError(
                    "Unexpected: total_tardiness variable is None after CP building."
                )
            # Define primary objective
            mdl1.minimize(vars1.total_tardiness)

            if (
                last_job_is_included
                and ctx.solution_manager.best_obj_bound is not None
                and not math.isnan(ctx.solution_manager.best_obj_bound)
            ):
                ctx.set_sumTj_lower_bound(
                    mdl1, vars1, bound=ctx.solution_manager.best_obj_bound
                )

            mdl1.clear_hints()
            # for idx, k in enumerate(params1.j_list):
            #     mdl1.add_hint(vars1.pi[k], init_pi_hint[idx])
            for j1_idx, j1 in enumerate(params1.j_list):
                for j2 in params1.j_list[j1_idx + 1 :]:
                    mdl1.add_hint(vars1.prec[j1, j2], 1)
                    mdl1.add_hint(vars1.prec[j2, j1], 0)

            _timelimit = ctx.get_remaining_time_limit(solver_timelimit)
            report1 = ctx.solve_cp_model_2(
                mdl1,
                _timelimit,
                solver_thread_cnt,
                e_timer=st.timer,
                obj_value_is_valid=False,
                obj_bound_is_valid=False,
                log_level_obj_value=logging.NOTSET,
                log_level_obj_bound=logging.NOTSET,
            )
            if not getattr(report1, "is_feasible", False):
                logging.info("No solution from phase 1 CP; skip phase 2.")
                return report1, subjob_id_list

            best_sumTj = int(ctx.solver.Value(vars1.total_tardiness))
            # phase1_pi: list[int] = [
            #     int(ctx.solver.Value(vars1.pi[k])) for k in params1.j_list
            # ]
            # job_seq: list[str] = [subjob_id_list[idx] for idx in phase1_pi]
            phase1_prec: dict[tuple[int, int], int] = {}
            for j1_idx, j1 in enumerate(params1.j_list):
                for j2 in params1.j_list[j1_idx + 1 :]:
                    prec_val = int(ctx.solver.Value(vars1.prec[j1, j2]))
                    phase1_prec[(j1, j2)] = prec_val
                    phase1_prec[(j2, j1)] = 1 - prec_val
            job_seq: list[str] = [
                subjob_id_list[idx]
                for idx in ctx.from_job_prec_get_sequence(params1, phase1_prec)
            ]
            logging.info(
                "Phase 1 complete(%s): best total tardiness = %d",
                report1.status,
                best_sumTj,
            )
        else:
            logging.info(
                "Skip CP solving in phase 1: total tardiness of given subsequence is zero."
            )
            report_obj_val = sumTj_offset if sumTj_offset is not None else 0
            elapsed_time = sub_timer.elapsed_sec
            report1 = CpsatSolverReport(
                elapsed_time=elapsed_time,
                obj_value=report_obj_val,
                obj_bound=None,
                status=CpsatStatus.OPTIMAL,
                obj_value_records=[(elapsed_time, report_obj_val)],
                obj_bound_records=[],
            )
            best_sumTj = report_obj_val
            # phase1_pi = init_pi_hint
            phase1_prec = prec_hint
            job_seq = subjob_id_list

        if last_job_is_included:
            logging.info("All jobs are included; skip phase 2.")
            return report1, job_seq

        mdl2, params2, vars2 = self.builder.build(
            sub_instance,
            stage_2_est_map=stage_2_est_map,
            stage_2_lct_map=stage_2_lct_map,
            sumTj_offset=sumTj_offset,
            profile_fixed_job_list=profile_fixed_job_list,
        )
        if vars2.sum_latest_completion is None:
            raise RuntimeError(
                "Unexpected: sum_latest_completion variable is None after CP building."
            )
        # Freeze primary objective value
        mdl2.add(vars2.total_tardiness == best_sumTj)
        # Define secondary objective
        mdl2.minimize(vars2.sum_latest_completion)

        # Add hints from phase1
        # for idx, k in enumerate(params2.j_list):
        #     mdl2.add_hint(vars2.pi[k], phase1_pi[idx])
        for j1_idx, j1 in enumerate(params2.j_list):
            for j2 in params2.j_list[j1_idx + 1 :]:
                mdl2.add_hint(vars2.prec[j1, j2], phase1_prec[(j1, j2)])
                mdl2.add_hint(vars2.prec[j2, j1], phase1_prec[(j2, j1)])

        _timelimit = ctx.get_remaining_time_limit(solver_timelimit)
        # Solve without logging objective values / bounds
        report2 = ctx.solve_cp_model_2(
            mdl2,
            _timelimit,
            solver_thread_cnt,
            e_timer=st.timer,
            obj_value_is_valid=False,
            obj_bound_is_valid=False,
            log_level_obj_value=logging.NOTSET,
            log_level_obj_bound=logging.NOTSET,
        )
        if not getattr(report2, "is_feasible", False):
            logging.info("Sub-CP infeasible in phase 2; use phase 1 solution.")
            return report1, job_seq

        # phase2_pi: list[int] = [
        #     int(ctx.solver.Value(vars2.pi[k])) for k in params2.j_list
        # ]
        # job_seq2: list[str] = [subjob_id_list[idx] for idx in phase2_pi]
        phase2_prec: dict[tuple[int, int], int] = {}
        for j1_idx, j1 in enumerate(params2.j_list):
            for j2 in params2.j_list[j1_idx + 1 :]:
                prec_val = int(ctx.solver.Value(vars2.prec[j1, j2]))
                phase2_prec[(j1, j2)] = prec_val
                phase2_prec[(j2, j1)] = 1 - prec_val
        job_seq2: list[str] = []
        unscheduled_set = set(subjob_id_list)
        while len(unscheduled_set) > 0:
            for j in params2.j_list:
                if subjob_id_list[j] not in unscheduled_set:
                    continue
                # check if all predecessors are scheduled
                all_preds_scheduled = True
                for k in params2.j_list:
                    if k == j:
                        continue
                    if (
                        phase2_prec[(k, j)] == 1
                        and subjob_id_list[k] in unscheduled_set
                    ):
                        all_preds_scheduled = False
                        break
                if all_preds_scheduled:
                    job_seq2.append(subjob_id_list[j])
                    unscheduled_set.remove(subjob_id_list[j])
                    break
        # use report 1 which has total tardiness info
        return report1, job_seq2

    def _run_loop(
        self,
        given_sol: PermutationFlowshopScheduleLite,
        solver_thread_cnt: int | None = None,
        max_time_per_add: float | None = None,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> PwCpResult:
        """
        Main loop for the PW-CP algorithm.

        Iteratively adds jobs, defines the profile-fixed and optimization windows,
        runs the CP solver, and updates the schedule based on the results.

        Args:
            given_sol (PermutationFlowshopScheduleLite): The initial schedule used as a reference
                for LCT estimation (push-back mechanism).
            solver_thread_cnt (int | None, optional): Number of threads for the CP solver.
                Defaults to None (1 thread).
            max_time_per_add (float | None, optional): Time limit for each CP solving iteration.
                Defaults to None.
            error_if_infeasible (bool, optional): Whether to raise an error if infeasibility occurs.
                Defaults to False.
            draw_gantt (bool, optional): Whether to save Gantt charts. Defaults to False.

        Returns:
            PwCpResult: The final result of the PW-CP run.
        """
        st = self._require_state()
        ctx = self.ctx

        if solver_thread_cnt is None:
            solver_thread_cnt = 1

        # Initial step size = added batch size
        step_size: int = st.added_batch_size
        # Last step size used
        last_step_size: int = st.added_batch_size

        # --- Monotonicity diagnostics ---
        # Each committed batch is kept within its per-stage window [EST, LCT]
        # derived from the right-justified reference schedule S^R, which keeps the
        # realized objective <= the incumbent pw_cp started from. That does NOT make
        # the internal trajectory monotone: S^R's slack (vs already-improved states)
        # can be spent by the phase-2 secondary objective, raising the last-stage
        # makespan and delaying the tail. Track the previous fully-dispatched
        # objective so any such cross-iteration increase is flagged and explained.
        prev_full_obj_val: int | None = None

        # Repeat until no more jobs to optimize
        while len(st.remaining_jobs) > (step_size - last_step_size):
            st.iter_idx += 1
            _timelimit = ctx.get_remaining_time_limit(max_time_per_add)
            if float_a_leq_b(_timelimit, 0):
                logging.info(
                    "(batch %d) Time over before CP -> dispatch remaining %d jobs.",
                    st.iter_idx + 1,
                    len(st.remaining_jobs),
                )
                break

            added_job_list: list[str] = st.added_job_list
            if not added_job_list:
                logging.info(
                    "(batch %d) No more jobs to add -> finish.",
                    st.iter_idx + 1,
                )
                break

            sub_jobs: list[str] = (
                st.iter_cp_job_list
            )  # profile_fixed + added(front batch)
            last_job_idx = self.job_sequence.index(sub_jobs[-1])
            base_seq: list[str] = list(sub_jobs)  # "before optimization" reference
            prev_pf_len: int = len(st.profile_fixed_jobs)

            # Bounds from previous iteration
            sumTj_offset: int
            stage_2_est_map: dict[str, int]
            stage_2_lct_map: dict[str, int]

            if isinstance(st.time_fixed_sol, PermutationFlowshopScheduleLite):
                sumTj_offset = st.time_fixed_sol.get_total_tardiness()
                stage_2_est_map = st.time_fixed_sol.get_stage_2_makespan_map()
            else:
                sumTj_offset = 0
                stage_2_est_map = {}

            if not st.last_job_is_included:
                stage_2_lct_map = given_sol.get_stage_2_start_time_map(
                    st.not_added_first_job
                )
            else:
                stage_2_lct_map = {}

            logging.info(
                "(iter %d) CP on sub_jobs=%d (pf=%d + added=%d) with %d-th last job, timelimit=%.2fs",
                st.iter_idx,
                len(sub_jobs),
                len(st.profile_fixed_jobs),
                len(added_job_list),
                last_job_idx + 1,
                _timelimit,
            )

            # ---- build subinstance + solve ----
            sub_instance = ctx.instance.get_subinstance(sub_jobs)

            iter_report, solver_seq = self._solve_cp_model_lexico_for_batch(
                sub_instance,
                _timelimit,
                solver_thread_cnt,
                last_job_is_included=st.last_job_is_included,
                stage_2_est_map=stage_2_est_map,
                stage_2_lct_map=stage_2_lct_map,
                sumTj_offset=sumTj_offset,
                profile_fixed_job_list=st.profile_fixed_jobs,
            )
            last_timestamp = st.timer.elapsed_sec

            feasible = getattr(iter_report, "is_feasible", False)
            if not feasible:
                logging.info(
                    "(iter %d) Sub-CP infeasible -> keep base order.", st.iter_idx
                )
                solver_seq = base_seq  # no change

            # Update state
            improved = solver_seq != base_seq
            st.last_improved = improved
            st.last_cp_subseq = solver_seq
            st.last_cp_obj = getattr(iter_report, "obj_value", None)

            # ---- update last step size & (next) step size ----
            last_step_size = step_size
            if improved:
                step_size = (
                    self.step_size_on_improve
                    if self.step_size_on_improve is not None
                    else st.added_batch_size
                )
            else:
                step_size = (
                    self.step_size_on_no_improve
                    if self.step_size_on_no_improve is not None
                    else st.added_batch_size
                )

            # clamp step_size to remaining length
            if step_size > len(st.remaining_jobs):
                step_size = len(st.remaining_jobs)

            logging.info(
                "(iter %d) improve=%s -> step_size=%d (rule=%s, batch=%d)",
                st.iter_idx,
                improved,
                step_size,
                "improve" if improved else "no_improve",
                len(added_job_list),
            )

            # ---- update partitions ----
            # 1) profile_fixed replace:
            #    pf := opt_seq[: (prev_pf_len + step_size)]
            new_pf_len: int = prev_pf_len + step_size
            # opt_seq length is prev_pf_len + len(added_job_list); new_pf_len should not exceed it
            # new_pf_len = min(new_pf_len, len(opt_seq))
            st.profile_fixed_jobs = solver_seq[:new_pf_len]

            # Update profile_fixed_sol for logging
            st.profile_fixed_sol = PermutationFlowshopScheduleLite(
                ctx.stage_ids,
                job_2_stage_2_p_map=ctx.job_2_stage_2_p_dict,
                job_2_due_map=ctx.instance.job_2_duedate_map,
            )
            st.profile_fixed_sol.extend_jobs(
                st.profile_fixed_jobs, stage_2_est_map=stage_2_est_map
            )

            # 2) overflow -> commit into time-fixed
            overflow_cnt: int = len(st.profile_fixed_jobs) - self.profile_fixed_cnt
            newly_committed_jobs: list[str] = []
            if overflow_cnt > 0:
                newly_committed_jobs = st.profile_fixed_jobs[:overflow_cnt]
                st.profile_fixed_jobs = st.profile_fixed_jobs[overflow_cnt:]
                st.extend_time_fixed_jobs(newly_committed_jobs)

            # 3) remaining_jobs update
            st.remaining_jobs = [
                j for j in st.remaining_jobs if j not in st.time_fixed_pool
            ]
            if self.profile_fixed_cnt > 0:
                st.remaining_jobs = [
                    j for j in st.remaining_jobs if j not in st.profile_fixed_jobs
                ]

            # --- Log ---
            # Realized objective = total tardiness of the fully-dispatched schedule:
            #   time-fixed prefix (CP-decided order) + remaining jobs (incumbent order).
            full_sol = self._make_all_dispatched(st.time_fixed_sol, st.time_fixed_pool)
            full_obj_val = full_sol.get_total_tardiness()
            note = str(len(st.time_fixed_pool) + len(st.profile_fixed_jobs))

            # --- Monotonicity diagnostics ---
            # pw_cp is non-increasing *relative to the incumbent it started from*:
            # each committed batch is bounded, on every stage, by its LCT = the next
            # job's start in the right-justified reference schedule S^R, which keeps
            # the greedily dispatched tail within S^R. That bound is NECESSARY but
            # not sufficient for monotonicity *across iterations*: S^R is computed
            # once from the original incumbent, so it carries slack relative to
            # already-improved intermediate states. The phase-2 secondary objective
            # (minimize the sum of per-stage makespans) can spend that slack by
            # raising the LAST-stage makespan to lower earlier stages, delaying the
            # tail and raising total tardiness vs the previous iteration (while still
            # staying <= the original incumbent). So an increase here is a property
            # of the current design, not a window-bound violation -- but verify the
            # bound really holds and record what the reorder spent.
            committed_obj = st.time_fixed_sol.get_total_tardiness()
            tail_obj = full_obj_val - committed_obj  # not seen by the CP objective
            cp_obj = getattr(iter_report, "obj_value", None)

            # Sanity check: makespan of the last committed job must be <= LCT on
            # every stage. A violation here would be a genuine bug.
            makespan_after = st.time_fixed_sol.get_stage_2_makespan_map()
            window_violations: list[tuple[str, int, int]] = [
                (stage_name, makespan_after[stage_name], lct_i)
                for stage_name, lct_i in stage_2_lct_map.items()
                if stage_name in makespan_after and makespan_after[stage_name] > lct_i
            ]

            logging.info(
                "(iter %d) full_obj=%d [committed=%d + tail=%d] "
                "cp(committed+window)=%s newly_committed=%d window_violations=%d",
                st.iter_idx,
                full_obj_val,
                committed_obj,
                tail_obj,
                cp_obj,
                len(newly_committed_jobs),
                len(window_violations),
            )

            if window_violations:
                logging.warning(
                    "(iter %d) WINDOW BOUND VIOLATED on %d stage(s) (committed makespan "
                    "> LCT): %s. The bound must hold, so this is a genuine bug. "
                    "newly_committed=%s",
                    st.iter_idx,
                    len(window_violations),
                    window_violations[:10],
                    newly_committed_jobs,
                )

            if prev_full_obj_val is not None and full_obj_val > prev_full_obj_val:
                # Smoking gun: last-stage makespan of the SAME committed jobs in
                # incumbent order vs the CP-chosen order (both within LCT). The
                # difference is the S^R slack the reorder spent on the last stage.
                last_stage = ctx.stage_ids[-1] if ctx.stage_ids else None
                cp_msp_last = (
                    makespan_after.get(last_stage) if last_stage is not None else None
                )
                lct_last = (
                    stage_2_lct_map.get(last_stage) if last_stage is not None else None
                )
                inc_msp_last = None
                if last_stage is not None and newly_committed_jobs:
                    committed_set = set(newly_committed_jobs)
                    inc_order = [j for j in self.job_sequence if j in committed_set]
                    inc_sched = PermutationFlowshopScheduleLite(
                        ctx.stage_ids,
                        job_2_stage_2_p_map=ctx.job_2_stage_2_p_dict,
                        job_2_due_map=ctx.instance.job_2_duedate_map,
                    )
                    inc_sched.extend_jobs(inc_order, stage_2_est_map=stage_2_est_map)
                    inc_msp_last = inc_sched.get_stage_2_makespan_map().get(last_stage)
                logging.warning(
                    "(iter %d) FULL OBJECTIVE INCREASED %d -> %d (%+d) "
                    "[committed=%d + tail=%d]; window bound %s. Reorder spent S^R slack: "
                    "last-stage committed makespan incumbent-order=%s -> CP-order=%s "
                    "(LCT=%s); the larger last-stage makespan delays the greedily "
                    "dispatched tail. Still <= the starting incumbent, but non-monotone "
                    "across iterations. Newly committed (CP order): %s.",
                    st.iter_idx,
                    prev_full_obj_val,
                    full_obj_val,
                    full_obj_val - prev_full_obj_val,
                    committed_obj,
                    tail_obj,
                    "HELD" if not window_violations else "VIOLATED",
                    inc_msp_last,
                    cp_msp_last,
                    lct_last,
                    newly_committed_jobs,
                )

            prev_full_obj_val = full_obj_val

            if draw_gantt:
                if len(st.time_fixed_pool) > 1:
                    time_fixed_sol_output_path = ctx.get_file_path_for_subroutine(
                        f"_{note}_1_time_fixed_solution.yaml"
                    )
                    self.save_schedule_lite_to_yaml(
                        st.time_fixed_sol, time_fixed_sol_output_path
                    )
                    logging.info(
                        "Saved time-fixed solution to: %s", time_fixed_sol_output_path
                    )

                if len(st.profile_fixed_jobs) > 0:
                    profile_fixed_sol_output_path = ctx.get_file_path_for_subroutine(
                        f"_{note}_2_profile_fixed_solution.yaml"
                    )
                    self.save_schedule_lite_to_yaml(
                        st.profile_fixed_sol, profile_fixed_sol_output_path
                    )
                    logging.info(
                        "Saved profile-fixed solution to: %s",
                        profile_fixed_sol_output_path,
                    )

                full_sol_output_path = ctx.get_file_path_for_subroutine(
                    f"_{note}_3_full_solution.yaml"
                )
                self.save_schedule_lite_to_yaml(full_sol, full_sol_output_path)
                logging.info(
                    "Saved full-dispatched solution to: %s", full_sol_output_path
                )

            self._log_snapshot(
                picked_obj_val=full_obj_val,
                note=note,
                timestamp=last_timestamp,
                bound_val=committed_obj,
            )

        # ---- finalize: full sequence = committed + profile_fixed + remaining ----
        final_seq = (
            list(st.committed_time_fixed_jobs)
            + list(st.profile_fixed_jobs)
            + list(st.remaining_jobs)
        )

        # Build full FlowshopSchedule for return
        final_solution = FlowshopSchedule.from_stage_name_list(ctx.stage_ids)
        for j in final_seq:
            final_solution.dispatch_job_by_stages(
                j, ctx.stage_ids, ctx.job_2_stage_2_p_dict[j], after_last=True
            )

        if error_if_infeasible:
            ctx.check_feasibility(final_solution)

        # record final
        last_obj_val = ctx.get_obj_value(final_solution)

        return PwCpResult(
            schedule=final_solution,
            sub_obj_store=st.sub_obj_store,
            last_obj_value=last_obj_val,
        )
