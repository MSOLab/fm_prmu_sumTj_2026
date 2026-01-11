import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from mbls.cpsat import CpsatSolverReport, CpsatStatus, ObjValueBoundStore
from ortools.sat.python.cp_model import CpSolver
from routix import ElapsedTimer
from routix.util.comparison import float_a_leq_b
from schore.parameters_examples.shop.flow import FlowshopDuedateParameters
from schore.schedule_examples.shop.flow import FlowshopSchedule

from flowshop_tardiness.cpsat_model_2.position import BaseModelBuilder, Params, Vars
from flowshop_tardiness.fm_prmu import PermutationFlowshopScheduleLite

# TODO: add tests for this module


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
        mdl,
        computational_time: float,
        solver_thread_cnt: int,
        e_timer: ElapsedTimer | None = None,
        obj_value_is_valid: bool = False,
        obj_bound_is_valid: bool = False,
        log_level_obj_value: int = logging.INFO,
        log_level_obj_bound: int = logging.INFO,
    ) -> CpsatSolverReport: ...

    def get_job_sequence_from_solver(self, params: Params, vars: Vars) -> list[str]: ...

    # schedule build
    def create_schedule_from_sequence(
        self, params: Params, j_name_sequence: list[str]
    ) -> FlowshopSchedule: ...

    # optional
    def set_sumTj_lower_bound(self, mdl, vars: Vars, bound: float | None) -> None: ...
    def add_obj_value_log(self, ts: float, value: float, is_maximize=None) -> None: ...
    @property
    def obj_store(self): ...
    def export_solution_to_yaml(
        self,
        start_time_map: dict[tuple[str, str], int],
        end_time_map: dict[tuple[str, str], int],
        output_path: Path | None = None,
        encoding="utf-8",
    ) -> None: ...
    def get_file_path_for_subroutine(self, suffix: str): ...
    def _get_call_context_of_current_method(self) -> str: ...


@dataclass
class PwCpRunState:
    timer: ElapsedTimer

    job_sequence: list[str]
    """Given full job sequence."""
    added_batch_size: int

    sub_obj_store: ObjValueBoundStore[int]
    job_cnt: int
    given_sol: PermutationFlowshopScheduleLite
    """Schedule by the given job sequence with tail jobs pushed back."""

    target_job_subset: set[str]
    last_obj_val: int
    last_job_seq: list[str]
    last_solution: PermutationFlowshopScheduleLite | None
    """The schedule to be built incrementally."""


@dataclass
class PwCpResult:
    schedule: FlowshopSchedule
    sub_obj_store: ObjValueBoundStore[int]
    last_obj_value: int


class PwCpConstructor:
    def __init__(self, ctx: PwCpContext):
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
        added_batch_size: int = 1,
        solver_thread_cnt: int | None = None,
        max_time_per_add: float | None = None,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> PwCpResult:
        ctx = self.ctx
        timer = ElapsedTimer()
        sub_obj_store = ObjValueBoundStore[int]()
        sub_obj_store.obj_value_series.name = "ObjVal after dispatch"
        sub_obj_store.obj_bound_series.name = "ObjVal before dispatch"

        job_cnt: int = len(job_sequence)

        given_sol = PermutationFlowshopScheduleLite(
            ctx.stage_ids,
            job_2_stage_2_p_map=ctx.job_2_stage_2_p_dict,
            job_2_due_map=ctx.instance.job_2_duedate_map,
        )
        given_sol.extend_jobs(job_sequence)
        given_sol.push_back_tail_jobs_keep_tardiness(job_cnt)
        if draw_gantt:
            given_sol_output_path = ctx.get_file_path_for_subroutine(
                "_0_pushed_back_solution.yaml"
            )
            self.save_schedule_lite_to_yaml(given_sol, given_sol_output_path)
            logging.info("Saved pushed-back solution to: %s", given_sol_output_path)

        self._st = PwCpRunState(
            timer=timer,
            job_sequence=job_sequence,
            added_batch_size=added_batch_size,
            sub_obj_store=sub_obj_store,
            job_cnt=job_cnt,
            given_sol=given_sol,
            target_job_subset=set(),
            last_solution=None,
            last_job_seq=[],
            last_obj_val=0,
        )

        try:
            return self._run_loop(
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
        st = self._require_state()
        _base_sol = base_sol.deepcopy()

        if len(already_scheduled_job_set) == st.job_cnt:
            return _base_sol
        remaining_jobs = [
            j for j in st.job_sequence if j not in already_scheduled_job_set
        ]
        full_sched = _base_sol
        full_sched.extend_jobs(remaining_jobs)
        return full_sched

    def _log_snapshot(
        self,
        picked_sol: PermutationFlowshopScheduleLite,
        already_scheduled_job_set: set[str],
        note: str,
        timestamp: float,
        iter_report: CpsatSolverReport | None = None,
        draw_gantt: bool = False,
    ) -> None:
        """Log a snapshot of the current state.

        Args:
            picked_sol (PermutationFlowshopScheduleLite): The picked solution to log.
            already_scheduled_job_set (set[str]): The set of already scheduled jobs.
            note (str): A note to attach to the log.
            timestamp (float): The timestamp for the log entry.
                If None, the current subroutine timer's elapsed time is used.
                Defaults to None.
            iter_report (CpsatSolverReport | None, optional): CP solver report for the iteration.
                If provided, objective bounds are extracted.
                Defaults to None.
        """
        ctx = self.ctx
        st = self._require_state()
        sub_obj_store = st.sub_obj_store

        full_sched = self._make_all_dispatched(picked_sol, already_scheduled_job_set)
        full_sched_value = full_sched.get_total_tardiness()
        if draw_gantt:
            number = len(already_scheduled_job_set)
            full_sol_output_path = ctx.get_file_path_for_subroutine(
                f"_{number}_full_disp_solution.yaml"
            )
            self.save_schedule_lite_to_yaml(full_sched, full_sol_output_path)
            logging.info("Saved full-dispatched solution to: %s", full_sol_output_path)
        sub_obj_store.add_obj_value(timestamp, full_sched_value, is_maximize=None)

        if iter_report is not None and getattr(iter_report, "is_feasible", False):
            records = iter_report.obj_value_records
            seen = set()
            for elapsed, val in records:
                sub_obj_store.add_obj_bound(elapsed, val, is_maximize=None)
                seen.add((elapsed, val))
            final_val = picked_sol.get_total_tardiness()
            if (timestamp, final_val) not in seen:
                sub_obj_store.add_obj_bound(timestamp, final_val, is_maximize=None)
            sub_obj_store.add_last_timestamp_note(
                note, obj_value_is_valid=True, obj_bound_is_valid=True
            )
        else:
            picked_val = picked_sol.get_total_tardiness()
            sub_obj_store.add_obj_bound(timestamp, picked_val, is_maximize=None)
            sub_obj_store.add_last_timestamp_note(
                note, obj_value_is_valid=True, obj_bound_is_valid=True
            )

    def _solve_cp_model_lexico_for_batch(
        self,
        sub_instance: FlowshopDuedateParameters,
        stage_2_est_map: dict[str, int] | None,
        stage_2_lct_map: dict[str, int] | None,
        sumTj_offset: int | None,
        solver_timelimit: float,
        solver_thread_cnt: int | None,
        all_jobs_are_included: bool,
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
        if solver_thread_cnt is None:
            solver_thread_cnt = 1
        st = self._require_state()
        ctx = self.ctx

        subjob_id_list: list[str] = sub_instance.job_id_list
        init_pi_hint: list[int] = list(range(len(subjob_id_list)))

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
            )
            if vars1.total_tardiness is None:
                raise RuntimeError(
                    "Unexpected: total_tardiness variable is None after CP building."
                )
            mdl1.minimize(vars1.total_tardiness)

            if (
                all_jobs_are_included
                and ctx.solution_manager.best_obj_bound is not None
                and not math.isnan(ctx.solution_manager.best_obj_bound)
            ):
                ctx.set_sumTj_lower_bound(
                    mdl1, vars1, bound=ctx.solution_manager.best_obj_bound
                )

            mdl1.clear_hints()
            for idx, k in enumerate(params1.j_list):
                mdl1.add_hint(vars1.pi[k], init_pi_hint[idx])

            _timelimit = ctx.get_remaining_time_limit(solver_timelimit)
            report1 = ctx.solve_cp_model_2(
                mdl1,
                _timelimit,
                solver_thread_cnt,
                e_timer=st.timer,
                obj_value_is_valid=all_jobs_are_included,
                obj_bound_is_valid=False,
                log_level_obj_value=logging.NOTSET,
                log_level_obj_bound=logging.NOTSET,
            )
            if not getattr(report1, "is_feasible", False):
                logging.info("No solution from phase 1 CP; skip phase 2.")
                return report1, subjob_id_list

            best_sumTj = int(ctx.solver.Value(vars1.total_tardiness))
            phase1_pi: list[int] = [
                int(ctx.solver.Value(vars1.pi[k])) for k in params1.j_list
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
            elapsed_time = st.timer.elapsed_sec
            report1 = CpsatSolverReport(
                elapsed_time=elapsed_time,
                obj_value=report_obj_val,
                obj_bound=None,
                status=CpsatStatus.OPTIMAL,
                obj_value_records=[(elapsed_time, report_obj_val)],
                obj_bound_records=[],
            )
            best_sumTj = report_obj_val
            phase1_pi = init_pi_hint

        job_seq: list[str] = [subjob_id_list[idx] for idx in phase1_pi]

        if all_jobs_are_included:
            logging.info("All jobs are included; skip phase 2.")
            return report1, job_seq

        mdl2, params2, vars2 = self.builder.build(
            sub_instance,
            stage_2_est_map=stage_2_est_map,
            stage_2_lct_map=stage_2_lct_map,
            sumTj_offset=sumTj_offset,
        )
        if vars2.sum_latest_completion is None:
            raise RuntimeError(
                "Unexpected: sum_latest_completion variable is None after CP building."
            )
        # Freeze primary objective value
        mdl2.add(vars2.total_tardiness == best_sumTj)
        # Minimize secondary objective
        mdl2.minimize(vars2.sum_latest_completion)

        # Add hints from phase1
        for idx, k in enumerate(params2.j_list):
            mdl2.add_hint(vars2.pi[k], phase1_pi[idx])

        _timelimit = ctx.get_remaining_time_limit(solver_timelimit)
        # Solve without logging objective values / bounds
        report2 = ctx.solve_cp_model_2(
            mdl2,
            _timelimit,
            solver_thread_cnt,
            e_timer=st.timer,
            obj_value_is_valid=all_jobs_are_included,
            obj_bound_is_valid=False,
            log_level_obj_value=logging.NOTSET,
            log_level_obj_bound=logging.NOTSET,
        )
        if not getattr(report2, "is_feasible", False):
            logging.info("Sub-CP infeasible in phase 2; use phase 1 solution.")
            return report1, job_seq

        phase2_pi: list[int] = [
            int(ctx.solver.Value(vars2.pi[k])) for k in params2.j_list
        ]
        job_seq2: list[str] = [subjob_id_list[idx] for idx in phase2_pi]
        # use report 1 which has total tardiness info
        return report1, job_seq2

    def _run_loop(
        self,
        solver_thread_cnt: int | None = None,
        max_time_per_add: float | None = None,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> PwCpResult:
        st = self._require_state()
        ctx = self.ctx

        st.last_obj_val = 0
        st.last_job_seq = []
        st.last_solution = PermutationFlowshopScheduleLite(
            ctx.stage_ids,
            job_2_stage_2_p_map=ctx.job_2_stage_2_p_dict,
            job_2_due_map=ctx.instance.job_2_duedate_map,
        )

        sequence_of_job_sublist = [
            st.job_sequence[i : i + st.added_batch_size]
            for i in range(0, len(st.job_sequence), st.added_batch_size)
        ]
        job_sublist_cnt = len(sequence_of_job_sublist)

        for bidx, added_job_sublist in enumerate(sequence_of_job_sublist):
            _timelimit = ctx.get_remaining_time_limit(max_time_per_add)
            if float_a_leq_b(_timelimit, 0):
                logging.info(
                    "(batch %d/%d) Time over before CP -> dispatch remaining jobs.",
                    bidx + 1,
                    job_sublist_cnt,
                )
                break
            logging.info(
                "(batch %d/%d) Preparing to add %d jobs (time limit: %.2f sec).",
                bidx + 1,
                job_sublist_cnt,
                len(added_job_sublist),
                _timelimit,
            )

            job_subset_cnt = len(st.target_job_subset) + len(added_job_sublist)
            all_jobs_are_included = job_subset_cnt == st.job_cnt

            sub_instance = ctx.instance.get_subinstance(added_job_sublist)

            sumTj_offset = None
            stage_2_est_map = None
            stage_2_lct_map = None

            if isinstance(st.last_solution, PermutationFlowshopScheduleLite):
                sumTj_offset = int(st.last_obj_val)
                stage_2_est_map = st.last_solution.get_stage_2_makespan_map()

            if not all_jobs_are_included:
                next_job = st.given_sol.get_next_job_name(added_job_sublist[-1])
                stage_2_lct_map = st.given_sol.get_stage_2_start_time_map(next_job)

            logging.info(
                "(batch %d/%d) Start CP on %d jobs at %s",
                bidx + 1,
                job_sublist_cnt,
                len(added_job_sublist),
                st.timer.get_formatted_elapsed_time(),
            )

            iter_report, job_seq_to_be_appended = self._solve_cp_model_lexico_for_batch(
                sub_instance=sub_instance,
                stage_2_est_map=stage_2_est_map,
                stage_2_lct_map=stage_2_lct_map,
                sumTj_offset=sumTj_offset,
                solver_timelimit=_timelimit,
                solver_thread_cnt=solver_thread_cnt,
                all_jobs_are_included=all_jobs_are_included,
            )
            last_timestamp = st.timer.elapsed_sec

            if not getattr(iter_report, "is_feasible", False):
                logging.info(
                    "(batch %d/%d) no solution from Sub-CP -> dispatch remaining jobs.",
                    bidx + 1,
                    job_sublist_cnt,
                )

            # Update state
            st.target_job_subset.update(added_job_sublist)
            st.last_job_seq += job_seq_to_be_appended
            st.last_solution.extend_jobs(job_seq_to_be_appended)
            st.last_obj_val = st.last_solution.get_total_tardiness()

            seq_changed: bool = job_seq_to_be_appended != added_job_sublist
            if draw_gantt and seq_changed:
                batch_sol_output_path = ctx.get_file_path_for_subroutine(
                    f"_{job_subset_cnt}_batch_disp_solution.yaml"
                )
                self.save_schedule_lite_to_yaml(st.last_solution, batch_sol_output_path)
                logging.info(
                    "Saved batch-dispatched solution to: %s", batch_sol_output_path
                )

            self._log_snapshot(
                st.last_solution,
                st.target_job_subset,
                note=f"{len(st.target_job_subset)}/{st.job_cnt}",
                iter_report=iter_report,
                timestamp=last_timestamp,
                draw_gantt=draw_gantt and seq_changed,
            )

        # -------- finish by dispatch remaining --------
        remaining_jobs = [j for j in st.job_sequence if j not in st.target_job_subset]
        if remaining_jobs:
            st.last_solution.extend_jobs(remaining_jobs)
            self._log_snapshot(
                st.last_solution,
                set(st.job_sequence),
                note="Final dispatch",
                timestamp=st.timer.elapsed_sec,
            )
            st.last_obj_val = st.last_solution.get_total_tardiness()

        # Build a solution to return
        last_solution = FlowshopSchedule.from_stage_name_list(ctx.stage_ids)
        for j in st.last_solution.get_job_sequence():
            last_solution.dispatch_job_by_stages(
                j, ctx.stage_ids, ctx.job_2_stage_2_p_dict[j], after_last=True
            )
        if ctx.get_obj_value(last_solution) != st.last_obj_val:
            raise RuntimeError(
                "Discrepancy in total tardiness in final solution: %d (expected) vs %d (actual)."
                % (st.last_obj_val, ctx.get_obj_value(last_solution))
            )
        if error_if_infeasible:
            ctx.check_feasibility(last_solution)

        return PwCpResult(
            schedule=last_solution,
            sub_obj_store=st.sub_obj_store,
            last_obj_value=ctx.get_obj_value(last_solution),
        )
