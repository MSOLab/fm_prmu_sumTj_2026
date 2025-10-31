import logging
import math
import random
from typing import Callable

from mbls.cpsat import CpsatSolverReport, ObjValueBoundStore
from routix import ElapsedTimer
from routix.util.comparison import float_a_leq_b, float_a_stl_b
from schore.schedule_examples.shop.flow import FlowshopSchedule

from ..report import FsSubroutineReport
from .controller_core import FlowshopTardinessControllerCore

REL_TOL = 1e-9  # for safe float comparisons


class FlowshopTardinessCpLnsController(FlowshopTardinessControllerCore):
    cp_model_presolve: bool | None = False  # TODO: make it configurable
    """Whether to presolve the CP model before solving."""

    def set_random_seed(self, seed: int):
        return super().set_random_seed(seed)

    def set_cp_model_as_base_cp_model(self) -> None:
        return super().set_cp_model_as_base_cp_model()

    def set_0_as_lb(self) -> None:
        """
        Set zero as a valid lower bound for the objective (total tardiness).
        """
        log_time = self.timer.elapsed_sec
        report = FsSubroutineReport(
            elapsed_time=0.0, obj_value=None, obj_bound=0.0, is_init=True
        )
        self.solution_manager.register(report, None)

        self.add_obj_bound_log(log_time, 0, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

    # Subroutine: solve base CP model

    def solve_base_cp_model(
        self,
        computational_time: float,
        solver_thread_cnt: int,
        is_initial_solution: bool = False,
        draw_gantt: bool = False,
    ):
        """
        Solve the base CP model for the F|prmu|C_max problem.

        This method resets the CP model and solves it with the given computational time and number of workers.
        - If `is_initial_solution` is True, the solution is treated as the initial solution (e.g., for logging or summary purposes).
        - If `is_initial_solution` is False, the incumbent solution (if it exists) is applied as a hint to the CP model before solving.
        - If `draw_gantt` is True, a Gantt chart of the solution is generated after solving.

        Args:
            computational_time (float): The maximum computational time in seconds for solving the CP model.
            solver_thread_cnt (int): The number of parallel workers (threads) to use during search.
            is_initial_solution (bool, optional): If True, marks this run as producing the initial solution (affects summary/logging). Defaults to False.
            draw_gantt (bool, optional): If True, draws the Gantt chart of the solution after solving. Defaults to False.
        """
        if self.base_cp_model_is_set:
            self.cp_model.delete_added_constraints()
        else:
            raise RuntimeError(
                "Base CP model is not set. Call set_cp_model_as_base_cp_model() first."
            )

        if is_initial_solution:
            self.solve_current_cp_remaining_time_limit(
                computational_time,
                solver_thread_cnt,
                cp_model_presolve=self.cp_model_presolve,
                obj_value_is_valid=True,
                obj_bound_is_valid=True,
                is_initial_solution=True,
                error_if_infeasible=True,
                draw_gantt=draw_gantt,
            )
        else:
            # If it is not an initial solution, apply the incumbent solution as a hint
            self.solve_with_initial_solution(
                computational_time,
                solver_thread_cnt,
                cp_model_presolve=self.cp_model_presolve,
                obj_value_is_valid=True,
                obj_bound_is_valid=True,
                error_if_infeasible=True,
                draw_gantt=draw_gantt,
            )

    # Helper methods

    def get_dispatched_schedule(self, job_sequence: list[str]) -> FlowshopSchedule:
        # Create an empty schedule
        schedule = FlowshopSchedule.from_stage_name_list(self.stage_ids)
        # Dispatch
        for j in job_sequence:
            added = schedule.dispatch_job_by_stages(
                j, self.stage_ids, self.job_2_stage_2_p_dict[j], after_last=True
            )
            if added is None:
                raise ValueError(f"Failed to add job {j} to the schedule.")
        return schedule

    def initialize_by_sequential_dispatching(
        self,
        job_sequence: list[str],
        log_msg_format: str,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ):
        sub_timer = ElapsedTimer()
        schedule = self.get_dispatched_schedule(job_sequence)
        if error_if_infeasible:
            self.check_feasibility(schedule)
        obj_value = self.get_obj_value(schedule)
        logging.info(log_msg_format.format(obj_value=obj_value))

        # Create report and register the new solution
        report = FsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
        )
        was_updated = self.solution_manager.register(report, schedule)

        # Log
        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, obj_value, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )
        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    # Subroutine: dispatch by EDD rule

    def initialize_by_edd(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        job_sequence = self.get_edd_sequence()
        self.initialize_by_sequential_dispatching(
            job_sequence,
            "Initialized by EDD with total tardiness {obj_value}",
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def get_edd_sequence(self) -> list[str]:
        """Get the job sequence by EDD rule."""
        return sorted(
            self.instance.job_id_list,
            key=lambda j: self.instance.job_2_duedate_map[j],
        )

    # Subroutine: dispatch by EDDP rule

    def initialize_by_eddp(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        job_sequence = self.get_eddp_sequence()
        self.initialize_by_sequential_dispatching(
            job_sequence,
            "Initialized by EDDP with total tardiness {obj_value}",
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def get_eddp_sequence(self) -> list[str]:
        """
        Get job sequence by Earliest Due Date with Processing times (EDDP).
        Sort by d[j] / total_proc_time(j) ascending. Stable tie-breakers applied.

        Returns:
            list[str]: ordered job ids
        """
        dmap = self.instance.job_2_duedate_map
        pmap = self.job_2_stage_2_p_dict

        def total_p(j: str) -> float:
            # Sum processing times across all stages for job j
            return sum(pmap[j].get(s, 0.0) for s in self.stage_ids)

        def key_fn(j: str):
            tp = total_p(j)
            # safe division (handles pathological zero-proc jobs)
            denom = tp if tp > 0.0 else REL_TOL
            ratio = dmap[j] / denom
            # tie-breakers: due date, total proc time, id (for determinism)
            return (ratio, dmap[j], tp, j)

        return sorted(self.instance.job_id_list, key=key_fn)

    # Subroutine: dispatch by modified due date rule

    def append_one_by_one(
        self,
        min_value_function: Callable[[str, int], int | float],
        log_format: str,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        sub_timer = ElapsedTimer()

        i_list = self.stage_ids
        dmap = self.instance.job_2_duedate_map
        pmap = self.job_2_stage_2_p_dict

        # pre-calc total processing time used for tie-breaking
        total_proc_time_map = {
            j: sum(pmap[j][i] for i in i_list) for j in self.instance.job_id_list
        }

        remaining = list(self.instance.job_id_list)
        schedule = FlowshopSchedule.from_stage_name_list(i_list)
        frontier = {i: 0 for i in i_list}

        def simulate_completion_if_appended(job_id: str) -> tuple[int, dict[str, int]]:
            """
            Given current frontier, simulate appending 'job_id' and return its completion time on the last stage.
            Does NOT mutate the real frontier.
            """
            f = frontier.copy()
            prev = 0
            for i in i_list:
                p = pmap[job_id][i]
                start_time = f[i]
                if prev > start_time:
                    start_time = prev
                end_time = start_time + p
                f[i] = end_time
                prev = end_time
            return f[i_list[-1]], f

        while remaining:
            best_job = remaining[0]
            Cj, best_f = simulate_completion_if_appended(best_job)
            best_criteria = min_value_function(best_job, Cj)

            for j in remaining[1:]:
                Cj, f = simulate_completion_if_appended(j)
                criteria = min_value_function(j, Cj)

                # Tie-breakers for determinism: (criteria, d_j, total_p, job_id)
                if best_criteria is None or criteria < best_criteria:
                    best_job = j
                    best_criteria = criteria
                    best_f = f
                elif criteria == best_criteria:
                    # secondary criteria
                    dj = float(dmap[j])
                    bj = float(dmap[best_job])
                    if dj < bj:
                        best_job = j
                        best_f = f
                    elif dj == bj:
                        # total processing time tie-breaker
                        tp_j = total_proc_time_map[j]
                        tp_b = total_proc_time_map[best_job]
                        if tp_j < tp_b or (tp_j == tp_b and j < best_job):
                            best_job = j
                            best_f = f

            # append the selected job and commit its frontier
            remaining.remove(best_job)
            schedule.dispatch_job_by_stages(
                best_job, i_list, pmap[best_job], after_last=True
            )
            frontier = best_f

        if error_if_infeasible:
            self.check_feasibility(schedule)
        obj_value = self.get_obj_value(schedule)
        logging.info(log_format.format(obj_value=obj_value))

        # Create report and register the new solution
        report = FsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
        )
        was_updated = self.solution_manager.register(report, schedule)

        # Log
        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, obj_value, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )
        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def initialize_by_mdd(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        self.append_one_by_one(
            self.mdd_min_value,
            "Initialized by MDD with total tardiness {obj_value}",
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def mdd_min_value(self, job_id: str, completion_time: int) -> int:
        return_val = self.instance.job_2_duedate_map[job_id]
        if completion_time > return_val:
            return completion_time
        return return_val

    def initialize_by_slack(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        self.append_one_by_one(
            self.slack_min_value,
            "Initialized by SLACK with total tardiness {obj_value}",
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def slack_min_value(self, job_id: str, completion_time: int) -> int:
        return self.instance.job_2_duedate_map[job_id] - completion_time

    def initialize_by_srmwk(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        self.append_one_by_one(
            self.srmwk_min_value,
            "Initialized by SRMWK with total tardiness {obj_value}",
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def srmwk_min_value(self, job_id: str, completion_time: int) -> float:
        p_total = sum(
            self.job_2_stage_2_p_dict[job_id].get(s, 0.0) for s in self.stage_ids
        )
        if p_total <= 0.0:
            return float("inf")  # Handle zero processing time jobs
        return (
            float(self.instance.job_2_duedate_map[job_id] - completion_time) / p_total
        )

    # Subroutine: insertion heuristics

    def _simulate_append(
        self, stage_2_endtime_map: dict[str, int], job_id: str
    ) -> dict[str, int]:
        """Simulate appending a job to the schedule.

        Args:
            stage_2_endtime_map (dict[str, int]): Current completion times at each stage.
                Missing stages are assumed to have 0 completion time.
            job_id (str): Job ID to append.

        Returns:
            dict[str, int]: Updated completion times after appending the job.
        """
        pmap = self.job_2_stage_2_p_dict[job_id]
        return_dict: dict[str, int] = {}
        prev = 0
        for i in self.stage_ids:
            start = stage_2_endtime_map[i]
            if prev > start:
                start = prev
            end = start + pmap[i]
            return_dict[i] = end
            prev = end
        return return_dict

    def _compute_prefix_frontiers_and_sumTj(
        self, job_seq: list[str]
    ) -> tuple[dict[int, dict[str, int]], dict[int, int]]:
        """Compute prefix frontiers and cumulative total tardiness for a job sequence.

        For each prefix length k (0 .. len(job_seq)):
          - prefix_frontiers[k]: dict[stage_id -> completion_time] after scheduling
            the first k jobs of job_seq in the given order (flow shop permutation).
          - prefix_sumTj[k]: cumulative total tardiness sum_{j in first k jobs} T_j,
            where T_j = max(C_j_last - due_date_j, 0) and C_j_last is the job's
            completion time on the last stage.

        Returned dictionaries always contain key 0:
          prefix_frontiers[0] = {stage_id: 0 for all stages}
          prefix_sumTj[0] = 0

        This function is used to enable fast reuse of head (prefix) information
        during insertion evaluations: head tardiness can be looked up directly
        instead of recomputed.

        Args:
            job_seq (list[str]): Ordered list of job ids to evaluate.

        Returns:
            tuple[dict[int, dict[str, int]], dict[int, int]]: A tuple containing:
            - prefix_frontiers: mapping position -> stage ID -> completion times.
            - prefix_sumTj: mapping position -> cumulative total tardiness.

        Notes:
            - Tardiness never negative (clamped at 0).
            - Uses _simulate_append for each incremental extension.
            - If job_seq is empty, returns ({0: {stage:0}}, {0:0}).
            - Assumes every job has defined processing time on every stage.
        """
        dmap = self.instance.job_2_duedate_map
        pos_2_stage_2_endtime_map = {0: {i: 0 for i in self.stage_ids}}

        prefix_sumTj: dict[int, int] = {0: 0}
        for j_idx, j in enumerate(job_seq):
            f_prev = pos_2_stage_2_endtime_map[j_idx]
            f_new = self._simulate_append(f_prev, j)
            C_last = f_new[self.last_stage_id]
            Tj = C_last - dmap[j]
            if Tj < 0:
                Tj = 0
            pos_2_stage_2_endtime_map[j_idx + 1] = f_new
            prefix_sumTj[j_idx + 1] = prefix_sumTj.get(j_idx, 0) + Tj
        return pos_2_stage_2_endtime_map, prefix_sumTj

    def _compute_sumTj_Cmax_from_sequence(self, job_seq: list[str]) -> tuple[int, int]:
        """Compute total tardiness and makespan from a job sequence.

        Args:
            job_seq (list[str]): Ordered list of job ids to evaluate.

        Returns:
            tuple[int, int]: (total_tardiness, makespan) for the given job sequence.
        """
        prefix_frontiers, prefix_sumTj = self._compute_prefix_frontiers_and_sumTj(
            job_seq
        )
        n = len(job_seq)
        return prefix_sumTj[n], prefix_frontiers[n][self.last_stage_id]

    @staticmethod
    def _tie_crit_from_tm(sumTj: int, Cmax: int, tie_breaker: str) -> tuple[int, int]:
        """Calculate tie breaking criteria from tie breaker name.

        Args:
            sumTj (int): total tardiness
            Cmax (int): makespan
            tie_breaker (str): tie breaker name

        Raises:
            ValueError: if tie_breaker is unknown

        Returns:
            tuple[int, int]: (crit1, crit2) for comparison
        """
        if tie_breaker == "default":
            return (sumTj, 0)
        if tie_breaker == "NEH-T":
            return (sumTj, Cmax)
        if tie_breaker == "NEH(M)":
            return (sumTj + Cmax, 0)
        if tie_breaker == "nehedd3":
            return (sumTj + Cmax, sumTj)
        if tie_breaker == "nehedd4":
            return (sumTj + Cmax, Cmax)
        raise ValueError(f"Unknown tie_breaker: {tie_breaker}")

    def _eval_insert_with_criteria(
        self, seq_now: list[str], job_id: str, tie_breaker: str = "default"
    ) -> tuple[int, int, int]:
        """
        Evaluate all insertion positions of job_id into seq_now.
        Returns (best_pos, best_sumTj, best_Cmax).
        Uses prefix reuse: head tardiness reused; tail recomputed from the chosen frontier.
        """
        dmap = self.instance.job_2_duedate_map

        if not seq_now:
            # only one position
            f0 = {i: 0 for i in self.stage_ids}
            new_f = self._simulate_append(f0, job_id)
            Cmax = new_f[self.last_stage_id]
            sumTj = Cmax - dmap[job_id]
            if sumTj < 0:
                sumTj = 0
            return 0, sumTj, Cmax

        prefix_frontiers, prefix_tardy = self._compute_prefix_frontiers_and_sumTj(
            seq_now
        )
        best_pos = 0
        best_sumTj: int | None = None
        best_Cmax: int | None = None
        best_crit1: int | None = None
        best_crit2: int | None = None

        # try all positions pos \in [0..len]
        for pos in range(len(seq_now) + 1):
            # head part tardiness is reused
            head_tardy = prefix_tardy[pos]
            frontier = prefix_frontiers[pos]

            # insert the new job
            new_frontier = self._simulate_append(frontier, job_id)
            C_new = new_frontier[self.last_stage_id]
            new_sumTj = head_tardy + max(C_new - dmap[job_id], 0)

            # simulate tail jobs (pos..end) on this new frontier
            for k in range(pos, len(seq_now)):
                j_tail = seq_now[k]
                new_frontier = self._simulate_append(new_frontier, j_tail)
                C_tail = new_frontier[self.last_stage_id]
                new_sumTj += max(C_tail - dmap[j_tail], 0)

            new_Cmax = new_frontier[self.last_stage_id]
            crit1, crit2 = self._tie_crit_from_tm(new_sumTj, new_Cmax, tie_breaker)

            # choose best
            if (best_crit1 is None) or (crit1 < best_crit1):
                best_pos, best_sumTj, best_Cmax = pos, new_sumTj, new_Cmax
                best_crit1, best_crit2 = crit1, crit2
            elif (crit1 == best_crit1) and (best_crit2 is None or crit2 < best_crit2):
                best_pos, best_sumTj, best_Cmax = pos, new_sumTj, new_Cmax
                best_crit1, best_crit2 = crit1, crit2
                # if still tied, earlier position is preferred (stable)

        if best_sumTj is None:
            raise RuntimeError("Unexpected: best_sumTj is None after evaluation.")
        if best_Cmax is None:
            raise RuntimeError("Unexpected: best_Cmax is None after evaluation.")
        return best_pos, best_sumTj, best_Cmax

    def _run_neh_edd(
        self,
        method_name: str,
        tie_breaker: str,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """
        Generic NEH with EDD ordering (sum of tardiness objective), array-based fast evaluation.
        - No schedule/deepcopy during insertion trials (only once at the end).
        Complexity: O(n^2 * m) with small constants via prefix reuse.
        """
        sub_timer = ElapsedTimer()

        # 1) EDD order
        edd_order = self.get_edd_sequence()

        seq: list[str] = []

        # 2) NEH insertion by EDD order with fast evaluation
        for j in edd_order:
            pos, _, _ = self._eval_insert_with_criteria(seq, j, tie_breaker)
            seq.insert(pos, j)

        # 3) Build schedule once and register/log
        schedule = self.get_dispatched_schedule(seq)
        if error_if_infeasible:
            self.check_feasibility(schedule)

        obj_value = self.get_obj_value(schedule)
        logging.info(f"Initialized by {method_name} with total tardiness {obj_value}")

        report = FsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=True,
        )
        was_updated = self.solution_manager.register(report, schedule)

        log_time = self.timer.elapsed_sec
        self.add_obj_value_log(log_time, obj_value, is_maximize=False)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def initialize_by_nehedd(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        self._run_neh_edd("NEHEDD", "default", error_if_infeasible, draw_gantt)

    def initialize_by_neht(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        self._run_neh_edd("NEH-T", "NEH-T", error_if_infeasible, draw_gantt)

    def initialize_by_nehedd2(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        self._run_neh_edd("NEHEDD2", "NEH(M)", error_if_infeasible, draw_gantt)

    def initialize_by_nehedd3(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        self._run_neh_edd("NEHEDD3", "nehedd3", error_if_infeasible, draw_gantt)

    def initialize_by_nehedd4(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        self._run_neh_edd("NEHEDD4", "nehedd4", error_if_infeasible, draw_gantt)

    def improve_job_seq_by_insertion_single_pass(
        self, job_seq: list[str], tie_breaker: str = "default"
    ) -> list[str]:
        job_cnt = len(job_seq)
        if job_cnt <= 1:
            logging.info("Insertion improvement skipped: only one or zero jobs.")
            return job_seq
        # Quick sanity: unique IDs
        if len(set(job_seq)) != job_cnt:
            logging.warning(
                "Duplicate job IDs detected in sequence. Insertion pass may misbehave."
            )

        seq_before = list(job_seq)
        seq_after = list(seq_before)

        best_sumTj, best_Cmax = self._compute_sumTj_Cmax_from_sequence(seq_before)
        best_crit1, best_crit2 = self._tie_crit_from_tm(
            best_sumTj, best_Cmax, tie_breaker
        )

        for j in seq_before:
            j_idx = seq_after.index(j)
            seq_after.remove(j)
            pos, after_sumTj, after_Cmax = self._eval_insert_with_criteria(
                seq_after, j, tie_breaker
            )
            crit1, crit2 = self._tie_crit_from_tm(after_sumTj, after_Cmax, tie_breaker)

            after_is_better = (crit1 < best_crit1) or (
                crit1 == best_crit1 and crit2 < best_crit2
            )
            if after_is_better:
                seq_after.insert(pos, j)
                best_sumTj = after_sumTj
                best_Cmax = after_Cmax
                best_crit1 = crit1
                best_crit2 = crit2
            else:
                seq_after.insert(j_idx, j)  # revert to original spot for stability
            if self.time_is_up():
                logging.info(
                    f"Time limit reached during {j_idx + 1} / {job_cnt} insertion improvement."
                )
                break

        return seq_after

    def improve_by_insertion(
        self,
        tie_breaker: str = "default",
        max_passes: int | None = None,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        if self.solution_manager.incumbent_solution is None:
            raise RuntimeError("No incumbent solution to improve.")

        sub_timer = ElapsedTimer()

        # Incumbent job sequence
        seq_before = self.solution_manager.incumbent_solution.get_last_stage_job_list()
        job_cnt = len(seq_before)
        if job_cnt <= 1:
            logging.info("Insertion improvement skipped: only one or zero jobs.")
            return

        seq_after = list(seq_before)
        best_sumTj, best_Cmax = self._compute_sumTj_Cmax_from_sequence(seq_before)
        best_crit1, best_crit2 = self._tie_crit_from_tm(
            best_sumTj, best_Cmax, tie_breaker
        )
        logging.info(
            f"Initial: total tardiness {best_sumTj}, makespan {best_Cmax} (criteria: {best_crit1}, {best_crit2})."
        )

        improved_globally = False
        passes = 0
        # list of (global elapsed time, obj value)
        obj_value_log: list[tuple[float, float]] = []
        while max_passes is None or passes < max_passes:
            passes += 1

            seq_after = self.improve_job_seq_by_insertion_single_pass(
                seq_before, tie_breaker
            )
            after_sumTj, after_Cmax = self._compute_sumTj_Cmax_from_sequence(seq_after)
            crit1, crit2 = self._tie_crit_from_tm(after_sumTj, after_Cmax, tie_breaker)
            logging.info(
                f"Pass {passes}: total tardiness {after_sumTj}, makespan {after_Cmax} (criteria: {crit1}, {crit2})."
            )
            after_is_better = (crit1 < best_crit1) or (
                crit1 == best_crit1 and crit2 < best_crit2
            )
            if after_is_better:
                seq_before = list(seq_after)
                best_sumTj = after_sumTj
                best_Cmax = after_Cmax
                best_crit1 = crit1
                best_crit2 = crit2
                logging.info(
                    f"Pass {passes}: improved to total tardiness {best_sumTj}."
                )
                improved_globally = True
                obj_value_log.append((self.timer.elapsed_sec, best_sumTj))
            else:
                logging.info(f"Pass {passes}: no improvement, stopping.")
                break
            if self.time_is_up():
                max_pass_str = (
                    str(max_passes) if max_passes is not None else "unlimited"
                )
                logging.info(
                    f"Time limit reached during {passes} / {max_pass_str} insertion improvement passes."
                )
                break

        schedule = self.get_dispatched_schedule(seq_before)
        if error_if_infeasible:
            self.check_feasibility(schedule)
        obj_value = self.get_obj_value(schedule)
        logging.info(
            "Repeated-insertion improvement %s (tie=%s, passes=%d): total tardiness %d",
            "applied" if improved_globally else "no change",
            tie_breaker,
            passes,
            obj_value,
        )
        # Create report for the solution and register it
        report = FsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=False,
        )
        was_updated = self.solution_manager.register(report, schedule)

        if was_updated:
            log_time = self.timer.elapsed_sec
            obj_value_log.append((log_time, obj_value))
            self.extend_obj_value_log(obj_value_log, is_maximize=False)
            _last_timestamp_note = self._get_call_context_of_current_method()
            self.obj_store.add_last_timestamp_note(
                _last_timestamp_note, obj_value_is_valid=True
            )
            if draw_gantt:
                self.draw_incumbent_gantt()

    # Subroutine: incremental CP construction by incumbent solution's sequence

    def initialize_by_incremental_cp_incumbent_sequence(
        self,
        solver_thread_cnt: int,
        added_batch_size: int = 1,
        max_time_per_add: float | None = None,
        no_improvement_timelimit: float | None = None,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ):
        if self.solution_manager.incumbent_solution is None:
            raise RuntimeError(
                "No incumbent solution available. Cannot use its sequence for initialization."
            )
        job_sequence = (
            self.solution_manager.incumbent_solution.get_last_stage_job_list()
        )
        self.construct_solution_by_incremental_cp(
            job_sequence,
            solver_thread_cnt,
            added_batch_size,
            max_time_per_add,
            no_improvement_timelimit,
            is_init=False,  # TODO: set as True later
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def construct_solution_by_incremental_cp(
        self,
        job_sequence: list[str],
        solver_thread_cnt: int,
        added_batch_size: int = 2,
        max_time_per_add: float | None = None,
        no_improvement_timelimit: float | None = None,
        is_init: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ):
        """Constructs a solution by incrementally building and solving a CP model.

        This method takes an initial job sequence and iteratively builds a schedule.
        In each iteration, it adds a batch of jobs from the sequence to the
        problem, solves the resulting subproblem using a CP model, and then
        decides whether to keep the CP's solution or a simple dispatch-based
        solution for that subset of jobs. This process continues until all jobs
        are scheduled.

        The objective values of partial solutions (bounds) and full solutions
        (values, obtained by dispatching remaining jobs) are logged throughout
        the process for detailed analysis.

        Args:
            job_sequence (list[str]): The initial sequence of jobs to guide the
                incremental construction.
            solver_thread_cnt (int): The number of threads for the CP solver.
            added_batch_size (int, optional): The number of jobs to add in each
                incremental step. Defaults to 2.
            max_time_per_add (float | None, optional): The maximum time limit
                in seconds for each CP solve step. Defaults to None.
            no_improvement_timelimit (float | None, optional): A time limit for
                the CP solver to find an improvement. Defaults to None.
            is_init (bool, optional): Flag indicating if this is an initial
                solution construction. Defaults to False.
            error_if_infeasible (bool, optional): If True, raises an error if the
                final schedule is infeasible. Defaults to False.
            draw_gantt (bool, optional): If True, draws a Gantt chart if the
                resulting solution improves the incumbent. Defaults to False.
        """
        sub_timer = ElapsedTimer()
        last_solution: FlowshopSchedule | None = None
        sub_obj_store = ObjValueBoundStore[float]()
        sub_obj_store.obj_value_series.name = "ObjVal after dispatch"
        sub_obj_store.obj_bound_series.name = "ObjVal before dispatch"

        job_cnt = len(job_sequence)
        sequence_of_job_sublist = [
            job_sequence[i : i + added_batch_size]
            for i in range(0, job_cnt, added_batch_size)
        ]
        job_sublist_cnt = len(sequence_of_job_sublist)

        # ================== helpers (method-internal) ==================

        def _make_all_dispatched(
            base_sol: FlowshopSchedule, already_scheduled_job_set: set[str]
        ) -> FlowshopSchedule:
            """Create a new FlowshopSchedule by dispatching remaining jobs.

            Args:
                base_sol (FlowshopSchedule): The base solution to modify.
                already_scheduled_job_set (set[str]): The set of already scheduled jobs.

            Raises:
                ValueError: If a job cannot be dispatched to all stages.

            Returns:
                FlowshopSchedule: The modified flow shop schedule.
            """
            if len(already_scheduled_job_set) == job_cnt:
                return base_sol.deepcopy()

            remaining_jobs = [
                j for j in job_sequence if j not in already_scheduled_job_set
            ]
            full_sched = base_sol.deepcopy()
            for j in remaining_jobs:
                full_sched.dispatch_job_by_stages(
                    j,
                    self.stage_ids,
                    self.job_2_stage_2_p_dict[j],
                    after_last=True,
                )
            # Safety check
            self.check_feasibility(full_sched)
            return full_sched.deepcopy()

        def _log_snapshot(
            picked_sol: FlowshopSchedule,
            already_scheduled_job_set: set[str],
            note: str,
            *,
            iter_report: CpsatSolverReport | None = None,
            timestamp: float | None = None,
        ) -> None:
            """Log a snapshot of the current state.

            Args:
                picked_sol (FlowshopSchedule): The picked solution to log.
                already_scheduled_job_set (set[str]): The set of already scheduled jobs.
                note (str): A note to attach to the log.
                iter_report (CpsatSolverReport | None, optional): CP solver report for the iteration.
                    If provided, objective bounds are extracted. Defaults to None.
                timestamp (float | None, optional): The timestamp for the log entry.
                    If None, the current subroutine timer's elapsed time is used. Defaults to None.
            """
            ts = sub_timer.elapsed_sec if timestamp is None else timestamp

            # value = objective value of the schedule with all jobs
            full_sched = _make_all_dispatched(picked_sol, already_scheduled_job_set)
            full_sched_value = self.get_obj_value(full_sched)
            sub_obj_store.add_obj_value(ts, full_sched_value, is_maximize=None)

            # bounds = objective value of the partial schedule (by CP or by partially dispatched)
            if iter_report is not None and getattr(iter_report, "is_feasible", False):
                records = iter_report.obj_value_records
                seen = set()
                for elapsed, val in records:
                    sub_obj_store.add_obj_bound(elapsed, val, is_maximize=None)
                    seen.add((elapsed, val))
                # record the final value as a bound if not already recorded
                final_val = self.get_obj_value(picked_sol)
                if (ts, final_val) not in seen:
                    sub_obj_store.add_obj_bound(ts, final_val, is_maximize=None)
                sub_obj_store.add_last_timestamp_note(
                    note, obj_value_is_valid=True, obj_bound_is_valid=True
                )
            else:
                # if iter_report is None or infeasible, just log the final value as a bound
                picked_val = self.get_obj_value(picked_sol)
                sub_obj_store.add_obj_bound(ts, picked_val, is_maximize=None)
                sub_obj_store.add_last_timestamp_note(
                    note, obj_value_is_valid=True, obj_bound_is_valid=True
                )

        # ================== /helpers ==================

        halt_incremental_cp_processing = False
        job_subset: set[str] = set()

        for bidx, job_sublist in enumerate(sequence_of_job_sublist):
            # ---------- [Time over?] : cutoff before partial dispatch ----------
            _timelimit = self.get_remaining_time_limit(max_time_per_add)
            if float_a_leq_b(_timelimit, 0):
                logging.info(
                    "(batch %d/%d) Time over before dispatch -> finish by dispatching remaining jobs.",
                    bidx + 1,
                    job_sublist_cnt,
                )
                halt_incremental_cp_processing = True
                break  # End loop & go to 'after-loop finishing'

            # ---------- [Partial dispatch] ----------
            partial_sol = (
                FlowshopSchedule.from_stage_name_list(self.stage_ids)
                if last_solution is None
                else last_solution.deepcopy()
            )
            for j in job_sublist:
                partial_sol.dispatch_job_by_stages(
                    j,
                    self.stage_ids,
                    self.job_2_stage_2_p_dict[j],
                    after_last=True,
                )
            job_subset.update(job_sublist)
            dispatch_obj_val = self.get_obj_value(partial_sol)

            # ---------- [sumTj == 0 ?] ----------
            if dispatch_obj_val == 0:
                logging.info(
                    "(batch %d/%d) sumTj=0 -> skip CP; keep partial dispatched schedule.",
                    bidx + 1,
                    job_sublist_cnt,
                )
                last_solution = partial_sol
                _log_snapshot(
                    last_solution,
                    job_subset,
                    note=f"{len(job_subset)}/{job_cnt} (sumTj=0; skip CP)",
                )
                continue

            # ---------- [Sub CP build & solve] ----------
            sub_cp_mdl = self.cp_model.create_problem_of_job_subset(job_subset)
            if last_solution is not None:
                sub_cp_mdl.add_indirect_precedence_constraints_by_sequence(
                    last_solution.get_last_stage_job_list()
                )
            sub_cp_mdl.add_hints_from_schedule(partial_sol)
            # set obj lower bound if all jobs are included and we have a valid bound
            all_jobs_are_included = len(job_subset) == job_cnt
            if (
                all_jobs_are_included
                and self.solution_manager.best_obj_bound is not None
                and not math.isnan(self.solution_manager.best_obj_bound)
            ):
                sub_cp_mdl.set_obj_lower_bound(self.solution_manager.best_obj_bound)

            # ---------- [Time over?] : cutoff before CP solving ----------
            _timelimit = self.get_remaining_time_limit(max_time_per_add)
            if float_a_leq_b(_timelimit, 0):
                logging.info(
                    "(batch %d/%d) Time over before CP solving -> finish by dispatching remaining jobs.",
                )
                last_solution = partial_sol  # keep the dispatched partial schedule
                halt_incremental_cp_processing = True
                break  # End loop & go to 'after-loop finishing'

            logging.info(
                "(batch %d/%d) Starting CP on subproblem with %d jobs (dispatched obj val: %d) at %s",
                bidx + 1,
                job_sublist_cnt,
                len(job_subset),
                dispatch_obj_val,
                sub_timer.get_formatted_elapsed_time(),
            )
            iter_report = self.solve_cp_model(
                sub_cp_mdl,
                _timelimit,
                solver_thread_cnt,
                random_seed=self.random_seed,
                no_improvement_timelimit=no_improvement_timelimit,
                cp_model_presolve=self.cp_model_presolve,
                e_timer=sub_timer,
                log_level_obj_value=logging.INFO,
                log_level_obj_bound=logging.INFO,
                log_level_solver=self.log_solver_level,
                log_search_progress=self.log_search_progress,
                obj_value_is_valid=all_jobs_are_included,
            )
            last_timestamp = sub_timer.elapsed_sec

            # ---------- [CP feasible? & CP is better?] ----------
            # Use sequence-based schedule creation since the starting times by CP model
            # does not minimize the total completion time (its main goal is to minimize
            # tardiness).
            cp_sched = (
                sub_cp_mdl.create_schedule_from_sequence()
                if iter_report.is_feasible
                else None
            )
            cp_obj_val = None if cp_sched is None else self.get_obj_value(cp_sched)
            cp_is_better = cp_obj_val is not None and float_a_stl_b(
                cp_obj_val, dispatch_obj_val
            )

            if cp_is_better:
                assert cp_obj_val is not None
                logging.info(
                    "CP is better (%d < %d) -> use CP schedule.",
                    cp_obj_val,
                    dispatch_obj_val,
                )
                assert cp_sched is not None
                last_solution = cp_sched
            else:
                reason = (
                    "infeasible" if not iter_report.is_feasible else "no improvement"
                )
                logging.info("CP %s -> keep dispatched partial.", reason)
                last_solution = partial_sol

            # ---------- [CP is not better & timeover?] ----------
            if (not cp_is_better) and float_a_leq_b(
                _timelimit, iter_report.elapsed_time
            ):
                logging.warning(
                    "CP timeover at this batch -> halt further incremental CP."
                )
                halt_incremental_cp_processing = True

            # ---------- [Log snapshot & halt] ----------
            _log_snapshot(
                last_solution,
                job_subset,
                note=f"{len(job_subset)}/{job_cnt}",
                iter_report=iter_report,
                timestamp=last_timestamp,
            )
            if halt_incremental_cp_processing:
                break

        # ---------- [End] : Dispatch remaining jobs ----------
        if last_solution is None:
            last_solution = FlowshopSchedule.from_stage_name_list(self.stage_ids)

        remaining_jobs = [j for j in job_sequence if j not in job_subset]
        if remaining_jobs:
            logging.info("Dispatch remaining %d jobs and finish.", len(remaining_jobs))
            for j in remaining_jobs:
                last_solution.dispatch_job_by_stages(
                    j,
                    self.stage_ids,
                    self.job_2_stage_2_p_dict[j],
                    after_last=True,
                )
            _log_snapshot(
                last_solution,
                set(job_sequence),
                note="Final dispatch",
                timestamp=sub_timer.elapsed_sec,
            )

        if error_if_infeasible:
            self.check_feasibility(last_solution)
        last_solution_obj_value = self.get_obj_value(last_solution)
        logging.info(
            f"Initialized by IC with total tardiness {last_solution_obj_value}"
        )
        # Create report for the final solution and register it
        final_report = FsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=float(last_solution_obj_value),
            obj_bound=None,
            is_init=is_init,
        )
        was_updated = self.solution_manager.register(final_report, last_solution)

        if was_updated:
            log_time = self.timer.elapsed_sec
            self.add_obj_value_log(log_time, last_solution_obj_value, is_maximize=None)
            _last_timestamp_note = self._get_call_context_of_current_method()
            self.obj_store.add_last_timestamp_note(
                _last_timestamp_note, obj_value_is_valid=True
            )
            if draw_gantt:
                self.draw_incumbent_gantt()

        # Write the objective store to a YAML file
        # TODO: suffix from output_metadata
        if sub_obj_store:
            sub_obj_store.save_yaml(self.get_file_path_for_subroutine("_obj_log.yaml"))

    # Subroutine: job block neighbor search

    def job_block_ns(
        self,
        rho: float,
        computational_time: float,
        solver_thread_cnt: int,
        no_improvement_timelimit: float | None = None,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        self.fix_profile_solve_reset(
            lambda: self.apply_job_block_operator(rho),
            computational_time,
            solver_thread_cnt,
            no_improvement_timelimit=no_improvement_timelimit,
            obj_value_is_valid=True,
            obj_bound_is_valid=False,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def apply_job_block_operator(
        self, rho: float, randomize_selection: bool = True
    ) -> None:
        if rho <= 0:
            raise ValueError(f"Invalid value for rho {rho}; it must be positive.")
        if rho > 1:
            raise ValueError(f"Invalid value for rho {rho}; it must be at most 1.")
        logging.info(f"Applying job block operator with rho={rho}")
        if not self.solution_manager.has_incumbent():
            raise ValueError("No incumbent solution available for job block operator.")
        incumbent_solution = self.solution_manager.get_incumbent()
        if incumbent_solution is None:
            raise ValueError("No incumbent solution to select job block.")

        job_sequence = incumbent_solution.get_last_stage_job_list()
        sched_job_cnt = len(job_sequence)
        if sched_job_cnt == 0:
            raise ValueError("Incumbent solution has no scheduled jobs.")
        logging.info(f"Current job sequence: {job_sequence}")
        num_to_select = max(1, int(math.ceil(rho * sched_job_cnt)))

        # Choose an operation
        if randomize_selection:
            seed_job = random.choice(job_sequence)
        else:
            # if not random, choose center operation
            seed_job = job_sequence[sched_job_cnt // 2]
        # Select (num_to_select - 1) more jobs; if reached the end, continue from the first element
        selected_jobs: list[str] = [seed_job]
        seed_job_idx = job_sequence.index(seed_job)
        logging.info(
            "Seed job for block operator: %s (index %d)", seed_job, seed_job_idx
        )
        job_cnt_until_last = len(job_sequence) - seed_job_idx
        logging.info(
            "Jobs from seed to end: %d, need to select total %d jobs.",
            job_cnt_until_last,
            num_to_select,
        )

        # Select a contiguous block of num_to_select jobs, wrapping around if needed
        selected_jobs.extend(
            job_sequence[(seed_job_idx + 1 + i) % sched_job_cnt]
            for i in range(num_to_select - 1)
        )
        logging.info(
            f"job block operator selected {len(selected_jobs)} jobs"
            f" (target={num_to_select})"
        )
        logging.info(f"Selected jobs: {selected_jobs}")

        # Profile out-of-block operations
        self._fix_job_profile_except_selected(set(selected_jobs))

    # Subroutine: lower bound by preemptive scheduling of the last stage only

    def compute_preemptive_last_stage_lb(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        from ..graph_model.single_mc_pmtn import SingleMachinePreemptionMcf

        sub_timer = ElapsedTimer()
        last_stage_only_mdl = SingleMachinePreemptionMcf.from_instance(self.instance)
        last_stage_only_mdl.solve()

        if last_stage_only_mdl.is_optimal():
            obj_bound = last_stage_only_mdl.get_obj_value()
            logging.info(
                "Preemptive last-stage-only model solved optimally with objective value %d; took %s",
                obj_bound,
                sub_timer.get_formatted_elapsed_time(),
            )
            seq_by_start = last_stage_only_mdl.get_job_start_sequence()
            schedule_by_start = self.get_dispatched_schedule(seq_by_start)
            obj_value_by_start = self.get_obj_value(schedule_by_start)

            seq_by_end = last_stage_only_mdl.get_job_completion_sequence()
            schedule_by_end = self.get_dispatched_schedule(seq_by_end)
            obj_value_by_end = self.get_obj_value(schedule_by_end)

            seq_by_avg = last_stage_only_mdl.get_job_average_sequence()
            schedule_by_avg = self.get_dispatched_schedule(seq_by_avg)
            obj_value_by_avg = self.get_obj_value(schedule_by_avg)

            logging.info("Dispatched schedules' total tardiness:")
            logging.info(" - by start time sequence: %d", obj_value_by_start)
            logging.info(" - by completion time sequence: %d", obj_value_by_end)
            logging.info(" - by average time sequence: %d", obj_value_by_avg)
            # Choose the best among the three dispatched sequences
            best_obj_value = min(obj_value_by_start, obj_value_by_end, obj_value_by_avg)
            if best_obj_value == obj_value_by_start:
                best_schedule = schedule_by_start
                method_used = "start time"
            elif best_obj_value == obj_value_by_end:
                best_schedule = schedule_by_end
                method_used = "completion time"
            else:
                best_schedule = schedule_by_avg
                method_used = "average time"
            logging.info(
                "Among dispatched schedules, best total tardiness is %d by %s sequence.",
                best_obj_value,
                method_used,
            )

            # Create report and register the new solution
            report = FsSubroutineReport(
                elapsed_time=sub_timer.elapsed_sec,
                obj_value=best_obj_value,
                obj_bound=obj_bound,
                is_init=True,
            )
            was_updated = self.solution_manager.register(report, best_schedule)

            # Log
            log_time = self.timer.elapsed_sec
            self.add_obj_value_log(log_time, best_obj_value, is_maximize=False)
            self.add_obj_bound_log(log_time, obj_bound, is_maximize=False)
            _last_timestamp_note = self._get_call_context_of_current_method()
            self.obj_store.add_last_timestamp_note(
                _last_timestamp_note, obj_value_is_valid=True, obj_bound_is_valid=True
            )
            # Draw Gantt chart if the solution is an improvement
            if was_updated and draw_gantt:
                self.draw_incumbent_gantt()
