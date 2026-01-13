import logging
import math
import random
import time
from collections import defaultdict
from typing import Callable, Sequence

from routix import DynamicDataObject, ElapsedTimer
from routix.util.comparison import float_a_stl_b
from schore.schedule_examples.shop.flow import FlowshopSchedule

from ..report import FsSubroutineReport
from .controller_core import FlowshopTardinessControllerCore
from .flowshop_batch_eval import PermutationFlowshopSubseqEvaluator
from .list_window_slider import window_slide_over_list
from .schedule_metric import ScheduleMetric

REL_TOL = 1e-9  # for safe float comparisons


class FlowshopTardinessCpLnsController(FlowshopTardinessControllerCore):
    cp_model_presolve: bool | None = None  # TODO: make it configurable
    """
    Whether to presolve the CP model before solving.
    If None, use the default behavior of the CP solver.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._profile_timing_enabled: bool = False  # change to True to enable timing
        self._profile_timing_stats: defaultdict[str, int | float] = defaultdict(float)
        self._profile_timing_counts: defaultdict[str, int] = defaultdict(int)

    def log_insertion_timing_summary_as_info(self):
        if not getattr(self, "_profile_timing_stats", None):
            return
        stats = self._profile_timing_stats
        counts = self._profile_timing_counts
        log_str = "\n==== Insertion Timing Summary ===="
        keys = sorted(stats.keys(), key=lambda k: stats[k], reverse=True)
        for k in keys:
            c = counts.get(k, 0)
            tot = stats[k]
            avg = tot / c if c else 0.0
            log_str += f"\n{k:25s} total={tot:10.6f}s  cnt={c:6d}  avg={avg:10.6f}s"
        log_str += "\n=================================="
        logging.info(log_str)

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
        # from google.protobuf.json_format import MessageToDict
        # from routix.io import object_to_yaml

        # model_file_suffix = f"_model_{self.instance.name}"
        # model_file_path = self.get_file_path_for_subroutine(model_file_suffix + ".yaml")

        # # Export JSON representation as yaml
        # proto_dict = MessageToDict(
        #     self.cp_model.Proto(), preserving_proto_field_name=True
        # )
        # object_to_yaml(proto_dict, model_file_path)

        # # Export in LP format
        # self.cp_model.export_to_file(
        #     self.get_file_path_for_subroutine(model_file_suffix + ".lp").as_posix()
        # )

        # # Export in MPS format
        # self.cp_model.export_to_file(
        #     self.get_file_path_for_subroutine(model_file_suffix + ".mps").as_posix()
        # )
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
        incumbent_schedule = self.solution_manager.get_incumbent()
        if isinstance(incumbent_schedule, FlowshopSchedule):
            obj_value = self.get_obj_value(incumbent_schedule)

            log_time = self.timer.elapsed_sec
            last_obj_value = self.obj_store.get_last_obj_value()
            best_obj_value = (
                obj_value
                if last_obj_value is None or obj_value < last_obj_value
                else last_obj_value
            )
            self.add_obj_value_log(log_time, best_obj_value, is_maximize=None)
            _last_timestamp_note = self._get_call_context_of_current_method()
            self.obj_store.add_last_timestamp_note(
                _last_timestamp_note, obj_value_is_valid=True
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
        last_obj_value = self.obj_store.get_last_obj_value()
        best_obj_value = (
            obj_value
            if last_obj_value is None or obj_value < last_obj_value
            else last_obj_value
        )
        self.add_obj_value_log(log_time, best_obj_value, is_maximize=None)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )
        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.export_incumbent_to_yaml()

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
        dmap: dict[str, int] = self.instance.job_2_duedate_map
        pmap: dict[str, dict[str, int]] = self.job_2_stage_2_p_dict

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

        i_list: tuple[str, ...] = self.stage_ids
        dmap: dict[str, int] = self.instance.job_2_duedate_map
        pmap: dict[str, dict[str, int]] = self.job_2_stage_2_p_dict

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
        last_obj_value = self.obj_store.get_last_obj_value()
        best_obj_value = (
            obj_value
            if last_obj_value is None or obj_value < last_obj_value
            else last_obj_value
        )
        self.add_obj_value_log(log_time, best_obj_value, is_maximize=None)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )
        # Draw Gantt chart if the solution is an improvement
        if was_updated and draw_gantt:
            self.export_incumbent_to_yaml()

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
        return_val: int = self.instance.job_2_duedate_map[job_id]
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
        dmap: dict[str, int] = self.instance.job_2_duedate_map
        pos_2_stage_2_endtime_map: dict[int, dict[str, int]] = {
            0: {i: 0 for i in self.stage_ids}
        }

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

    def _compute_schedule_metric_from_sequence(
        self, job_seq: list[str]
    ) -> ScheduleMetric:
        """Compute total tardiness and makespan from a job sequence.

        Args:
            job_seq (list[str]): Ordered list of job ids to evaluate.

        Returns:
            ScheduleMetric: computed schedule metrics.
        """
        prefix_frontiers, prefix_sumTj = self._compute_prefix_frontiers_and_sumTj(
            job_seq
        )
        n = len(job_seq)
        return ScheduleMetric(
            prefix_sumTj[n],
            [prefix_frontiers[n][i] for i in self.stage_ids],
            {
                (i, j): self.stage_job_2_p_dict[i, j]
                for j in job_seq
                for i in self.stage_ids
            },
        )

    @staticmethod
    def _tie_crit_from_tm(
        metric: ScheduleMetric, tie_breaker: str, makespan_multiplier: float = 1.0
    ) -> tuple[float, float]:
        """Calculate tie breaking criteria from tie breaker name.

        Args:
            metric (ScheduleMetric): schedule metrics
            tie_breaker (str): tie breaker name
            makespan_multiplier (float, optional): multiplier for Cmax in some tie breakers.
                Defaults to 1.0.

        Raises:
            ValueError: if tie_breaker is unknown

        Returns:
            tuple[float, float]: (crit1, crit2) for comparison
        """
        if tie_breaker == "default":
            return (metric.sumTj, 0)
        if tie_breaker == "makespan":
            return (metric.sumTj, metric.makespan)
        if tie_breaker == "NEH-M":
            return (metric.sumTj + metric.makespan * makespan_multiplier, 0)
        if tie_breaker == "NEH-IT1":
            return (metric.sumTj, metric.get_total_idle_time())
        raise ValueError(f"Unknown tie_breaker: {tie_breaker}")

    def _eval_insert_with_criteria(
        self,
        seq_now: list[str],
        job_id: str,
        tie_breaker: str = "default",
        first_improvement: bool = False,
        baseline_metric: ScheduleMetric | None = None,
        makespan_multiplier: float = 1.0,
    ) -> tuple[int, ScheduleMetric]:
        """Evaluate insertion of job_id into seq_now.

        Args:
            seq_now (list[str]): Current job sequence.
            job_id (str): Job ID to insert.
            tie_breaker (str, optional): Tie breaking strategy.
                Defaults to "default".
            first_improvement (bool, optional): If True, stop at first improvement.
                Defaults to False.
            baseline_metric (ScheduleMetrics | None, optional): Baseline metrics for comparison.
                Required if first_improvement is True. Defaults to None.
            makespan_multiplier (float, optional): Multiplier for makespan in some tie breakers.
                Defaults to 1.0.

        Returns:
            tuple[int, ScheduleMetrics]: (best position, best schedule metrics).
        """
        dmap: dict[str, int] = self.instance.job_2_duedate_map

        if not seq_now:
            # only one position
            f0 = {i: 0 for i in self.stage_ids}
            new_f = self._simulate_append(f0, job_id)
            Cmax = new_f[self.last_stage_id]
            sumTj = Cmax - dmap[job_id]
            if sumTj < 0:
                sumTj = 0
            return 0, ScheduleMetric(
                sumTj,
                [new_f[i] for i in self.stage_ids],
                {
                    (i, job_id): self.stage_job_2_p_dict[i, job_id]
                    for i in self.stage_ids
                },
            )

        prefix_frontiers, prefix_tardy = self._compute_prefix_frontiers_and_sumTj(
            seq_now
        )
        best_pos = 0
        best_metric: ScheduleMetric | None = None
        best_crit1: float | None = None
        best_crit2: float | None = None

        baseline_crit1: float | None
        baseline_crit2: float | None
        if first_improvement:
            if baseline_metric is not None:
                baseline_crit1, baseline_crit2 = self._tie_crit_from_tm(
                    baseline_metric,
                    tie_breaker,
                    makespan_multiplier=makespan_multiplier,
                )
            else:
                raise ValueError(
                    "baseline_crit must be provided when first_improvement is True."
                )
        else:
            baseline_crit1, baseline_crit2 = None, None

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

            new_Cmax_list = [new_frontier[i] for i in self.stage_ids]
            new_metric = ScheduleMetric(
                new_sumTj,
                new_Cmax_list,
                {
                    (i, j): self.stage_job_2_p_dict[i, j]
                    for j in seq_now + [job_id]
                    for i in self.stage_ids
                },
            )
            crit1, crit2 = self._tie_crit_from_tm(
                new_metric, tie_breaker, makespan_multiplier=makespan_multiplier
            )
            # logging.info(f"Position {pos}: {crit1}, {crit2}")

            # Early exit for first-improvement policy
            if (
                first_improvement
                and baseline_crit1 is not None
                and baseline_crit2 is not None
            ):
                if (crit1 < baseline_crit1) or (
                    crit1 == baseline_crit1 and crit2 < baseline_crit2
                ):
                    return pos, new_metric

            # choose best
            if (best_crit1 is None) or (crit1 < best_crit1):
                best_pos, best_metric = pos, new_metric
                best_crit1, best_crit2 = crit1, crit2
            elif (crit1 == best_crit1) and (best_crit2 is None or crit2 < best_crit2):
                best_pos, best_metric = pos, new_metric
                best_crit1, best_crit2 = crit1, crit2
                # if still tied, earlier position is preferred (stable)

        if best_metric is None:
            raise RuntimeError("Unexpected: best_metric is None after evaluation.")
        return best_pos, best_metric

    # -----------------------------
    # NEW acceleration helper: build evaluator with index-mapped p, due
    # -----------------------------
    def _get_new_acc_evaluator(self):
        """
        Build (and cache) a PermutationFlowshopSubseqEvaluator that uses 0-based
        integer job indices internally.

        Returns:
            (evaluator, job_id_to_idx, idx_to_job_id)
        """
        # cache on controller instance to avoid rebuilding on every call
        if hasattr(self, "_new_acc_cache") and self._new_acc_cache is not None:
            return self._new_acc_cache

        job_ids: list[str] = list(self.instance.job_id_list)
        stage_ids: list[str] = list(self.stage_ids)

        job_id_to_idx: dict[str, int] = {jid: k for k, jid in enumerate(job_ids)}
        idx_to_job_id: dict[int, str] = {k: jid for jid, k in job_id_to_idx.items()}

        m = len(stage_ids)
        n = len(job_ids)

        # Build p[m][n] aligned with (stage_idx, job_idx)
        p = [[0] * n for _ in range(m)]
        for i, stage_id in enumerate(stage_ids):
            for jid in job_ids:
                j = job_id_to_idx[jid]
                p[i][j] = int(self.stage_2_job_2_p_dict[stage_id][jid])

        # Build due[n]
        due = [0] * n
        for jid in job_ids:
            j = job_id_to_idx[jid]
            due[j] = int(self.instance.job_2_duedate_map[jid])

        evaluator = PermutationFlowshopSubseqEvaluator(p, due)

        self._new_acc_cache: tuple[
            PermutationFlowshopSubseqEvaluator, dict[str, int], dict[int, str]
        ] = (evaluator, job_id_to_idx, idx_to_job_id)
        return self._new_acc_cache

    def _get_best_pos_and_metric_new_acc(
        self,
        seq_now: list[str],
        job_id_seq: Sequence[str] | str,
        tie_breaker: str = "default",
    ) -> tuple[int, ScheduleMetric]:
        """
        Evaluate insertion of job_id into seq_now using NEW acceleration
        (Fernandez-Viagas et al., 2020) evaluator.

        Args:
            seq_now: current sequence of job IDs (strings)
            job_id_seq: job ID (string) or sequence of job IDs to insert
            tie_breaker: tie breaking strategy

        Returns:
            (best_pos, ScheduleMetric) for the best insertion position.
        """
        _job_id_seq: Sequence[str]
        if isinstance(job_id_seq, str):
            _job_id_seq = [job_id_seq]
        else:
            _job_id_seq = job_id_seq

        evaluator, job_id_to_idx, _ = self._get_new_acc_evaluator()

        # convert current sequence to index sequence
        pi_idx = [job_id_to_idx[j] for j in seq_now]
        sigma_idx_seq = [job_id_to_idx[job_id] for job_id in _job_id_seq]

        # NEW evaluator returns best position and best objective1 value (sumTj)
        best_pos, _ = evaluator.get_best_position(
            pi_idx, sigma_idx_seq, tie_breaker=tie_breaker
        )

        # Build resulting sequence and compute ScheduleMetric using existing method
        new_seq = seq_now[:best_pos] + list(_job_id_seq) + seq_now[best_pos:]
        metric = self._compute_schedule_metric_from_sequence(new_seq)

        return best_pos, metric

    def _run_neh_edd(
        self,
        method_name: str,
        tie_breaker: str,
        first_improvement: bool = False,
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
        incumbent_sol = self.solution_manager.get_incumbent()
        if incumbent_sol is None:
            job_sequence = self.get_edd_sequence()
        else:
            job_sequence = incumbent_sol.get_last_stage_job_list()

        seq: list[str] = []

        # 2) NEH insertion by EDD order with NEW acceleration evaluation
        for j in job_sequence:
            pos, _ = self._get_best_pos_and_metric_new_acc(
                seq, j, tie_breaker=tie_breaker
            )
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
        last_obj_value = self.obj_store.get_last_obj_value()
        best_obj_value = (
            obj_value
            if last_obj_value is None or obj_value < last_obj_value
            else last_obj_value
        )
        self.add_obj_value_log(log_time, best_obj_value, is_maximize=None)
        _last_timestamp_note = self._get_call_context_of_current_method()
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note, obj_value_is_valid=True
        )

        if was_updated and draw_gantt:
            self.export_incumbent_to_yaml()

    def initialize_by_nehedd(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        self._run_neh_edd(
            "NEHedd",
            "default",
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def initialize_by_nehms(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        self._run_neh_edd(
            "makespan",
            "makespan",
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def initialize_by_nehm(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        self._run_neh_edd(
            "NEH-M",
            "NEH-M",
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def initialize_by_neh_it1(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        self._run_neh_edd(
            "NEH-IT1",
            "NEH-IT1",
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    def improve_job_seq_by_insertion_single_pass(
        self,
        job_seq: list[str],
        subseq_size: int | None = None,
        tie_breaker: str = "default",
        first_improvement: bool = False,
    ) -> list[str]:
        """Perform a single pass of insertion-based improvement on the job sequence.

        Args:
            job_seq (list[str]): The current sequence of jobs.
            subseq_size (int | None, optional): Size of subsequences to consider for insertion.
                Defaults to None.
            tie_breaker (str, optional): Criteria for breaking ties when choosing positions.
                Defaults to "default".
            first_improvement (bool, optional): Whether to stop at the first improvement found.
                Defaults to False.
                If subseq_size > 1, this is forced to True.

        Raises:
            ValueError: If job_seq length <= 1.
            ValueError: If job_seq contains duplicate job IDs.
            ValueError: If subseq_size is less than 1.

        Returns:
            list[str]: The improved job sequence after the insertion pass.
        """
        job_cnt = len(job_seq)
        if job_cnt <= 1:
            logging.info("Insertion improvement skipped: only one or zero jobs.")
            return job_seq
        # Quick sanity: unique IDs
        if len(set(job_seq)) != job_cnt:
            raise ValueError("job_seq contains duplicate job IDs.")

        _first_improvement = first_improvement

        if subseq_size is None or subseq_size == 1:
            _subseq_size = 1
        elif subseq_size < 1:
            raise ValueError("subseq_size must be at least 1.")
        else:
            _subseq_size = subseq_size
            if first_improvement is False:
                logging.warning(
                    "first_improvement=False was requested but is not supported when "
                    "subseq_size > 1; overriding first_improvement to True."
                )
                _first_improvement = (
                    True  # always first-improvement for subsequences > 1
                )

        if _subseq_size > job_cnt:
            raise ValueError(
                f"subseq_size ({_subseq_size}) cannot be greater than the number of jobs ({job_cnt})."
            )

        timing_enabled: bool = getattr(self, "_profile_timing_enabled", False)
        if timing_enabled:
            stats = self._profile_timing_stats
            counts = self._profile_timing_counts

        seq_before = list(job_seq)
        incumbent_seq = list(seq_before)

        best_metric = self._compute_schedule_metric_from_sequence(seq_before)
        best_crit1, best_crit2 = self._tie_crit_from_tm(best_metric, tie_breaker)

        max_iter_cnt = job_cnt - _subseq_size + 1
        logging.info(
            f"Starting insertion improvement pass (subseq_size={_subseq_size}, max iterations={max_iter_cnt})."
        )
        iter_cnt = 0
        for j_subseq in window_slide_over_list(job_seq, _subseq_size):
            if timing_enabled:
                # overall iteration timer
                t_iter0 = time.perf_counter()
                # A) profile_fixed creation timer
                t0 = time.perf_counter()

            iter_cnt += 1
            profile_fixed: list[str] = [
                j for j in incumbent_seq if j not in j_subseq
            ]  # remove subseq
            if timing_enabled:
                stats["A_profile_fixed"] += time.perf_counter() - t0
                counts["A_profile_fixed"] += 1

            # B) get best position timer
            if timing_enabled:
                t0 = time.perf_counter()
            pos, after_metric = self._get_best_pos_and_metric_new_acc(
                profile_fixed, j_subseq, tie_breaker=tie_breaker
            )
            if timing_enabled:
                stats["B_get_best_pos_metric"] += time.perf_counter() - t0
                counts["B_get_best_pos_metric"] += 1

            # C) tie-criteria calc timer
            if timing_enabled:
                t0 = time.perf_counter()
            crit1, crit2 = self._tie_crit_from_tm(after_metric, tie_breaker)
            after_is_better = (crit1 < best_crit1) or (
                crit1 == best_crit1 and crit2 < best_crit2
            )
            if timing_enabled:
                stats["C_tie_compare"] += time.perf_counter() - t0
                counts["C_tie_compare"] += 1

            # D) incumbent sequence update timer
            if after_is_better:
                if timing_enabled:
                    t0 = time.perf_counter()

                incumbent_seq = (
                    profile_fixed[:pos] + list(j_subseq) + profile_fixed[pos:]
                )
                best_metric = after_metric
                best_crit1 = crit1
                best_crit2 = crit2

                if timing_enabled:
                    stats["D_apply_update"] += time.perf_counter() - t0
                    counts["D_apply_update"] += 1

                if _first_improvement:
                    logging.info(
                        f"Insertion improvement pass: iteration {iter_cnt} / {max_iter_cnt}, found improvement to total tardiness {best_metric.sumTj}."
                    )
                    break  # exit after 1st improvement

            if timing_enabled:
                stats["ITER_total"] += time.perf_counter() - t_iter0
                counts["ITER_total"] += 1
            if self.time_is_up():
                logging.info(
                    f"Time limit reached during {iter_cnt} / {max_iter_cnt} insertion improvement."
                )
                break

        return incumbent_seq

    def improve_by_insertion(
        self,
        subseq_size: int | None = None,
        tie_breaker: str = "default",
        max_passes: int | None = None,
        first_improvement: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ) -> None:
        """Repeated insertion-improvement passes on the incumbent sequence.

        Args:
            tie_breaker (str, optional): tie-breaking rule.
                Defaults to "default".
            max_passes (int | None, optional): Maximum number of passes. If None, unlimited passes are allowed.
                Defaults to None.
            first_improvement (bool, optional): Whether to use first-improvement strategy.
                Defaults to False.
            error_if_infeasible (bool, optional): Whether to raise an error if the solution is infeasible.
                Defaults to False.
            draw_gantt (bool, optional): Whether to draw Gantt chart.
                Defaults to False.

        Raises:
            RuntimeError: If no incumbent solution is available for improvement.
        """
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
        best_metric = self._compute_schedule_metric_from_sequence(seq_before)
        best_crit1, best_crit2 = self._tie_crit_from_tm(best_metric, tie_breaker)
        logging.info(
            f"Initial: total tardiness {best_metric.sumTj}, makespan {best_metric.makespan} (criteria: {best_crit1}, {best_crit2})."
        )

        improved_globally = False
        passes = 0
        # list of (global elapsed time, obj value)
        obj_value_log: list[tuple[float, float]] = []
        while max_passes is None or passes < max_passes:
            passes += 1

            seq_after = self.improve_job_seq_by_insertion_single_pass(
                seq_before,
                subseq_size=subseq_size,
                tie_breaker=tie_breaker,
                first_improvement=first_improvement,
            )
            after_metric = self._compute_schedule_metric_from_sequence(seq_after)
            crit1, crit2 = self._tie_crit_from_tm(after_metric, tie_breaker)
            logging.info(
                f"Pass {passes}: total tardiness {after_metric.sumTj}, makespan {after_metric.makespan} (criteria: {crit1}, {crit2})."
            )
            after_is_better = (crit1 < best_crit1) or (
                crit1 == best_crit1 and crit2 < best_crit2
            )
            if after_is_better:
                seq_before = list(seq_after)
                best_metric = after_metric
                best_crit1 = crit1
                best_crit2 = crit2
                logging.info(
                    f"Pass {passes}: improved to total tardiness {best_metric.sumTj}."
                )
                improved_globally = True
                obj_value_log.append((self.timer.elapsed_sec, best_metric.sumTj))
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
        timing_enabled = getattr(self, "_profile_timing_enabled", False)
        if timing_enabled:
            self.log_insertion_timing_summary_as_info()
            cache = getattr(self, "_new_acc_cache", None)
            if cache:
                first_evaluator = cache[0]
                if first_evaluator is not None:
                    first_evaluator.log_timing_as_info()

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
                self.export_incumbent_to_yaml()

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
            last_obj_value = self.obj_store.get_last_obj_value()
            best_obj_value = (
                best_obj_value
                if last_obj_value is None or best_obj_value < last_obj_value
                else last_obj_value
            )
            self.add_obj_value_log(log_time, best_obj_value, is_maximize=None)
            last_obj_bound = self.obj_store.get_last_obj_bound()
            best_obj_bound = (
                obj_bound
                if last_obj_bound is None or obj_bound > last_obj_bound
                else last_obj_bound
            )
            self.add_obj_bound_log(log_time, best_obj_bound, is_maximize=None)
            _last_timestamp_note = self._get_call_context_of_current_method()
            self.obj_store.add_last_timestamp_note(
                _last_timestamp_note, obj_value_is_valid=True, obj_bound_is_valid=True
            )
            # Draw Gantt chart if the solution is an improvement
            if was_updated and draw_gantt:
                self.export_incumbent_to_yaml()

    def pw_cp(
        self,
        solver_thread_cnt: int,
        added_batch_size: int = 1,
        max_time_per_add: float | None = None,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ):
        """
        Builds a CP-guided solution using job sequence of the incumbent solution.

        Args:
            solver_thread_cnt (int): The number of parallel workers (i.e. threads) to use during search.
            added_batch_size (int, optional): The number of jobs to add in each iteration.
                Defaults to 1.
            max_time_per_add (float | None, optional): Time limit (in seconds) for solving each incremental subproblem.
                If None, uses the remaining time limit. Defaults to None.
            error_if_infeasible (bool, optional): If True, raises an error if the solution is infeasible.
                Defaults to False.
            draw_gantt (bool, optional): If True, draws a Gantt chart of the solution.
                Defaults to False.
        """
        sub_timer = ElapsedTimer()

        # Job sequence from the incumbent solution
        incumbent_sol = self.solution_manager.get_incumbent()
        if incumbent_sol is None:
            # use EDD order if no incumbent
            job_sequence = self.get_edd_sequence()
        else:
            job_sequence = incumbent_sol.get_last_stage_job_list()

        # Draw Gantt chart of the initial solution if requested
        if draw_gantt:
            output_path = self.get_file_path_for_subroutine("_0_init_solution.yaml")
            self.export_incumbent_to_yaml(output_path=output_path)

        from .pw_cp import PwCpConstructor, PwCpResult

        constructor = PwCpConstructor(self)
        result: PwCpResult = constructor.run(
            job_sequence,
            added_batch_size=added_batch_size,
            solver_thread_cnt=solver_thread_cnt,
            max_time_per_add=max_time_per_add,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )
        obj_value = self.get_obj_value(result.schedule)
        logging.info(f"PW-CP done with total tardiness {obj_value}")
        # Create report for the final solution and register it
        final_report = FsSubroutineReport(
            elapsed_time=sub_timer.elapsed_sec,
            obj_value=float(obj_value),
            obj_bound=None,
            is_init=False,
        )
        was_updated: bool = self.solution_manager.register(
            final_report, result.schedule
        )

        if was_updated:
            log_time = self.timer.elapsed_sec
            last_obj_value = self.obj_store.get_last_obj_value()
            best_obj_value = (
                obj_value
                if last_obj_value is None or obj_value < last_obj_value
                else last_obj_value
            )
            self.add_obj_value_log(log_time, best_obj_value, is_maximize=None)
            self.add_obj_value_log(log_time, obj_value, is_maximize=None)
            _last_timestamp_note = self._get_call_context_of_current_method()
            self.obj_store.add_last_timestamp_note(
                _last_timestamp_note, obj_value_is_valid=True
            )
            if draw_gantt:
                self.export_incumbent_to_yaml()

        # Write the objective store to a YAML file
        # TODO: suffix from output_metadata
        result.sub_obj_store.save_yaml(
            self.get_file_path_for_subroutine("_obj_log.yaml")
        )

    def repeat_while_improvement(self, n_repeats: int, routine_data: DynamicDataObject):
        """
        Repeats the execution of a routine a specified number of times.

        Args:
            n_repeats (int): Number of times to repeat the routine.
            routine_data (DynamicDataObject): The routine data to be executed.
        """

        subroutine_name = "reps"  # TODO: define how to manage this
        incumbent_sol = self.solution_manager.get_incumbent()
        if incumbent_sol is None:
            obj_before = math.inf
        else:
            obj_before = self.get_obj_value(incumbent_sol)

        for i in range(n_repeats):
            if self.is_stopping_condition():
                logging.info(
                    f"[Repeat] Stopping condition met at iteration {i + 1}/{n_repeats}."
                )
                break
            logging.info(f"[Repeat] Starting repeat {i + 1}/{n_repeats}")

            with self.temporarily_extended_context(subroutine_name):
                self._run_flow(DynamicDataObject.from_obj(routine_data))

            incumbent_sol = self.solution_manager.get_incumbent()
            if incumbent_sol is None:
                obj_after = math.inf
            else:
                obj_after = self.get_obj_value(incumbent_sol)

            if float_a_stl_b(obj_after, obj_before):
                logging.info(
                    f"[Repeat] Improvement observed ({obj_before} -> {obj_after}). Continuing."
                )
                obj_before = obj_after
            else:
                logging.info(
                    f"[Repeat] No improvement observed ({obj_before} -> {obj_after}). Stopping repeats."
                )
                break
