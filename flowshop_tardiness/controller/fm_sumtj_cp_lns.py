import logging
from typing import Callable

from routix import ElapsedTimer

from ..report import FsSubroutineReport
from ..scheduling.flowshop_schedule import FlowshopSchedule
from .controller_core import FlowshopTardinessControllerCore


class FlowshopTardinessCpLnsController(FlowshopTardinessControllerCore):
    # Subroutine: methods to run before resuming from a paused state.

    def set_random_seed(self, seed: int):
        return super().set_random_seed(seed)

    def set_cp_model_as_base_cp_model(self) -> None:
        return super().set_cp_model_as_base_cp_model()

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
        self.cp_model.delete_added_constraints()
        if is_initial_solution:
            self.solve_current_cp_remaining_time_limit(
                computational_time,
                solver_thread_cnt,
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
                obj_value_is_valid=True,
                obj_bound_is_valid=True,
                error_if_infeasible=True,
                draw_gantt=draw_gantt,
            )

    # Helper methods

    def get_dispatched_schedule(self, job_sequence: list[str]) -> FlowshopSchedule:
        # Create an empty schedule
        schedule = FlowshopSchedule.from_stage_name_list(self.instance.stage_id_list)
        # Dispatch
        for j in job_sequence:
            added = schedule.dispatch_job_by_stages(
                j, self.instance.stage_id_list, self.job_2_stage_2_p_dict[j]
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
        obj_value = schedule.get_total_tardiness(self.instance.job_2_duedate_map)
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
        stage_list = self.instance.stage_id_list
        dmap = self.instance.job_2_duedate_map
        pmap = self.job_2_stage_2_p_dict

        def total_p(j: str) -> float:
            # Sum processing times across all stages for job j
            return sum(pmap[j].get(s, 0.0) for s in stage_list)

        def key_fn(j: str):
            tp = total_p(j)
            # safe division (handles pathological zero-proc jobs)
            denom = tp if tp > 0.0 else 1e-9
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

        i_list = self.instance.stage_id_list
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
                start_time = max(f[i], prev)
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
            schedule.dispatch_job_by_stages(best_job, i_list, pmap[best_job])
            frontier = best_f

        if error_if_infeasible:
            self.check_feasibility(schedule)
        obj_value = schedule.get_total_tardiness(self.instance.job_2_duedate_map)
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
        return max(self.instance.job_2_duedate_map[job_id], completion_time)

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
            self.job_2_stage_2_p_dict[job_id].get(s, 0.0)
            for s in self.instance.stage_id_list
        )
        if p_total <= 0.0:
            return float("inf")  # Handle zero processing time jobs
        return (
            float(self.instance.job_2_duedate_map[job_id] - completion_time) / p_total
        )

    def initialize_by_nehedd(
        self, error_if_infeasible: bool = False, draw_gantt: bool = False
    ) -> None:
        """
        NEH with EDD ordering (sum of tardiness objective), array-based fast evaluation.
        - No schedule/deepcopy during insertion trials (only once at the end).
        Complexity: O(n^2 * m) with small constants via prefix reuse.
        """
        sub_timer = ElapsedTimer()

        i_list = list(self.instance.stage_id_list)
        dmap = self.instance.job_2_duedate_map
        pmap = self.job_2_stage_2_p_dict

        # 1) EDD order
        edd_order = self.get_edd_sequence()

        seq: list[str] = []

        # ---- helpers ----

        def _simulate_append(frontier: list[int], job_id: str) -> tuple[list[int], int]:
            """Simulate appending job_id onto given machine frontier; return (new_frontier, C_last)."""
            f = frontier[:]  # copy
            prev = 0
            for i_idx, i in enumerate(i_list):
                p = pmap[job_id][i]
                start = f[i_idx] if i_idx < len(f) else 0  # safety
                start = max(start, prev)
                end = start + p
                if i_idx < len(f):
                    f[i_idx] = end
                else:
                    f.append(end)
                prev = end
            return f, f[-1]

        def _compute_prefix_frontiers_and_tardy(
            seq_now: list[str],
        ) -> tuple[list[list[int]], list[int]]:
            """
            For current seq_now, build:
            - prefix_frontiers[k]: frontier after first k jobs (k=0..len)
            - prefix_tardy[k]: sum of tardiness of first k jobs
            """
            m = len(i_list)
            prefix_frontiers: list[list[int]] = [[0] * m]
            prefix_tardy: list[int] = [0]
            for j in seq_now:
                f_prev = prefix_frontiers[-1]
                f_new, C_last = _simulate_append(f_prev, j)
                Tj = max(C_last - dmap[j], 0)
                prefix_frontiers.append(f_new)
                prefix_tardy.append(prefix_tardy[-1] + Tj)
            return prefix_frontiers, prefix_tardy

        def _eval_insert_total_tardy(
            seq_now: list[str], job_id: str
        ) -> tuple[int, int]:
            """
            Evaluate all insertion positions of job_id into seq_now.
            Returns (best_pos, best_total_tardy).
            Uses prefix reuse: head tardiness reused; tail recomputed from the chosen frontier.
            """
            if not seq_now:
                # only one position
                f0 = [0] * len(i_list)
                _, Cj = _simulate_append(f0, job_id)
                total = max(Cj - dmap[job_id], 0)
                return 0, total

            prefix_frontiers, prefix_tardy = _compute_prefix_frontiers_and_tardy(
                seq_now
            )
            best_pos = 0
            best_val: int | None = None

            # try all positions pos \in [0..len]
            for pos in range(len(seq_now) + 1):
                # head part tardiness is reused
                head_tardy = prefix_tardy[pos]
                frontier = prefix_frontiers[pos][:]  # start frontier at pos
                total_tardy = head_tardy

                # insert the new job
                frontier, C_new = _simulate_append(frontier, job_id)
                total_tardy += max(C_new - dmap[job_id], 0)

                # simulate tail jobs (pos..end) on this new frontier
                for k in range(pos, len(seq_now)):
                    j_tail = seq_now[k]
                    frontier, C_tail = _simulate_append(frontier, j_tail)
                    total_tardy += max(C_tail - dmap[j_tail], 0)

                # choose best; tie-breaker: earlier position (stable)
                if best_val is None or total_tardy < best_val:
                    best_val = total_tardy
                    best_pos = pos

            if best_val is None:
                raise RuntimeError("Unexpected: best_val is None after evaluation.")
            return best_pos, best_val

        # 2) NEH insertion by EDD order with fast evaluation
        for j in edd_order:
            pos, _ = _eval_insert_total_tardy(seq, j)
            seq.insert(pos, j)

        # 3) Build schedule once and register/log
        schedule = self.get_dispatched_schedule(seq)
        if error_if_infeasible:
            self.check_feasibility(schedule)

        obj_value = schedule.get_total_tardiness(self.instance.job_2_duedate_map)
        logging.info(f"Initialized by NEHEDD with total tardiness {obj_value}")

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
