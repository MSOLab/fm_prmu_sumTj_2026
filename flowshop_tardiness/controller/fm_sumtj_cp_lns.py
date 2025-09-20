import logging
from typing import Any

from routix import ElapsedTimer

from ..report import FsCpsatSolverReport, FsSubroutineReport
from ..scheduling.flowshop_schedule import FlowshopSchedule
from .controller_core import FlowshopTardinessControllerCore


class FlowshopTardinessCpLnsController(FlowshopTardinessControllerCore):
    # Start subroutine definition

    def solve_current_cp_remaining_time_limit(
        self,
        computational_time: float,
        solver_thread_cnt: int,
        no_improvement_timelimit: float | None = None,
        obj_value_is_valid: bool = False,
        obj_bound_is_valid: bool = False,
        is_initial_solution: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ):
        _timelimit = self.get_remaining_time_limit(computational_time)

        # Utilize the objective bound if available
        if obj_value_is_valid and self.solution_manager.best_obj_bound is not None:
            self.cp_model.set_obj_lower_bound(self.solution_manager.best_obj_bound)

        # mdl_txt_path = self.get_file_path_for_subroutine("_cp_sat_model.txt")
        # self.cp_model.export_to_file(str(mdl_txt_path))

        solver_report = self.solve_current_cp_model(
            _timelimit,
            solver_thread_cnt,
            no_improvement_timelimit=no_improvement_timelimit,
            random_seed=self.random_seed,
            e_timer=self.timer,
            log_level_obj_value=logging.INFO,
            log_level_obj_bound=logging.INFO,
            obj_value_is_valid=obj_value_is_valid,
            obj_bound_is_valid=obj_bound_is_valid,
        )

        solver_report = FsCpsatSolverReport.from_other(
            solver_report, is_init=is_initial_solution
        )

        # If the objective value or bound is not valid, use the best known values.
        report_updates: dict[str, Any] = {}
        if obj_value_is_valid:
            report_updates["obj_value"] = solver_report.obj_value
        else:
            report_updates["obj_value"] = self.solution_manager.best_obj_value
        if obj_bound_is_valid:
            report_updates["obj_bound"] = solver_report.obj_bound
        else:
            report_updates["obj_bound"] = self.solution_manager.best_obj_bound

        if report_updates:
            new_hfs_solver_report = solver_report.copy(
                obj_value=report_updates.get("obj_value"),
                obj_bound=report_updates.get("obj_bound"),
            )
            solver_report = new_hfs_solver_report

        solution: FlowshopSchedule | None = None
        if solver_report.is_feasible:
            solution = self.cp_model.create_schedule()
            if error_if_infeasible:
                self.check_feasibility(solution.get_start_time_map())
            obj_value_by_solution = solution.get_total_tardiness(
                self.instance.job_2_duedate_map
            )
            if obj_value_by_solution != solver_report.obj_value:
                raise ValueError(
                    f"Objective value mismatch: Reported {solver_report.obj_value}, "
                    f"Calculated {obj_value_by_solution}"
                )
        # Register the solution
        was_updated = self.solution_manager.register(solver_report, solution)
        if was_updated and draw_gantt:
            self.draw_incumbent_gantt()

    def solve_with_initial_solution(
        self,
        computational_time: float,
        solver_thread_cnt: int,
        no_improvement_timelimit: float | None = None,
        obj_value_is_valid: bool = False,
        obj_bound_is_valid: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ):
        incumbent_solution = self.solution_manager.get_incumbent()
        is_initial_run = incumbent_solution is None

        if incumbent_solution:
            self.cp_model.clear_hints()
            logging.info(
                "Applying incumbent solution with objValue "
                f"{incumbent_solution.makespan} as a hint."
            )
            self.cp_model.add_start_hints_from_start_time_map(
                incumbent_solution.get_start_time_map(),
            )

        self.solve_current_cp_remaining_time_limit(
            computational_time,
            solver_thread_cnt,
            no_improvement_timelimit=no_improvement_timelimit,
            obj_value_is_valid=obj_value_is_valid,
            obj_bound_is_valid=obj_bound_is_valid,
            is_initial_solution=is_initial_run,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
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
        Solve the base CP model for the hybrid flow shop problem.

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
        job_sequence = self.get_edd_sequence()
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
            self.check_feasibility(schedule.get_start_time_map())
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
