import logging

from routix import ElapsedTimer

from ..report import FsSubroutineReport
from ..scheduling.flowshop_schedule import FlowshopSchedule
from .controller_core import FlowshopTardinessControllerCore


class FlowshopTardinessCpLnsController(FlowshopTardinessControllerCore):
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
