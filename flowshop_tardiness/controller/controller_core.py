import logging
from pathlib import Path
from typing import Any, Sequence

from mbls.cpsat import CpsatStatus, CpSubroutineController
from routix import DynamicDataObject, StoppingCriteria
from routix.util.comparison import float_a_leq_b, float_equals
from schore.parameters_examples.shop.flow import FlowshopDuedateParameters

from ..cp_cpsat_circuit import CpCpsatCircuit
from ..painter.gantt import GanttPlotter
from ..report import FsCpsatSolverReport
from ..scheduling.flowshop_schedule import FlowshopSchedule
from ..solution_manager import FsSolutionManager


class FlowshopTardinessControllerCore(
    CpSubroutineController[FlowshopDuedateParameters, CpCpsatCircuit, StoppingCriteria]
):
    # Start controller states

    solution_manager: FsSolutionManager
    """Solution manager for Hybrid Flow Shop scheduling solutions."""
    total_elapsed_time: float
    """Total elapsed time for the controller."""

    # End controller state
    def __init__(
        self,
        instance: FlowshopDuedateParameters,
        shared_param_dict: dict,
        subroutine_flow: Sequence[DynamicDataObject] | DynamicDataObject,
        stopping_criteria: StoppingCriteria,
    ):
        super().__init__(
            instance,
            shared_param_dict,
            CpCpsatCircuit,
            subroutine_flow,
            stopping_criteria,
        )
        self.solution_manager = FsSolutionManager()
        self.solution_manager.set_job_2_duedate_map(instance.job_2_duedate_map)

        # Frequently used parameters
        self.job_2_stage_2_p_dict = self.instance.p_manager.job_2_stage_2_value_map(
            self.instance.job_id_list, self.instance.stage_id_list
        )
        """Job name -> stage name -> processing time map"""
        self.stage_2_job_2_p_dict = self.instance.p_manager.stage_2_job_2_value_map(
            self.instance.stage_id_list, self.instance.job_id_list
        )
        """Stage name -> job name -> processing time map"""

        logging.info(
            f"Start solving {self.instance.name} using CP model class:"
            f" {self.cp_model_class.__module__}.{self.cp_model_class.__name__}",
        )

    # Start abstract getters

    def create_base_cp_model(self, **kwargs) -> CpCpsatCircuit:
        return self.cp_model_class.from_instance(self.instance, self.get_horizon())

    # End abstract getters

    def get_horizon(self) -> int:
        """Returns the horizon of the scheduling problem."""
        if "horizon" not in self.shared_param_dict:
            raise ValueError("Horizon not found in shared parameters.")
        return self.shared_param_dict["horizon"]

    def set_working_dir(self, dir_path: Path | str):
        super().set_working_dir(dir_path)
        self.log_handlers: list[logging.StreamHandler] = []
        self.add_file_handler()

    def add_file_handler(
        self,
        log_filename: str | None = None,
        level=logging.INFO,
        fmt="%(asctime)s - %(levelname)s - %(message)s",
    ):
        logger = logging.getLogger()
        _log_filename = log_filename or "subroutine_controller.log"
        if self._working_dir_path is not None:
            log_path = self._working_dir_path / _log_filename
            # 이미 같은 파일 핸들러가 등록되어 있는지 확인 (중복 방지)
            for handler in logger.handlers:
                if isinstance(
                    handler, logging.FileHandler
                ) and handler.baseFilename == str(log_path):
                    return  # 이미 등록되어 있으면 추가하지 않음

            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter(fmt))
            logger.addHandler(file_handler)
            self.log_handlers = [file_handler]

    def release_log_handlers(self) -> None:
        logger = logging.getLogger()
        for handler in self.log_handlers:
            logger.removeHandler(handler)
            handler.close()

    # Start stopping condition

    def is_stopping_condition(self, log_reason_if_true: bool = True, **kwargs) -> bool:
        return self.ub_equals_lb(log_reason_if_true) or self.time_is_up(
            log_reason_if_true
        )

    def ub_equals_lb(self, log_reason_if_true: bool = True) -> bool:
        """Checks if the current best objective equals the best objective bound.

        Raises:
            ValueError: If the current best objective is better than the best objective bound.

        Returns:
            bool: True if the current best objective equals the best objective bound,
                False otherwise.
        """
        best_obj_value = self.solution_manager.best_obj_value
        best_obj_bound = self.solution_manager.best_obj_bound

        if best_obj_value is None or best_obj_bound is None:
            # Case 1: Either ObjValue or ObjBound is None
            return False
        # best_obj_value is not None and best_obj_bound is not None

        # Case 2: ObjValue equals ObjBound
        # Considered equal if close enough (considering floating point precision)
        if float_equals(best_obj_value, best_obj_bound):
            if log_reason_if_true:
                logging.info(
                    f"Stop by UB == LB: best objective value ({best_obj_value}) "
                    f"equals best objective bound ({best_obj_bound})."
                )
            return True
        # Case 3: ObjValue is strictly better than ObjBound
        if self.solution_manager._a_is_better_obj_value(best_obj_value, best_obj_bound):
            if log_reason_if_true:
                raise ValueError(
                    f"Inconsistent state: best objective value ({best_obj_value}) "
                    f"is strictly better than best objective bound ({best_obj_bound})."
                )
        # Case 4: ObjValue is worse than ObjBound
        return False

    def time_is_up(self, log_reason_if_true: bool = True) -> bool:
        if self.stopping_criteria.timelimit is None:
            return False
        # If total elapsed time exceeds the stopping criteria
        if float_a_leq_b(self.stopping_criteria.timelimit, self.timer.elapsed_sec):
            if log_reason_if_true:
                logging.info("Stop by timelimit")
            return True
        return False

    def get_remaining_sec(self) -> float:
        return self.timer.get_remaining_sec(self.stopping_criteria.timelimit)

    def get_remaining_time_limit(self, subroutine_time_limit: float | None) -> float:
        """Get the remaining time limit for the subroutine.

        Args:
            subroutine_time_limit (float | None, optional): The time limit for the subroutine in seconds.
                If None, the remaining time limit is used.

        Returns:
            float: The minimum of the subroutine time limit and the remaining time limit.
        """
        if subroutine_time_limit is None:
            return self.get_remaining_sec()
        return min(subroutine_time_limit, self.get_remaining_sec())

    # End stopping condition

    # Start visualization

    def draw_gantt(self, schedule: FlowshopSchedule, output_path: Path | None = None):
        if output_path is None:
            output_path = self.get_file_path_for_subroutine("_gantt.png")
        plotter = GanttPlotter()
        plotter.export_flowshop_plot(
            output_path,
            schedule.get_start_time_map(),
            schedule.get_end_time_map(),
            self.instance.job_id_list,
        )

    def draw_incumbent_gantt(self, output_path: Path | None = None) -> None:
        incumbent_solution = self.solution_manager.get_incumbent()
        if isinstance(incumbent_solution, FlowshopSchedule):
            self.draw_gantt(incumbent_solution, output_path)
        else:
            logging.warning(
                "Incumbent solution is not a FlowshopSchedule. Cannot draw Gantt chart."
            )

    # End visualization

    def run(self, flow_resume_idx: int = -1) -> None:
        """Overrides the run method to execute the subroutine flow.

        Args:
            flow_resume_idx (int, optional): The index to resume the flow from. Defaults to -1.
        """
        if isinstance(self._subroutine_flow, Sequence) and not isinstance(
            self._subroutine_flow, (str, bytes)
        ):
            for idx, subroutine_data in enumerate(self._subroutine_flow):
                skip_method_call = idx < flow_resume_idx
                self._run_flow(subroutine_data, skip_method_call=skip_method_call)
        else:
            logging.warning(
                "Subroutine flow is not a sequence; running as a single step."
            )
            self._run_flow(self._subroutine_flow)
        self.post_run_process()

    # Start post-run process

    def post_run_process(self) -> None:
        """
        Finalizes the run by checking the feasibility of the incumbent solution
        and releasing log handlers.
        """
        incumbent = self.solution_manager.get_incumbent()
        if incumbent:
            self.check_feasibility(incumbent.get_start_time_map())
        self.release_log_handlers()
        self.total_elapsed_time = self.timer.elapsed_sec

    def check_feasibility(self, start_time_map: dict[tuple[str, str], int]) -> float:
        """Check the feasibility of the given start times.

        Args:
            start_time_map (dict[tuple[str, str], int]): A mapping of (job, stage) to start time.

        Raises:
            ValueError: If any start time is negative or invalid.
            RuntimeError: If the feasibility check fails while solving the model.
            ValueError: If the feasibility check fails with an unexpected status.

        Returns:
            float: The objective value of the solution if feasible.
        """
        logging.info("Feasibility check starts")
        for (j, i), start_time in start_time_map.items():
            if start_time < 0:
                raise ValueError(
                    f"Invalid start time for job {j}, stage {i}: {start_time}"
                )
        base_cp = self.create_base_cp_model()

        # Freeze operation start times and machine assignments
        for (j, i), start_time in start_time_map.items():
            base_cp.add(self.cp_model.var_op_start[j, i] == start_time)

        # Solve with tight time limit
        timelimit = 2.0
        solver_thread_cnt = 1
        solver_report = self.solve_cp_model(base_cp, timelimit, solver_thread_cnt)
        if solver_report.status not in (CpsatStatus.FEASIBLE, CpsatStatus.OPTIMAL):
            mdl_txt_path = self.get_file_path_for_subroutine(
                "_feasibility_check_failed.txt"
            )
            base_cp.export_to_file(str(mdl_txt_path))
            if solver_report.status == CpsatStatus.INFEASIBLE:
                raise RuntimeError(
                    f"Feasibility check failed: INFEASIBLE. Model saved to {mdl_txt_path}"
                )
            else:
                raise ValueError(
                    f"Feasibility check failed with status {solver_report.status}. "
                    f"Model saved to {mdl_txt_path}"
                )
        logging.info("Feasibility check passed")
        if solver_report.obj_value is None:
            raise ValueError("Feasibility check did not return an objective value.")
        return solver_report.obj_value

    # End post-run process

    # Start solver call methods

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
            solver_report = solver_report.copy(
                obj_value=report_updates.get("obj_value"),
                obj_bound=report_updates.get("obj_bound"),
            )

        if solver_report.obj_value is None:
            if obj_value_is_valid:
                logging.warning("Failed to find a valid objective value.")
                return
        else:
            solution: FlowshopSchedule | None = None
            if solver_report.is_feasible:
                solution = self.cp_model.create_schedule()
                if error_if_infeasible:
                    self.check_feasibility(solution.get_start_time_map())
                obj_value_by_solution = solution.get_total_tardiness(
                    self.instance.job_2_duedate_map
                )
                if obj_value_by_solution != solver_report.obj_value:
                    # schedule_Tj_map = solution.get_tardiness_map(
                    #     self.instance.job_2_duedate_map
                    # )
                    # solver_Tj_map = self.cp_model.extract_Tj_map()
                    # # Compute per-job differences (solver Tj - schedule Tj)
                    # solver_minus_schedule_map = {}
                    # for j, schedule_val in schedule_Tj_map.items():
                    #     solver_val = solver_Tj_map.get(j, 0)
                    #     if solver_val != schedule_val:
                    #         solver_minus_schedule_map[j] = solver_val - schedule_val

                    # # If there are any differences, raise an error with details
                    # if solver_minus_schedule_map:
                    #     raise ValueError(
                    #         "Per-job tardiness differences (solver Tj - schedule Tj): %s",
                    #         solver_minus_schedule_map,
                    #     )

                    # if sum(solver_Tj_map.values()) != obj_value_by_solution:
                    #     raise ValueError(
                    #         "Sum of per-job tardiness from solver (%d) does not match "
                    #         "objective value from solution (%d)."
                    #         % (sum(solver_Tj_map.values()), obj_value_by_solution)
                    #     )

                    # if sum(schedule_Tj_map.values()) != solver_report.obj_value:
                    #     raise ValueError(
                    #         "Sum of per-job tardiness from schedule (%d) does not match "
                    #         "objective value from solver (%d)."
                    #         % (sum(schedule_Tj_map.values()), solver_report.obj_value)
                    #     )
                    if obj_value_by_solution > solver_report.obj_value:
                        raise ValueError(
                            f"Objective value mismatch: Reported {solver_report.obj_value}, "
                            f"Calculated {obj_value_by_solution}"
                        )
                    else:  # obj_value_by_solution < solver_report.obj_value
                        logging.warning(
                            f"Objective value discrepancy: Reported {solver_report.obj_value}, "
                            f"Calculated {obj_value_by_solution}"
                        )
                        last_timestamp = self.timer.elapsed_sec
                        self.add_obj_value_log(
                            last_timestamp,
                            obj_value_by_solution,
                            is_maximize=False,
                        )
                        extended_obj_value_records = solver_report.obj_value_records
                        extended_obj_value_records.append(
                            (last_timestamp, obj_value_by_solution)
                        )
                        solver_report = FsCpsatSolverReport(
                            elapsed_time=solver_report.elapsed_time,
                            obj_value=obj_value_by_solution,
                            obj_bound=solver_report.obj_bound,
                            status=solver_report.status,
                            obj_value_records=extended_obj_value_records,
                            obj_bound_records=solver_report.obj_bound_records,
                            is_init=solver_report.is_init,
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

    # End solver call methods
