import datetime
import logging
import math
from pathlib import Path
from typing import Any, Sequence

from mbls.cpsat import CpsatSolverReport, CpSubroutineController
from routix import DynamicDataObject, ElapsedTimer, StoppingCriteria
from routix.util.comparison import float_a_leq_b, float_equals
from schore.parameters_examples.shop.flow import FlowshopDuedateParameters

from ..cp_cpsat_indirect_prec import CpCpsatIndirectPrec
from ..painter.gantt import GanttPlotter
from ..report import FsCpsatSolverReport
from ..scheduling.flowshop_schedule import FlowshopSchedule
from ..solution_manager import FsSolutionManager


class FlowshopTardinessControllerCore(
    CpSubroutineController[
        FlowshopDuedateParameters, CpCpsatIndirectPrec, StoppingCriteria
    ]
):
    # Start controller states

    solution_manager: FsSolutionManager
    """Solution manager for Hybrid Flow Shop scheduling solutions."""
    total_elapsed_time: float
    """Total elapsed time for the controller."""

    # End controller state

    # Start controller pre-defined values
    method_names_to_run_before_resume: set[str]
    """Name of methods to run before resuming from a paused state."""
    log_solver_level: int
    """Logging level for the solver output."""
    log_search_progress: bool
    """Whether to log search progress during CP solving."""

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
            CpCpsatIndirectPrec,
            subroutine_flow,
            stopping_criteria,
        )
        self.solution_manager = FsSolutionManager()
        self.solution_manager.set_job_2_duedate_map(instance.job_2_duedate_map)

        self.method_names_to_run_before_resume = {
            "set_random_seed",
            "set_cp_model_as_base_cp_model",
        }
        assert "" not in self.method_names_to_run_before_resume
        self.log_solver_level = logging.INFO  # TODO: make it configurable
        self.log_search_progress = False  # TODO: make it configurable

        # Frequently used parameters
        self.job_2_stage_2_p_dict = self.instance.p_manager.job_2_stage_2_value_map(
            self.instance.job_id_list, self.instance.stage_id_list
        )
        """Job name -> stage name -> processing time map"""

        logging.info(
            f"Controller initialized; took {self.timer.elapsed_sec:.3f} sec. "
            f"Start solving {self.instance.name} using CP model class:"
            f" {self.cp_model_class.__module__}.{self.cp_model_class.__name__}",
        )

    # Start abstract getters

    def create_base_cp_model(self, **kwargs) -> CpCpsatIndirectPrec:
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
                # Always run specific initializer methods when resuming
                # even if they were already executed before pausing.
                # e.g., set_random_seed, set_cp_model_as_base_cp_model
                # Treat their execution as not consuming the global time limit.
                if idx < flow_resume_idx:
                    if (
                        subroutine_data.get("method", "")
                        in self.method_names_to_run_before_resume
                    ):
                        e_timer = ElapsedTimer()
                        self._run_flow(subroutine_data)
                        virtual_dt = datetime.datetime.now() - datetime.timedelta(
                            seconds=e_timer.elapsed_sec
                        )
                        self.timer.set_start_time(virtual_dt)
                    else:
                        self._run_flow(subroutine_data, skip_method_call=True)
                else:
                    self._run_flow(subroutine_data)
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
            self.check_feasibility(incumbent)
        self.release_log_handlers()
        self.total_elapsed_time = self.timer.elapsed_sec

    def get_obj_value(self, schedule: FlowshopSchedule) -> float:
        return schedule.get_total_tardiness(self.instance.job_2_duedate_map)

    def check_feasibility(self, schedule: FlowshopSchedule) -> float:
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
        sub_timer = ElapsedTimer()
        logging.info("Feasibility check starts")

        j_list = self.instance.job_id_list
        i_list = self.instance.stage_id_list

        # All operations should be scheduled
        total_ops = sum(len(schedule.get_stage_by_name(i).operations) for i in i_list)
        assert total_ops == len(j_list) * len(i_list), (
            f"Total operations {total_ops} does not match expected {len(j_list) * len(i_list)}"
        )

        # Each stage should have the same job sequence
        i_2_j_list_map = schedule.get_stage_2_job_list_map()
        reference_sequence = i_2_j_list_map[i_list[0]]
        for stage_name in i_list[1:]:
            if i_2_j_list_map[stage_name] != reference_sequence:
                raise ValueError(
                    f"Job sequence mismatch between stage {i_list[0]} & {stage_name}."
                )

        if schedule.makespan > self.get_horizon():
            logging.warning(
                f"Schedule makespan {schedule.makespan} exceeds horizon {self.get_horizon()}."
            )

        logging.info(f"Feasibility check passed; took {sub_timer.elapsed_sec:.2f} sec")

        return self.get_obj_value(schedule)

    def check_end_time_map_feasibility(
        self, end_time_map: dict[tuple[str, str], int]
    ) -> float:
        """Check the feasibility of the given end times.

        Args:
            end_time_map (dict[tuple[str, str], int]): A mapping of (job, stage) to end time.

        Returns:
            float: The objective value of the solution if feasible.
        """
        logging.info("Feasibility check starts")

        j_list = self.instance.job_id_list
        i_list = self.instance.stage_id_list

        # Validate end times
        for (j, i), end_time in end_time_map.items():
            if j not in j_list:
                raise ValueError(f"Job {j} not found in job list.")
            if i not in i_list:
                raise ValueError(f"Stage {i} not found in stage list.")
            if end_time < 0:
                raise ValueError(
                    f"Negative end time {end_time} for job {j}, stage {i}."
                )

        i_2_j_2_end_time_map: dict[str, dict[str, int]] = {}
        for (j, i), end_time in end_time_map.items():
            i_2_j_2_end_time_map.setdefault(i, {})[j] = end_time

        # All operations should be scheduled
        for i in i_list:
            if len(i_2_j_2_end_time_map.get(i, {})) != len(j_list):
                raise ValueError(f"Stage {i} does not have all jobs scheduled.")

        # Each stage should have the same job sequence
        # Sort jobs by end time within the first stage
        reference_sequence = sorted(
            j_list, key=lambda j: i_2_j_2_end_time_map[i_list[0]][j]
        )
        for stage_name in i_list[1:]:
            # Sort jobs by end time within the current stage
            current_sequence = sorted(
                j_list, key=lambda j: i_2_j_2_end_time_map[stage_name][j]
            )
            if current_sequence != reference_sequence:
                raise ValueError(
                    f"Job sequence mismatch between stage {i_list[0]} & {stage_name}"
                )

        logging.info("Feasibility check passed")

        # Calc objective value
        obj_value = 0
        last_stage = i_list[-1]
        for j in j_list:
            if j not in i_2_j_2_end_time_map[last_stage]:
                raise ValueError(f"Job {j} not found in last stage {last_stage}.")
            obj_value += max(
                0,
                i_2_j_2_end_time_map[last_stage][j]
                - self.instance.job_2_duedate_map[j],
            )
        return obj_value

    # End post-run process

    # Start solver call methods

    def solve_current_cp_remaining_time_limit(
        self,
        computational_time: float,
        solver_thread_cnt: int,
        no_improvement_timelimit: float | None = None,
        cp_model_presolve: bool | None = None,
        obj_value_is_valid: bool = False,
        obj_bound_is_valid: bool = False,
        is_initial_solution: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ):
        if not self.base_cp_model_is_set:
            raise RuntimeError(
                "Base CP model is not set. Call set_cp_model_as_base_cp_model() first."
            )

        _timelimit = self.get_remaining_time_limit(computational_time)

        # Utilize the objective bound if available
        if (
            obj_value_is_valid
            and self.solution_manager.best_obj_bound is not None
            and not math.isnan(self.solution_manager.best_obj_bound)
        ):
            self.cp_model.set_obj_lower_bound(self.solution_manager.best_obj_bound)

        # mdl_txt_path = self.get_file_path_for_subroutine("_cp_sat_model.txt")
        # self.cp_model.export_to_file(str(mdl_txt_path))

        solver_report: CpsatSolverReport = self.solve_current_cp_model(
            _timelimit,
            solver_thread_cnt,
            no_improvement_timelimit=no_improvement_timelimit,
            cp_model_presolve=cp_model_presolve,
            random_seed=self.random_seed,
            e_timer=self.timer,
            log_level_obj_value=logging.INFO,
            log_level_obj_bound=logging.INFO,
            log_level_solver=self.log_solver_level,
            log_search_progress=self.log_search_progress,
            obj_value_is_valid=obj_value_is_valid,
            obj_bound_is_valid=obj_bound_is_valid,
        )

        fs_solver_report = FsCpsatSolverReport.from_other(
            solver_report, is_init=is_initial_solution
        )

        # If the objective value or bound is not valid, use the best known values.
        report_updates: dict[str, Any] = {}
        if obj_value_is_valid:
            report_updates["obj_value"] = fs_solver_report.obj_value
        else:
            report_updates["obj_value"] = self.solution_manager.best_obj_value
        if obj_bound_is_valid:
            report_updates["obj_bound"] = fs_solver_report.obj_bound
        else:
            report_updates["obj_bound"] = self.solution_manager.best_obj_bound

        if report_updates:
            fs_solver_report = fs_solver_report.copy(
                obj_value=report_updates.get("obj_value"),
                obj_bound=report_updates.get("obj_bound"),
            )

        if fs_solver_report.obj_value is None:
            if obj_value_is_valid:
                logging.warning("Failed to find a valid objective value.")
                return
        else:
            solution: FlowshopSchedule | None = None
            if fs_solver_report.is_feasible:
                solution = self.cp_model.create_schedule_from_sequence()
                if error_if_infeasible:
                    self.check_feasibility(solution)
                obj_value_by_solution = self.get_obj_value(solution)
                if obj_value_by_solution != fs_solver_report.obj_value:
                    if obj_value_by_solution > fs_solver_report.obj_value:
                        raise ValueError(
                            f"Objective value mismatch: Reported {fs_solver_report.obj_value}, "
                            f"Calculated {obj_value_by_solution}"
                        )
                    else:  # obj_value_by_solution < solver_report.obj_value
                        logging.warning(
                            f"Objective value discrepancy: Reported {fs_solver_report.obj_value}, "
                            f"Calculated {obj_value_by_solution}"
                        )
                        last_timestamp = self.timer.elapsed_sec
                        self.add_obj_value_log(
                            last_timestamp,
                            obj_value_by_solution,
                            is_maximize=False,
                        )
                        extended_obj_value_records = fs_solver_report.obj_value_records
                        extended_obj_value_records.append(
                            (last_timestamp, obj_value_by_solution)
                        )
                        fs_solver_report = FsCpsatSolverReport(
                            elapsed_time=fs_solver_report.elapsed_time,
                            obj_value=obj_value_by_solution,
                            obj_bound=fs_solver_report.obj_bound,
                            status=fs_solver_report.status,
                            obj_value_records=extended_obj_value_records,
                            obj_bound_records=fs_solver_report.obj_bound_records,
                            is_init=fs_solver_report.is_init,
                        )
                # Register the solution
                was_updated = self.solution_manager.register(fs_solver_report, solution)
                if was_updated and draw_gantt:
                    self.draw_incumbent_gantt()

    def solve_with_initial_solution(
        self,
        computational_time: float,
        solver_thread_cnt: int,
        no_improvement_timelimit: float | None = None,
        cp_model_presolve: bool | None = None,
        obj_value_is_valid: bool = False,
        obj_bound_is_valid: bool = False,
        error_if_infeasible: bool = False,
        draw_gantt: bool = False,
    ):
        if not self.base_cp_model_is_set:
            raise RuntimeError(
                "Base CP model is not set. Call set_cp_model_as_base_cp_model() first."
            )

        incumbent_solution = self.solution_manager.get_incumbent()
        is_initial_run = incumbent_solution is None

        if incumbent_solution:
            incumbent_obj_value = self.get_obj_value(incumbent_solution)
            logging.info(
                "Applying incumbent solution with objValue "
                f"{incumbent_obj_value} as a hint."
            )
            self.cp_model.clear_hints()
            self.cp_model.add_hints_from_schedule(incumbent_solution)

        self.solve_current_cp_remaining_time_limit(
            computational_time,
            solver_thread_cnt,
            no_improvement_timelimit=no_improvement_timelimit,
            cp_model_presolve=cp_model_presolve,
            obj_value_is_valid=obj_value_is_valid,
            obj_bound_is_valid=obj_bound_is_valid,
            is_initial_solution=is_initial_run,
            error_if_infeasible=error_if_infeasible,
            draw_gantt=draw_gantt,
        )

    # End solver call methods
