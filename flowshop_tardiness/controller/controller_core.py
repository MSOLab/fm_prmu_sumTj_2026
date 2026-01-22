import datetime
import heapq
import logging
import math
from collections import defaultdict
from itertools import permutations
from pathlib import Path
from typing import Any, Sequence

from mbls.cpsat import (
    CpsatSolverReport,
    CpsatStatus,
    CpSubroutineController,
    CustomCpModel,
    ObjectiveBoundRecorder,
    ObjectiveValueRecorder,
)
from mbls.cpsat.callbacks import ValueBoundPair
from routix import DynamicDataObject, ElapsedTimer, StoppingCriteria
from routix.io import object_to_yaml, tuple_to_pyyaml_key
from routix.util.comparison import float_a_leq_b, float_equals
from schore.parameters_examples.shop.flow import FlowshopDuedateParameters
from schore.schedule_examples.shop.flow import FlowshopSchedule

from flowshop_tardiness.cpsat_model_2.indirect_prec import IndirectPrecVars
from flowshop_tardiness.cpsat_model_2.params import Params
from flowshop_tardiness.cpsat_model_2.position import PositionVars

from ..painter.gantt import GanttPlotter
from ..report import FsCpsatSolverReport
from ..solution_manager import FsSolutionManager


class FlowshopTardinessControllerCore(
    CpSubroutineController[FlowshopDuedateParameters, CustomCpModel, StoppingCriteria]
):
    instance: FlowshopDuedateParameters
    """The FlowshopDuedateParameters instance being solved."""

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
            CustomCpModel,
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
        self.job_cnt: int = len(self.instance.job_id_list)
        self.stage_ids: tuple[str, ...] = tuple(self.instance._stage_id_list)
        self.stage_cnt: int = len(self.stage_ids)
        self.last_stage_id: str = self.stage_ids[-1]
        self.job_2_stage_2_p_dict: dict[str, dict[str, int]] = (
            self.instance.get_job_2_stage_2_builtin_int_p_map()
        )
        """Job name -> stage name -> processing time map"""

        self.stage_2_job_2_p_dict: dict[str, dict[str, int]] = (
            self.instance.get_stage_2_job_2_builtin_int_p_map()
        )
        """Stage name -> job name -> processing time map"""

        self.stage_job_2_p_dict: dict[tuple[str, str], int] = (
            self.instance.get_stage_job_2_builtin_int_p_map()
        )
        """(Stage name, Job name) -> processing time map"""

        logging.info(
            f"Controller initialized; took {self.timer.elapsed_sec:.3f} sec. "
            f"Start solving {self.instance.name} using CP model class:"
            f" {self.cp_model_class.__module__}.{self.cp_model_class.__name__}",
        )

    # Start abstract getters

    def create_base_cp_model(self, **kwargs) -> CustomCpModel:
        from ..cpsat_model_2.indirect_prec import BaseModelBuilder

        builder = BaseModelBuilder()
        mdl, params, vars = builder.build(self.instance)
        self.params: Params = params
        self.vars: PositionVars = vars
        # Set total tardiness as the objective
        if vars.total_tardiness is None:
            raise ValueError("Total tardiness variable is not defined in the model.")
        mdl.minimize(vars.total_tardiness)
        return mdl

    # End abstract getters

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

    def export_solution_to_yaml(
        self,
        start_time_map: dict[tuple[str, str], int],
        end_time_map: dict[tuple[str, str], int],
        output_path: Path | None = None,
        encoding="utf-8",
    ):
        if output_path is None:
            # Filename suffix should be the same as in fs_single_instance_runner.py line 160
            output_path = self.get_file_path_for_subroutine("_solution.yaml")
        from ..io_solution import END_TIME_MAP_KEY, START_TIME_MAP_KEY

        solution_dict = {
            START_TIME_MAP_KEY: tuple_to_pyyaml_key(start_time_map),
            END_TIME_MAP_KEY: tuple_to_pyyaml_key(end_time_map),
        }
        object_to_yaml(solution_dict, output_path, encoding=encoding)

    def export_schedule_to_yaml(
        self, schedule: FlowshopSchedule, output_path: Path | None = None
    ):
        self.export_solution_to_yaml(
            schedule.get_start_time_map(),
            schedule.get_end_time_map(),
            output_path,
        )

    def export_incumbent_to_yaml(self, output_path: Path | None = None):
        incumbent_solution = self.solution_manager.get_incumbent()
        if isinstance(incumbent_solution, FlowshopSchedule):
            self.export_schedule_to_yaml(incumbent_solution, output_path)

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

    def get_job_processing_time_sum(self, i: str, j_list: Sequence[str]) -> int:
        """Get the sum of processing times for a list of jobs at a specific stage.

        Args:
            i (str): The stage ID.
            j_list (Sequence[str]): A sequence of job IDs.
        Returns:
            int: The sum of processing times for the specified jobs at the given stage.
        """
        return sum(self.stage_2_job_2_p_dict[i][j] for j in j_list)

    # Start solver call methods

    def solve_cp_model_2(
        self,
        mdl: CustomCpModel,
        computational_time: float,
        solver_thread_cnt: int,
        e_timer: ElapsedTimer | None = None,
        print_on_obj_value_update: bool = False,
        print_on_obj_bound_update: bool = False,
        log_level_obj_value: int = logging.INFO,
        log_level_obj_bound: int = logging.INFO,
        obj_value_is_valid: bool = False,
        obj_bound_is_valid: bool = False,
        last_timestamp_note: Any | None = None,
    ) -> CpsatSolverReport:
        from ..cpsat_model_2.solver import SolveConfig, configure_solver

        _timelimit = self.get_remaining_time_limit(computational_time)
        if e_timer is None:
            e_timer = self.timer

        solve_cfg = SolveConfig(
            log=self.log_search_progress,
            time_limit_s=_timelimit,
            num_workers=solver_thread_cnt,
            random_seed=self.random_seed,
        )
        self.solver = configure_solver(solve_cfg)
        obj_value_recorder = ObjectiveValueRecorder(
            e_timer,
            print_on_record=print_on_obj_value_update,
            log_level_on_record=log_level_obj_value,
        )

        obj_bound_recorder = ObjectiveBoundRecorder(
            e_timer,
            print_on_record=print_on_obj_bound_update,
            log_level_on_record=log_level_obj_bound,
        )
        self.solver.best_bound_callback = obj_bound_recorder

        cp_solver_status = self.solver.solve(mdl, solution_callback=obj_value_recorder)
        cpsat_status = CpsatStatus.from_cp_solver_status(cp_solver_status)
        elapsed_time = self.solver.wall_time
        if cpsat_status.is_feasible:
            obj_value = self.solver.objective_value
            if cpsat_status == CpsatStatus.OPTIMAL:
                obj_bound = obj_value
            else:
                obj_bound = self.solver.best_objective_bound
        else:
            obj_value, obj_bound = CpsatStatus.get_obj_value_and_bound_for_infeasible(
                False
            )

        last_timestamp = e_timer.elapsed_sec

        # Store the objective value and bound logs

        def get_obj_value_records() -> list[tuple[float, float]]:
            """Returns the recorded objective values and elapsed times.

            Returns:
                list[tuple[float, float]]: A list of tuples containing (elapsed time, objective value).
            """
            return_list: list[tuple[float, float]] = []
            list_by_value_recorder: list[tuple[float, ValueBoundPair]] = (
                obj_value_recorder.entries
            )
            for entry in list_by_value_recorder:
                return_list.append((entry[0], entry[1].value))
            return return_list

        obj_value_records: list[tuple[float, float]] = []
        if obj_value_is_valid:
            obj_value_records = get_obj_value_records()
            if cpsat_status.is_feasible:
                obj_value_records.append((last_timestamp, obj_value))
            self.extend_obj_value_log(
                obj_value_records, is_maximize=self.cp_model.is_maximize()
            )
            # Record value for the last timestamp if it is the same as the last value
            # and is not recorded for the last timestamp
            if (
                obj_value == self.obj_store.get_last_obj_value()
                and (last_timestamp, obj_value) not in obj_value_records
            ):
                self.add_obj_value_log(last_timestamp, obj_value, is_maximize=None)

        def get_obj_bound_records() -> list[tuple[float, float]]:
            """Returns the recorded objective bounds and elapsed times.

            Returns:
                list[tuple[float, float]]: A list of tuples containing (elapsed time, objective bound).
            """
            timestamp_list = []
            timestamp_2_bound_map: dict[float, float] = {}

            list_by_bound_recorder: list[tuple[float, float]] = (
                obj_bound_recorder.elapsed_time_and_bound
            )
            for b_entry in list_by_bound_recorder:
                timestamp = b_entry[0]
                bound = b_entry[1]
                if timestamp not in timestamp_list:
                    timestamp_list.append(timestamp)
                timestamp_2_bound_map[timestamp] = bound

            list_by_value_recorder: list[tuple[float, ValueBoundPair]] = (
                obj_value_recorder.entries
            )
            for v_entry in list_by_value_recorder:
                timestamp = v_entry[0]
                bound = v_entry[1].bound
                if timestamp not in timestamp_list:
                    timestamp_list.append(timestamp)
                if timestamp not in timestamp_2_bound_map:
                    timestamp_2_bound_map[timestamp] = bound

            timestamp_list.sort()
            return [
                (timestamp, timestamp_2_bound_map[timestamp])
                for timestamp in timestamp_list
            ]

        obj_bound_records: list[tuple[float, float]] = []
        if obj_bound_is_valid:
            obj_bound_records = get_obj_bound_records()
            if cpsat_status.is_feasible:
                obj_bound_records.append((last_timestamp, obj_bound))
            self.extend_obj_bound_log(obj_bound_records, is_maximize=False)
            # Record bound for the last timestamp if it is the same as the last bound
            # and is not recorded for the last timestamp
            if (
                obj_bound == self.obj_store.get_last_obj_bound()
                and (last_timestamp, obj_bound) not in obj_bound_records
            ):
                self.add_obj_bound_log(last_timestamp, obj_bound, is_maximize=None)

        _last_timestamp_note = (
            last_timestamp_note or self._get_call_context_of_current_method()
        )
        self.obj_store.add_last_timestamp_note(
            _last_timestamp_note,
            obj_value_is_valid=obj_value_is_valid,
            obj_bound_is_valid=obj_bound_is_valid,
        )

        solver_report = CpsatSolverReport(
            elapsed_time,
            obj_value,
            obj_bound,
            cpsat_status,
            obj_value_records,
            obj_bound_records,
        )
        return solver_report

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
    ) -> tuple[FsCpsatSolverReport, FlowshopSchedule | None]:
        if not self.base_cp_model_is_set:
            raise RuntimeError(
                "Base CP model is not set. Call set_cp_model_as_base_cp_model() first."
            )

        # Utilize the objective bound if available
        if (
            obj_value_is_valid
            and self.solution_manager.best_obj_bound is not None
            and not math.isnan(self.solution_manager.best_obj_bound)
        ):
            self.set_sumTj_lower_bound(
                self.cp_model, self.vars, bound=self.solution_manager.best_obj_bound
            )

        # mdl_txt_path = self.get_file_path_for_subroutine("_cp_sat_model.txt")
        # self.cp_model.export_to_file(str(mdl_txt_path))

        solver_report = self.solve_cp_model_2(
            self.cp_model,
            computational_time,
            solver_thread_cnt,
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

        solution: FlowshopSchedule | None = None
        if fs_solver_report.obj_value is None:
            if obj_value_is_valid:
                logging.warning("Failed to find a valid objective value.")
        else:
            if fs_solver_report.is_feasible:
                solution = self.create_schedule_from_params_and_vars(
                    params=self.params,
                    vars=self.vars,
                )
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
            else:
                logging.warning(
                    "No feasible solution found in the current CP model solving."
                )
        return fs_solver_report, solution

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
    ) -> tuple[FsCpsatSolverReport, FlowshopSchedule | None]:
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
            self.add_hints_from_schedule(
                self.cp_model, self.params, self.vars, incumbent_solution
            )

        return self.solve_current_cp_remaining_time_limit(
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

    def _get_j_sequence_from_solver(
        self, params: Params, vars: IndirectPrecVars
    ) -> list[int]:
        """Get the job index sequence from the solver.

        Args:
            params (Params): parameter instance
            vars (Vars): variable instance

        Returns:
            list[int]: The job index sequence as determined by the solver.
        """
        if isinstance(vars, PositionVars):
            return [self.solver.Value(vars.pi[k]) for k in params.j_list]
        if isinstance(vars, IndirectPrecVars):
            j_list = params.j_list
            prec: dict[tuple[int, int], bool] = {}
            for j1, j2 in permutations(j_list, 2):
                prec[j1, j2] = bool(self.solver.Value(vars.prec[j1, j2]))
            return self.from_job_prec_get_sequence(params, prec)
        raise TypeError(
            f"Unsupported variable type for getting job sequence: {type(vars)}"
        )

    @staticmethod
    def from_job_prec_get_sequence(
        params: Params, prec: dict[tuple[int, int], bool]
    ) -> list[int]:
        j_list = params.j_list.copy()

        # Reconstruct job sequence from precedence variables
        # 2) edge 구성: a->b if prec[(a,b)] is True
        succ = defaultdict(list)
        indeg = {j: 0 for j in j_list}

        for (a, b), v in prec.items():
            if not v:
                continue
            if a not in indeg or b not in indeg:
                if a not in j_list:
                    j_list.append(a)
                if b not in j_list:
                    j_list.append(b)
            succ[a].append(b)
            indeg[b] += 1

        # 3) 위상정렬 (tie-break: sorted)
        ready = [j for j in j_list if indeg[j] == 0]
        heapq.heapify(ready)

        seq = []
        while ready:
            j = heapq.heappop(ready)
            seq.append(j)
            for k in succ[j]:
                indeg[k] -= 1
                if indeg[k] == 0:
                    heapq.heappush(ready, k)

        # 4) cycle 검출
        if len(seq) != len(indeg):
            remaining = [j for j in indeg if indeg[j] > 0]
            raise ValueError(
                f"Cycle detected (or inconsistent precedence). Remaining jobs: {remaining}"
            )

        return seq

    def get_job_sequence_from_solver(
        self, params: Params, vars: IndirectPrecVars
    ) -> list[str]:
        """Get the job name sequence from the solver's job index sequence.

        Args:
            params (Params): parameter instance
            vars (Vars): variable instance

        Returns:
            list[str]: The job name sequence as determined by the solver.
        """
        j_sequence = self._get_j_sequence_from_solver(params, vars)
        j_name_sequence = [params.j_2_job_name_map[j] for j in j_sequence]
        return j_name_sequence

    def create_schedule_from_sequence(
        self, params: Params, j_name_sequence: list[str]
    ) -> FlowshopSchedule:
        j_sequence = [params.job_name_2_j_map[j_name] for j_name in j_name_sequence]
        i_name_list = [params.i_2_stage_name_map[i] for i in params.i_list]
        schedule = FlowshopSchedule.from_stage_name_list(i_name_list)

        for j in j_sequence:
            j_name = params.j_2_job_name_map[j]
            i_2_p_map = {
                i_name: params.P[i, j]
                for i, i_name in params.i_2_stage_name_map.items()
            }
            schedule.dispatch_job_by_stages(
                j_name, i_name_list, i_2_p_map, after_last=True
            )

        return schedule

    def create_schedule_from_params_and_vars(
        self, params: Params, vars: IndirectPrecVars
    ) -> FlowshopSchedule:
        j_sequence: list[int] = self._get_j_sequence_from_solver(params, vars)
        j_name_sequence: list[str] = [params.j_2_job_name_map[j] for j in j_sequence]
        return self.create_schedule_from_sequence(
            params=params, j_name_sequence=j_name_sequence
        )

    # End solver call methods

    # Start model modification methods

    def add_hints_from_schedule(
        self,
        mdl: CustomCpModel,
        params: Params,
        vars: IndirectPrecVars,
        schedule: FlowshopSchedule,
        job_subset: set[str] | None = None,
    ) -> None:
        logging.info("Adding hints from schedule to CP model.")
        last_i = params.i_list[-1]
        last_i_name = params.i_2_stage_name_map[last_i]
        j_sequence = schedule.get_last_stage_job_list()
        j_sequence = [j for j in j_sequence if (job_subset is None or j in job_subset)]
        start_time_map = schedule.get_start_time_map()
        sum_Tj = 0

        all_ops_in_schedule = True

        if isinstance(vars, PositionVars):
            for k, j_name in enumerate(j_sequence):
                j = params.job_name_2_j_map[j_name]
                all_ops_of_j_in_schedule = True
                for i, i_name in params.i_2_stage_name_map.items():
                    if (j_name, i_name) in start_time_map:
                        start_hint = start_time_map[j_name, i_name]
                        P_ij = params.P[i, j]
                        mdl.add_hint(vars.op_start[i, k], start_hint)
                        mdl.add_hint(vars.op_lth[i, k], P_ij)
                        mdl.add_hint(vars.op_end[i, k], start_hint + P_ij)
                    else:
                        all_ops_in_schedule = False
                        all_ops_of_j_in_schedule = False
                mdl.add_hint(vars.pi[k], j)
                mdl.add_hint(vars.d[k], params.D[j])
                if all_ops_of_j_in_schedule:
                    assert (j_name, last_i_name) in start_time_map, (
                        f"Last operation of job {j_name} not found in start_time_map"
                    )
                    Tj = max(
                        0,
                        start_time_map[j_name, last_i_name]
                        + params.P[last_i, j]
                        - params.D[j],
                    )
                    mdl.add_hint(vars.T[k], Tj)
                    sum_Tj += Tj
        elif isinstance(vars, IndirectPrecVars):
            for j_name in j_sequence:
                j = params.job_name_2_j_map[j_name]
                all_ops_of_j_in_schedule = True
                for i, i_name in params.i_2_stage_name_map.items():
                    if (j_name, i_name) in start_time_map:
                        mdl.add_hint(
                            vars.op_start[i, j], start_time_map[j_name, i_name]
                        )
                    else:
                        all_ops_in_schedule = False
                        all_ops_of_j_in_schedule = False
                if all_ops_of_j_in_schedule:
                    assert (j_name, last_i_name) in start_time_map, (
                        f"Last operation of job {j_name} not found in start_time_map"
                    )
                    Tj = max(
                        0,
                        start_time_map[j_name, last_i_name]
                        + params.P[last_i, j]
                        - params.D[j],
                    )
                    mdl.add_hint(vars.T[j], Tj)
                    sum_Tj += Tj
            for j1_idx, j1_name in enumerate(j_sequence):
                for j2_name in j_sequence[j1_idx + 1 :]:
                    j1 = params.job_name_2_j_map[j1_name]
                    j2 = params.job_name_2_j_map[j2_name]
                    mdl.add_hint(vars.prec[j1, j2], 1)
                    mdl.add_hint(vars.prec[j2, j1], 0)

        if all_ops_in_schedule:
            if vars.total_tardiness is None:
                raise ValueError("total_tardiness undefined in Vars")
            mdl.add_hint(vars.total_tardiness, sum_Tj)

    def set_sumTj_lower_bound(
        self, mdl: CustomCpModel, vars: IndirectPrecVars, bound: float | None
    ) -> None:
        if bound is None:
            return
        if math.isnan(bound):
            return
        if vars.total_tardiness is None:
            raise ValueError("Objective variable is not defined yet.")

        # If the bound is very close to an integer, treat it as such.
        # Otherwise, use ceiling to ensure we don't cut off valid integer solutions.
        if float_equals(bound, round(bound)):
            int_bound = round(bound)
        else:
            int_bound = math.ceil(bound)

        mdl.add(vars.total_tardiness >= int_bound)

    # End model modification methods
