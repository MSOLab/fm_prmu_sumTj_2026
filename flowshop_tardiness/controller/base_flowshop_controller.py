import datetime
import logging
from pathlib import Path
from typing import Any, Sequence

from mbls.cpsat import ObjValueBoundStore
from routix import (
    DynamicDataObject,
    ElapsedTimer,
    StoppingCriteria,
    SubroutineController,
)
from routix.io import dump_yaml
from routix.util.comparison import float_a_leq_b, float_equals
from schore.parameters_examples.shop.flow import FlowshopDuedateParameters
from schore.schedule_examples.shop.flow import FlowshopSchedule

from ..solution_manager import FsSolutionManager


class BaseFlowshopController(SubroutineController[StoppingCriteria, Any]):
    """Solver-agnostic base controller matching this project's runner expectations.

    This controller intentionally does NOT assume CP-SAT / CpSubroutineController.
    It provides:
    - compatible __init__ signature (instance/shared_param_dict/subroutine_flow/stopping_criteria)
    - resume-aware run(flow_resume_idx=...)
    - solution_manager + objective feasibility utilities used by runners
    - basic stopping condition helpers
    """

    instance: FlowshopDuedateParameters
    """The FlowshopDuedateParameters instance being solved."""

    shared_param_dict: dict

    solution_manager: FsSolutionManager
    """Solution manager for Hybrid Flow Shop scheduling solutions."""

    total_elapsed_time: float
    """Total elapsed time for the controller."""

    method_names_to_run_before_resume: set[str]
    """Name of methods to run before resuming from a paused state."""

    def __init__(
        self,
        instance: FlowshopDuedateParameters,
        shared_param_dict: dict,
        subroutine_flow: Sequence[DynamicDataObject] | DynamicDataObject,
        stopping_criteria: StoppingCriteria,
        start_dt: datetime.datetime | None = None,
    ):
        super().__init__(
            name=instance.name,
            subroutine_flow=subroutine_flow,
            stopping_criteria=stopping_criteria,
            start_dt=start_dt,
        )

        self.instance = instance
        self.shared_param_dict = shared_param_dict

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.obj_store = ObjValueBoundStore[float]()
        """Store for objective value and bound time series."""

        self.solution_manager = FsSolutionManager()
        self.solution_manager.set_job_2_duedate_map(instance.job_2_duedate_map)

        # Optional: subclasses can populate this for RESUME semantics.
        self.method_names_to_run_before_resume = set()

        # Frequently used parameters
        self.job_cnt: int = len(self.instance.job_id_list)
        self.stage_ids: tuple[str, ...] = tuple(self.instance.stage_id_list)
        self.stage_cnt: int = len(self.stage_ids)
        self.last_stage_id: str = self.stage_ids[-1]
        self.job_2_stage_2_p_dict: dict[str, dict[str, int]] = (
            self.instance.get_job_2_stage_2_builtin_int_p_map()
        )
        self.stage_2_job_2_p_dict: dict[str, dict[str, int]] = (
            self.instance.get_stage_2_job_2_builtin_int_p_map()
        )
        self.stage_job_2_p_dict: dict[tuple[str, str], int] = (
            self.instance.get_stage_job_2_builtin_int_p_map()
        )

        self.total_elapsed_time = 0.0

    def set_working_dir(self, dir_path: Path | str):
        super().set_working_dir(dir_path)
        self.log_handlers: list[logging.Handler] = []
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

    # -----------------
    # Stopping condition
    # -----------------

    def is_stopping_condition(self, log_reason_if_true: bool = True, **kwargs) -> bool:
        return self.ub_equals_lb(log_reason_if_true) or self.time_is_up(
            log_reason_if_true
        )

    def ub_equals_lb(self, log_reason_if_true: bool = True) -> bool:
        best_obj_value = self.solution_manager.best_obj_value
        best_obj_bound = self.solution_manager.best_obj_bound

        if best_obj_value is None or best_obj_bound is None:
            return False
        if float_equals(best_obj_value, best_obj_bound):
            if log_reason_if_true:
                logging.info(
                    f"Stop by UB == LB: best objective value ({best_obj_value}) "
                    f"equals best objective bound ({best_obj_bound})."
                )
            return True
        if self.solution_manager._a_is_better_obj_value(best_obj_value, best_obj_bound):
            raise ValueError(
                f"Inconsistent state: best objective value ({best_obj_value}) "
                f"is strictly better than best objective bound ({best_obj_bound})."
            )
        return False

    def time_is_up(self, log_reason_if_true: bool = True) -> bool:
        if self.stopping_criteria.timelimit is None:
            return False
        if float_a_leq_b(self.stopping_criteria.timelimit, self.timer.elapsed_sec):
            if log_reason_if_true:
                logging.info("Stop by timelimit")
            return True
        return False

    def get_remaining_sec(self) -> float:
        return self.timer.get_remaining_sec(self.stopping_criteria.timelimit)

    def get_remaining_time_limit(self, subroutine_time_limit: float | None) -> float:
        if subroutine_time_limit is None:
            return self.get_remaining_sec()
        return min(subroutine_time_limit, self.get_remaining_sec())

    # -----------------
    # Runner integration
    # -----------------

    def run(self, flow_resume_idx: int = -1) -> None:
        """Execute subroutine flow with optional RESUME semantics.

        - When resuming, steps before flow_resume_idx are skipped.
        - Methods listed in method_names_to_run_before_resume are executed even
          when skipped, and are treated as not consuming global timelimit.
        """
        if isinstance(self._subroutine_flow, Sequence) and not isinstance(
            self._subroutine_flow, (str, bytes)
        ):
            for idx, subroutine_data in enumerate(self._subroutine_flow):
                if idx < flow_resume_idx:
                    if (
                        subroutine_data.get("method", "")
                        in self.method_names_to_run_before_resume
                    ):
                        # Always run specific initializer methods when resuming
                        # even if they were already executed before pausing.
                        # e.g., set_random_seed
                        # Treat their execution as not consuming the global time limit.
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
        incumbent = self.solution_manager.get_incumbent()
        if incumbent:
            self.check_feasibility(incumbent)
        self.release_log_handlers()
        self.total_elapsed_time = self.timer.elapsed_sec

    def get_obj_value(self, schedule: FlowshopSchedule) -> float:
        return float(schedule.get_total_tardiness(self.instance.job_2_duedate_map))

    def check_feasibility(self, schedule: FlowshopSchedule) -> float:
        """Light feasibility checks used by runners and resume logic."""
        sub_timer = ElapsedTimer()
        logging.info("Feasibility check starts")

        j_list = self.instance.job_id_list
        i_list = self.instance.stage_id_list

        total_ops = sum(len(schedule.get_stage_by_name(i).operations) for i in i_list)
        assert total_ops == len(j_list) * len(i_list), (
            f"Total operations {total_ops} does not match expected {len(j_list) * len(i_list)}"
        )

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
        """Feasibility check used by RESUME loader (multi-instance runner)."""
        logging.info("Feasibility check starts")

        j_list = self.instance.job_id_list
        i_list = self.instance.stage_id_list

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

        for i in i_list:
            if len(i_2_j_2_end_time_map.get(i, {})) != len(j_list):
                raise ValueError(f"Stage {i} does not have all jobs scheduled.")

        reference_sequence = sorted(
            j_list, key=lambda j: i_2_j_2_end_time_map[i_list[0]][j]
        )
        for stage_name in i_list[1:]:
            current_sequence = sorted(
                j_list, key=lambda j: i_2_j_2_end_time_map[stage_name][j]
            )
            if current_sequence != reference_sequence:
                raise ValueError(
                    f"Job sequence mismatch between stage {i_list[0]} & {stage_name}"
                )

        logging.info("Feasibility check passed")

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
        return float(obj_value)

    # Start visualization

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
            START_TIME_MAP_KEY: start_time_map,
            END_TIME_MAP_KEY: end_time_map,
        }
        dump_yaml(solution_dict, output_path, encoding=encoding)

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
