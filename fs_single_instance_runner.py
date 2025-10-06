import datetime
import logging
from pathlib import Path
from typing import Any, Sequence

from mbls.cpsat import ObjValueBoundStore
from routix import DynamicDataObject, StoppingCriteria
from routix.io import object_to_yaml, tuple_to_pyyaml_key
from routix.runner import SingleInstanceRunner
from routix.type_defs import RunMode
from schore.parameters_examples.shop.flow import (
    FlowshopDuedateParameters,
)
from schore.schedule_examples.shop.flow import FlowshopOperation, FlowshopSchedule

from flowshop_tardiness.controller import FlowshopTardinessCpLnsController
from flowshop_tardiness.fs_input_summary import FsInputSummary
from flowshop_tardiness.fs_io_summary import FsIoSummary
from flowshop_tardiness.io_solution import END_TIME_MAP_KEY, START_TIME_MAP_KEY
from flowshop_tardiness.report.fs_subroutine_report import FsSubroutineReport
from flowshop_tardiness.report.fs_subroutine_report_statistics import (
    FsSubroutineReportStatistics,
)


class FsSingleInstanceRunner(
    SingleInstanceRunner[FlowshopDuedateParameters, FlowshopTardinessCpLnsController]
):
    # Optional member variables for RunMode.RESUME
    resume_start_time_map: dict | None = None
    """Start time map loaded from a resume solution file, if applicable."""
    resume_end_time_map: dict | None = None
    """End time map loaded from a resume solution file, if applicable."""
    resume_obj_store: ObjValueBoundStore | None = None
    """Objective value bound store loaded from a resume solution file, if applicable."""
    resume_summary_dict: dict[str, Any] | None = None
    """Summary dictionary loaded from a resume summary file, if applicable."""

    def __init__(
        self,
        instance: FlowshopDuedateParameters,
        shared_param_dict: dict,
        subroutine_flow: Sequence[DynamicDataObject] | DynamicDataObject,
        stopping_criteria: StoppingCriteria,
        output_dir: Path,
        output_metadata: dict[str, Any],
        mode: RunMode = RunMode.FULL_RUN,
    ):
        super().__init__(
            instance=instance,
            shared_param_dict=shared_param_dict,
            subroutine_flow=subroutine_flow,
            stopping_criteria=stopping_criteria,
            output_dir=output_dir,
            output_metadata=output_metadata,
            mode=mode,
        )
        self.name = self.instance.name
        self.encoding = self.output_metadata.get("encoding", "utf-8")
        result_dir_name = self.output_metadata.get("result_dir_name", "results")
        self.result_dir = self.working_dir / result_dir_name
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.prepare_saved_file_paths()

    # Start abstract methods

    def get_controller(self) -> FlowshopTardinessCpLnsController:
        return FlowshopTardinessCpLnsController(
            instance=self.instance,
            shared_param_dict=self.shared_param_dict,
            subroutine_flow=self.subroutine_flow,
            stopping_criteria=self.stopping_criteria,
        )

    def post_run_process(self) -> None:
        if self.mode in {RunMode.FULL_RUN, RunMode.RESUME}:
            self.save_files(self.encoding)

        self.from_files_save_analysis(self.encoding)

    # End abstract methods

    def _try_apply_resume(self) -> None:
        # Use resume data injected by the multi-instance runner
        if (
            self.resume_start_time_map is not None
            and self.resume_end_time_map is not None
            and self.resume_obj_store is not None
            and self.resume_summary_dict is not None
        ):
            logging.info(f"Applying injected resume data for instance '{self.name}'")
            self.ctrlr.obj_store = self.resume_obj_store
            init_report = FsSubroutineReport(
                elapsed_time=0.0,
                obj_value=self.resume_summary_dict.get("initObj", None),
                obj_bound=self.resume_summary_dict.get("initBound", None),
                is_init=True,
            )
            self.ctrlr.solution_manager.register(init_report, None)

            last_report = FsSubroutineReport(
                elapsed_time=self.resume_summary_dict.get("totalElapsedTime", 0.0),
                obj_value=self.resume_summary_dict.get("bestObj", None),
                obj_bound=self.resume_summary_dict.get("bestBound", None),
                is_init=False,
            )
            last_solution = FlowshopSchedule.from_stage_name_list(
                self.ctrlr.instance.stage_id_list
            )
            for key, start_time in self.resume_start_time_map.items():
                end_time = self.resume_end_time_map[key]
                j, i = key
                stage = last_solution.get_stage_by_name(i)
                operation = stage.add_operation(
                    FlowshopOperation(
                        job_name=j,
                        stage_name=i,
                        start=start_time,
                        end=end_time,
                    )
                )
                if operation is None:
                    raise RuntimeError(
                        f"Failed to schedule operation of job {j} at stage {i} during extraction "
                        f"with start time {start_time} and end time {end_time}."
                    )
            self.ctrlr.solution_manager.register(last_report, last_solution)

            # current datetime - last_report.elapsed_time
            virtual_dt = datetime.datetime.now() - datetime.timedelta(
                seconds=last_report.elapsed_time
            )
            self.ctrlr.timer.set_start_time(virtual_dt)

    def run(self):
        """
        Run the subroutine controller for the instance.

        - This method initializes the controller and runs it if the mode is FULL_RUN.
        - If the mode is POST_PROCESS_ONLY, it skips the controller run and directly
        calls the post_run_process method.
        """
        if self.mode == RunMode.RESUME:
            self.ctrlr = self.get_controller()
            self.ctrlr.set_working_dir(self.working_dir)
            self._try_apply_resume()
            self.ctrlr.run(flow_resume_idx=self.flow_resume_idx)

        return super().run()

    def prepare_saved_file_paths(self) -> None:
        self.summary_filename = (
            str(self.output_metadata.get("summary_filename_format", "{}_summary.csv"))
            .strip()
            .format(self.name)
        )
        self.summary_path = self.result_dir / self.summary_filename

        self.solution_filename_format = str(
            self.output_metadata.get("solution_filename_format", "{}_solution.yaml")
        ).strip()
        self.solution_path = self.result_dir / self.solution_filename_format.format(
            self.name
        )

        self.obj_log_filename_format = str(
            self.output_metadata.get("obj_log_filename_format", "{}_obj_log.yaml")
        ).strip()
        self.obj_log_path = self.result_dir / self.obj_log_filename_format.format(
            self.name
        )

    def save_files(self, encoding: str = "utf-8") -> None:
        self.save_summary(encoding=encoding)
        self.save_solution(encoding=encoding)
        self.save_obj_value_bound_store(encoding=encoding)

    def save_summary(self, encoding: str = "utf-8") -> None:
        stats = FsSubroutineReportStatistics(
            name=self.name,
            reports=[r.report for r in self.ctrlr.solution_manager.history],
            method_call_counts=self.ctrlr.method_call_counts,
        )
        summary = FsIoSummary(
            inputs=FsInputSummary(
                name=self.name,
                job_count=self.instance.job_count,
                stage_count=self.instance.stage_count,
                timelimit=self.stopping_criteria.timelimit,
            ),
            outputs=stats,
        )
        summary.save(self.summary_path, encoding=encoding)

    def save_solution(self, encoding: str = "utf-8") -> None:
        incumbent_solution = self.ctrlr.solution_manager.get_incumbent()
        if incumbent_solution:
            solution_dict = {
                START_TIME_MAP_KEY: tuple_to_pyyaml_key(
                    incumbent_solution.get_start_time_map()
                ),
                END_TIME_MAP_KEY: tuple_to_pyyaml_key(
                    incumbent_solution.get_end_time_map()
                ),
            }
            object_to_yaml(solution_dict, self.solution_path, encoding=encoding)

    def save_obj_value_bound_store(self, encoding: str = "utf-8") -> None:
        self.ctrlr.obj_store.save_yaml(self.obj_log_path, encoding=encoding)

    def from_files_save_analysis(self, encoding: str = "utf-8") -> None:
        if self.output_metadata.get("draw_gantt", False):
            self.from_files_draw_gantt_chart(encoding=encoding)
        if self.output_metadata.get("draw_progress_plot", False):
            self.from_files_draw_progress_plot(encoding=encoding)

    def from_files_draw_gantt_chart(self, encoding: str = "utf-8") -> None:
        """
        Draws Gantt charts from the saved solution files.
        This method looks for files matching the `solution_filename_format` in the working directory
        and generates Gantt charts based on the start and end times stored in the solution files.

        Args:
            encoding (str, optional): The encoding to use when reading files. Defaults to "utf-8".
        """
        result_gantt_filename_format = str(
            self.output_metadata.get("result_gantt_filename_format", "{}_gantt.png")
        ).strip()

        from concurrent_painter import draw_gantt_charts_from_solutions

        draw_gantt_charts_from_solutions(
            working_dir=self.working_dir,
            solution_filename_format=self.solution_filename_format,
            all_job_id_list=self.instance.job_id_list,
            result_gantt_filename_format=result_gantt_filename_format,
            encoding=encoding,
            painter_thread_cnt=self.output_metadata.get("painter_thread_cnt", 4),
        )

    def from_files_draw_progress_plot(self, encoding: str = "utf-8") -> None:
        """
        Draws a progress plot from the saved objective log files.
        This method looks for files matching the `obj_log_filename_format` in the working directory
        and generates a plot based on the objective value records stored in the log.

        Args:
            encoding (str, optional): The encoding to use when reading files. Defaults to "utf-8".
        """
        progress_plot_filename_format = str(
            self.output_metadata.get("progress_plot_filename_format", "{}_progress.png")
        ).strip()

        from concurrent_painter import draw_progress_plots_from_logs

        draw_progress_plots_from_logs(
            working_dir=self.working_dir,
            obj_log_filename_format=self.obj_log_filename_format,
            progress_plot_filename_format=progress_plot_filename_format,
            drop_first_values_percent=self.output_metadata.get(
                "drop_first_values_percent", 0.0
            ),
            encoding=encoding,
            painter_thread_cnt=self.output_metadata.get("painter_thread_cnt", 4),
        )


if __name__ == "__main__":
    import sys

    # Ensure repository root is on sys.path so imports using package layout work
    repo_root = Path(__file__).resolve().parents[0]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    vrm_path = repo_root / "resources" / "vrm" / "1.txt"
    assert vrm_path.exists(), f"VRM file not found: {vrm_path}"

    with vrm_path.open("r") as f:
        instance = FlowshopDuedateParameters.from_vrm_data("test", f)
        shared_param_dict = {"horizon": 100000}
        # subroutine_flow = DynamicDataObject.from_dict({"method": "initialize_by_edd"})
        subroutine_flow = DynamicDataObject.from_sequence(
            [
                {"method": "set_random_seed", "seed": 0},
                {"method": "initialize_by_edd"},
                {
                    "method": "solve_base_cp_model",
                    "computational_time": 12,
                    "solver_thread_cnt": 1,
                    "is_initial_solution": False,
                    "draw_gantt": False,
                },
            ]
        )
        stopping_criteria = StoppingCriteria.from_dict({"timelimit": 60})
        output_dir = repo_root / "Outputs/singleInsRunnerMain"
        output_metadata = {}

        runner = FsSingleInstanceRunner(
            instance,
            shared_param_dict,
            subroutine_flow,
            stopping_criteria,
            output_dir,
            output_metadata,
            RunMode.FULL_RUN,
        )
        runner.run()
