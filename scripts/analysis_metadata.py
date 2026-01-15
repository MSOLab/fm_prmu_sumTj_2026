import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalysisMetadata:
    name: str
    result_dir_path_str: str
    reactive_loop_report_rel_path: str = "6-run_reactive_loop_report.csv"
    obj_log_filename_format: str = "results/{instance_id}_obj_log.yaml"
    controller_log_rel_path: str = "subroutine_controller.log"
    obj_log_top_level_methods: tuple[tuple[str, str], ...] = (
        ("1-set_random_seed", "set_random_seed"),
        ("2-compute_preemptive_last_stage_lb", "compute_preemptive_last_stage_lb"),
        ("3-initialize_by_edd", "initialize_by_edd"),
        ("4-initialize_by_nehms", "initialize_by_nehms"),
        ("5-set_cp_model_as_base_cp_model", "set_cp_model_as_base_cp_model"),
        ("6-pw_cp", "pw_cp"),
        ("7-repeat_while_improvement", "repeat_while_improvement"),
        ("8-solve_base_cp_model", "solve_base_cp_model"),
    )
    reactive_loop_report_required_cols: frozenset[str] = frozenset(
        {
            "iterCount",
            "rho",
            "timelimit",
            "subroutineName",
            "isImproved",
        }
    )

    def get_analysis_dir_path(self) -> Path:
        expanded = os.path.expandvars(self.result_dir_path_str)
        return Path(expanded).expanduser()

    def assert_reactive_loop_report_columns(self, columns: set[str]) -> None:
        missing = set(self.reactive_loop_report_required_cols) - columns
        if missing:
            raise ValueError(
                f"Missing columns in reactive loop report for {self.name}: {missing}"
            )
