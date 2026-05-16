import logging
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from routix import DynamicDataObject, StoppingCriteria
from routix.runner import MultiScenarioRunner
from routix.type_defs import RunMode
from schore.parameters_examples.shop.flow import (
    FlowshopDuedateParameters,
)

from flowshop_tardiness.report.dashboards import (
    apply_timelimit_trim,
    write_post_run_dashboard_artifacts,
)
from flowshop_tardiness.report.dashboards.multi_scenario_report import (
    DEFAULT_RPD_FORMATS,
    build_dashboard_df,
    build_info_df,
    write_multi_scenario_excel_report,
)
from fs_config import BaselineColumnMapping
from fs_multi_instance_runner import FsMultiInstanceRunner
from fs_single_instance_runner import FsSingleInstanceRunner
from output_filenames import OutputFilenames


class FsMultiScenarioRunner(
    MultiScenarioRunner[
        FlowshopDuedateParameters, FsSingleInstanceRunner, FsMultiInstanceRunner
    ]
):
    stat_name_func_pairs = [("Average", "mean"), ("Max", "max"), ("Min", "min")]

    relative_percentage_deviation_col_prefix = "RPDv_"
    relative_percentage_deviation_col_format = "RPDv_{}"
    gap_col_prefix = "Gap_"
    gap_col_format = "Gap_{}"
    relative_percentage_difference_col_prefix = "RPDf_"
    relative_percentage_difference_col_format = "RPDf_{}"

    def __init__(
        self,
        m_i_runner_class: type[FsMultiInstanceRunner],
        s_i_runner_class: type[FsSingleInstanceRunner],
        instances: Sequence[FlowshopDuedateParameters],
        shared_param_dict: dict,
        scenario_configs: Sequence[dict[str, Any]],
        output_dir: Path,
        base_output_metadata: dict[str, Any],
        mode: RunMode = RunMode.FULL_RUN,
        instance_worker_cnt: int = 1,
    ):
        super().__init__(
            m_i_runner_class,
            s_i_runner_class,
            instances,
            shared_param_dict,
            scenario_configs,
            output_dir,
            base_output_metadata,
            mode=mode,
            instance_worker_cnt=instance_worker_cnt,
        )
        self.baseline_df: pd.DataFrame | None = None
        """DataFrame containing baseline results for comparison in the report."""

        if self.mode in {RunMode.FULL_RUN, RunMode.RESUME}:
            # --- Save scenario-specific config files for reproducibility ---
            for i, scenario_config in enumerate(self.scenario_configs):
                subroutine_flow: DynamicDataObject | None = scenario_config.get(
                    "subroutine_flow"
                )
                stopping_criteria: StoppingCriteria | None = scenario_config.get(
                    "stopping_criteria"
                )
                if subroutine_flow is None or stopping_criteria is None:
                    continue
                # Use a specific output subdir from config, or create a default one
                scenario_output_dir = self.output_dir / f"scenario_{i + 1}"
                if "output_subdir" in scenario_config:
                    scenario_output_dir = self.output_dir / str(
                        scenario_config["output_subdir"]
                    )
                scenario_output_dir.mkdir(parents=True, exist_ok=True)
                DynamicDataObject.safe_save_yaml(
                    subroutine_flow,
                    scenario_output_dir / OutputFilenames.SUBROUTINE_FLOW_CACHE_FN,
                )
                DynamicDataObject.safe_save_yaml(
                    stopping_criteria,
                    scenario_output_dir / OutputFilenames.STOPPING_CRITERIA_CACHE_FN,
                )

    # Start abstract methods

    def post_run_process(self):
        """
        Aggregates results from all scenarios and generates a comprehensive Excel report
        that includes a comparative dashboard, raw data, and scenario information.
        """
        # 1. Aggregate all scenario summaries
        all_summary_dfs = []
        for i, runner in enumerate(self.runners):
            summary_path = runner.working_dir / "multi_instance_summary.csv"
            if summary_path.exists():
                df = pd.read_csv(summary_path)
                scenario_name = self.scenario_configs[i].get(
                    "output_subdir", f"scenario_{i + 1}"
                )
                df["scenario"] = str(scenario_name)
                all_summary_dfs.append(df)
            else:
                logging.warning(
                    f"Summary file not found for scenario {i + 1} at {summary_path}"
                )

        if not all_summary_dfs:
            logging.warning("No scenario summaries found to aggregate.")
            return

        raw_summary_df = pd.concat(all_summary_dfs, ignore_index=True)
        # Save the raw, pre-trim aggregate for traceability.
        raw_summary_df.to_csv(
            self.output_dir / "all_scenarios_summary_endpoint.csv", index=False
        )
        # Trim bestObj/bestBound/totalElapsedTime to the configured timelimit so
        # downstream dashboards reflect the deadline-truncated view. Originals
        # are preserved as ``*_endpoint`` columns on the resulting frame.
        raw_summary_df = apply_timelimit_trim(raw_summary_df, self.output_dir)
        raw_summary_df.to_csv(
            self.output_dir / "all_scenarios_summary.csv", index=False
        )
        logging.info(f"Aggregated summary saved to {self.output_dir}")

        # 2. Create the comparison dashboard
        dashboard_df = self.create_dashboard(raw_summary_df)

        # 3. Create the scenario info sheet
        info_df = self.create_info_sheet()

        # 4. Write all DataFrames to a styled Excel report
        excel_report_path = self.output_dir / "multi_scenario_report.xlsx"
        self.write_excel_report(
            excel_report_path,
            dashboard_df=dashboard_df,
            raw_summary_df=raw_summary_df,
            info_df=info_df,
            baseline_df=self.baseline_df,
        )

        # 5. Write the interactive HTML dashboards (RPDf pivot + multi-scenario
        # subroutine flow comparison). Ported from ffc_ddw_sum_et — see
        # flowshop_tardiness/report/dashboards/.
        try:
            instance_col = getattr(self, "baseline_instance_col", "Instance")
            obj_val_col = getattr(self, "baseline_obj_val_col", "BKS")
            obj_bound_col = getattr(self, "baseline_obj_bound_col", "LB")
            write_post_run_dashboard_artifacts(
                self.output_dir,
                baseline_df=self.baseline_df,
                baseline_instance_col=instance_col,
                baseline_obj_val_col=obj_val_col,
                baseline_obj_bound_col=obj_bound_col,
            )
        except Exception:
            logging.exception("Failed to write post-run HTML dashboards")

    # End abstract methods

    def set_baseline_df(
        self, baseline_csv_path: Path, column_mapping: BaselineColumnMapping
    ):
        """
        Sets the baseline DataFrame for comparison in the report.
        This DataFrame should contain the baseline results for the scenarios.
        """
        if baseline_csv_path.exists():
            self.baseline_df = pd.read_csv(baseline_csv_path)
            logging.info(f"Baseline DataFrame loaded from {baseline_csv_path}")
            self.baseline_instance_col = column_mapping.instance
            self.baseline_obj_val_col = column_mapping.obj_val
            self.baseline_obj_bound_col = column_mapping.obj_bound
        else:
            logging.warning(f"Baseline CSV file not found at {baseline_csv_path}")
            self.baseline_df = pd.DataFrame()

    def create_dashboard(self, raw_summary_df: pd.DataFrame) -> pd.DataFrame:
        return build_dashboard_df(
            raw_summary_df,
            baseline_df=self.baseline_df,
            baseline_instance_col=getattr(self, "baseline_instance_col", "Instance"),
            baseline_obj_val_col=getattr(self, "baseline_obj_val_col", "BKS"),
            baseline_obj_bound_col=getattr(self, "baseline_obj_bound_col", "LB"),
            base_output_metadata=self.base_output_metadata,
            stat_pairs=self.stat_name_func_pairs,
            rpd_col_formats=DEFAULT_RPD_FORMATS,
        )

    def create_info_sheet(self) -> pd.DataFrame:
        info_data = []
        for i, config in enumerate(self.scenario_configs):
            scenario_name = config.get("output_subdir", f"scenario_{i + 1}")
            info_data.append(
                {
                    "Scenario": str(scenario_name),
                    "Subroutine Flow": str(config.get("subroutine_flow")),
                    "Stopping Criteria": str(config.get("stopping_criteria")),
                    "Description": config.get("description", ""),
                }
            )
        return build_info_df(info_data)

    def write_excel_report(
        self,
        path: Path,
        dashboard_df: pd.DataFrame,
        raw_summary_df: pd.DataFrame,
        info_df: pd.DataFrame,
        baseline_df: pd.DataFrame | None,
    ):
        write_multi_scenario_excel_report(
            path,
            dashboard_df=dashboard_df,
            raw_summary_df=raw_summary_df,
            info_df=info_df,
            baseline_df=baseline_df,
            rpd_col_formats=DEFAULT_RPD_FORMATS,
        )


if __name__ == "__main__":
    import sys

    from routix import DynamicDataObject, StoppingCriteria
    from routix.type_defs import RunMode

    # Ensure repository root is on sys.path so imports using package layout work
    repo_root = Path(__file__).resolve().parents[0]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    instances = []
    for i in range(1, 2):
        vrm_path = repo_root / "resources" / "vrm" / f"{i}.txt"
        assert vrm_path.exists(), f"VRM file not found: {vrm_path}"
        with vrm_path.open("r") as f:
            instance = FlowshopDuedateParameters.from_vrm_data(str(i), f)
            instances.append(instance)

    shared_param_dict = {"horizon": 100000}

    subroutine_flow_list = []
    for i in range(9, 14):
        subroutine_flow_list.append(
            DynamicDataObject.from_sequence(
                [
                    {"method": "set_random_seed", "seed": i},
                    {"method": "initialize_by_edd"},
                    {
                        "method": "solve_base_cp_model",
                        "computational_time": 20,
                        "solver_thread_cnt": 1,
                        "is_initial_solution": False,
                        "draw_gantt": False,
                    },
                ]
            )
        )

    stopping_criteria = StoppingCriteria.from_dict({"timelimit": 60})

    scenario_configs = []
    for idx, subroutine_flow in enumerate(subroutine_flow_list):
        scenario_configs.append(
            {
                "subroutine_flow": subroutine_flow,
                "stopping_criteria": stopping_criteria,
                "output_subdir": f"scenario_{idx + 9}",
                "description": f"Seed={idx}, EDD init, 20s CP solve, 1 thread",
            }
        )

    output_dir = repo_root / "Outputs/multiScenarioRunnerMain"

    output_metadata: dict[str, Any] = {}

    run_mode = RunMode.FULL_RUN

    worker_cnt = 1

    # --- Create and run the multi-scenario runner ---
    multi_scenario_runner = FsMultiScenarioRunner(
        m_i_runner_class=FsMultiInstanceRunner,
        s_i_runner_class=FsSingleInstanceRunner,
        instances=instances,
        shared_param_dict=shared_param_dict,
        scenario_configs=scenario_configs,
        output_dir=output_dir,
        base_output_metadata=output_metadata,
        mode=run_mode,
        instance_worker_cnt=worker_cnt,
    )
    multi_scenario_runner.set_baseline_df(
        repo_root / "resources/vrm_ref/2023IJoC.csv",
        BaselineColumnMapping(instance="Instance", obj_val="BKS", obj_bound="LB"),
    )
    logging.info("Starting Multi-Scenario Runner.")
    multi_scenario_runner.run()
