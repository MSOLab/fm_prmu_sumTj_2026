import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from routix import DynamicDataObject, StoppingCriteria
from routix.runner import MultiScenarioRunner
from routix.type_defs import RunMode
from schore.parameters_examples.shop.flow import (
    FlowshopDuedateParameters,
)
from xlsxwriter import Workbook
from xlsxwriter.worksheet import Worksheet

from flowshop_tardiness.report.dashboards import (
    write_post_run_dashboard_artifacts,
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
        # Save the aggregated raw summary
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
        """
        Creates a pivoted and styled dashboard for performance comparison,
        with a specific column order.
        """
        try:
            # 1. Pivot the raw data to get scenarios as columns
            best_obj_value_df = raw_summary_df.pivot_table(
                index="insName", columns="scenario", values="bestObj"
            ).reset_index()

            # 2. Merge with baseline data if available
            if self.baseline_df is not None and not self.baseline_df.empty:
                rename_map = {
                    self.baseline_instance_col: "insName",
                    self.baseline_obj_val_col: "baselineObjVal",
                    self.baseline_obj_bound_col: "baselineBound",
                }
                cols = [
                    self.baseline_instance_col,
                    self.baseline_obj_val_col,
                    self.baseline_obj_bound_col,
                ]
                baseline_subset = self.baseline_df.loc[:, cols].copy()

                # Ensure the merge key has the same dtype on both sides (use str for safety)
                baseline_subset[self.baseline_instance_col] = baseline_subset[
                    self.baseline_instance_col
                ].astype(str)
                best_obj_value_df["insName"] = best_obj_value_df[
                    "insName"
                ].astype(str)
                baseline_subset.rename(columns=rename_map, inplace=True)

                if baseline_subset.columns.duplicated().any():
                    dup = baseline_subset.columns[baseline_subset.columns.duplicated()]
                    logging.warning(f"Dropping duplicate baseline columns: {list(dup)}")
                    baseline_subset = baseline_subset.loc[
                        :, ~baseline_subset.columns.duplicated()
                    ]

                dashboard_df = pd.merge(
                    best_obj_value_df,
                    baseline_subset,
                    on="insName",
                    how="left",
                )
            else:
                logging.warning("Baseline data not available. Skipping merge.")
                dashboard_df = best_obj_value_df
                dashboard_df["baselineObjVal"] = None

            # 3. Calculate gaps for each scenario
            rpdv_col_name_format = self.base_output_metadata.get(
                "rpdv_col_name_format", self.relative_percentage_deviation_col_format
            )
            gap_col_name_format = self.base_output_metadata.get(
                "gap_col_name_format", self.gap_col_format
            )
            rpdf_col_name_format = self.base_output_metadata.get(
                "rpdf_col_name_format", self.relative_percentage_difference_col_format
            )
            scenarios = [
                col for col in best_obj_value_df.columns if col != "insName"
            ]

            has_baseline_val = (
                "baselineObjVal" in dashboard_df.columns
                and dashboard_df["baselineObjVal"].notna().any()
            )
            has_baseline_bound = (
                "baselineBound" in dashboard_df.columns
                and dashboard_df["baselineBound"].notna().any()
            )

            if has_baseline_val:
                bks = dashboard_df["baselineObjVal"]
                for scenario in scenarios:
                    sc = dashboard_df[scenario]
                    # dashboard_df[rpdv_col_name_format.format(scenario)] = case_ratio(
                    #     sc - bks, sc, both_zero_val=0, bks_zero_sc_pos_val=0.01
                    # )
                    rpdv_series = pd.Series(np.nan, index=sc.index, dtype="Float64")
                    # vectorized masks
                    mask_na = sc.isna() | bks.isna()
                    mask_both_zero = (bks == 0) & (sc == 0)
                    mask_bks_zero_sc_pos = (bks == 0) & (sc > 0)
                    mask_bks_zero_sc_neg = (bks == 0) & (sc < 0)
                    mask_else = ~(
                        mask_na
                        | mask_both_zero
                        | mask_bks_zero_sc_pos
                        | mask_bks_zero_sc_neg
                    )
                    # Assign values (use np.nan instead of pd.NA to satisfy type checker)
                    # rpdv_series.loc[mask_na] = np.nan  # already NaN by initialization
                    rpdv_series.loc[mask_both_zero] = 0.0
                    rpdv_series.loc[mask_bks_zero_sc_pos] = 0.01
                    rpdv_series.loc[mask_bks_zero_sc_neg] = -0.01
                    rpdv_series.loc[mask_else] = ((sc - bks) / sc)[mask_else].astype(
                        "Float64"
                    )
                    dashboard_df[rpdv_col_name_format.format(scenario)] = rpdv_series

            if has_baseline_bound:
                lb = dashboard_df["baselineBound"]
                for scenario in scenarios:
                    ub = dashboard_df[scenario]  # here "ub" is the scenario objective
                    dashboard_df[gap_col_name_format.format(scenario)] = case_ratio(
                        ub - lb, ub, both_zero_val=0
                    )

            if has_baseline_val:
                bks = dashboard_df["baselineObjVal"]
                for scenario in scenarios:
                    sc = dashboard_df[scenario]
                    dashboard_df[rpdf_col_name_format.format(scenario)] = case_ratio(
                        sc - bks, (sc + bks) / 2, both_zero_val=0
                    )

            # 4. Define the desired column order
            ordered_columns = ["insName"]
            obj_val_cols = [col for col in scenarios]
            baseline_obj_val_col = (
                ["baselineObjVal"] if "baselineObjVal" in dashboard_df.columns else []
            )
            baseline_bound_col = (
                ["baselineBound"] if "baselineBound" in dashboard_df.columns else []
            )
            relative_percentage_deviation_cols = [
                rpdv_col_name_format.format(scenario)
                for scenario in scenarios
                if rpdv_col_name_format.format(scenario) in dashboard_df
            ]
            gap_cols = [
                gap_col_name_format.format(scenario)
                for scenario in scenarios
                if gap_col_name_format.format(scenario) in dashboard_df
            ]
            relative_percentage_difference_cols = [
                rpdf_col_name_format.format(scenario)
                for scenario in scenarios
                if rpdf_col_name_format.format(scenario) in dashboard_df
            ]

            # Combine lists in the desired order
            final_column_order = (
                ordered_columns
                + obj_val_cols
                + baseline_obj_val_col
                + baseline_bound_col
                + relative_percentage_deviation_cols
                + gap_cols
                + relative_percentage_difference_cols
            )

            # Reorder the DataFrame
            final_dashboard = dashboard_df[final_column_order]

            # 5. Add summary statistics at the bottom

            summary_rows: list[dict[str, Any]] = []
            for stat_name, stat_func in self.stat_name_func_pairs:
                row: dict[str, Any] = {"insName": stat_name}
                for col in final_dashboard.columns:
                    if col != "insName":
                        if pd.api.types.is_numeric_dtype(final_dashboard[col]):
                            row[col] = getattr(final_dashboard[col], stat_func)()
                summary_rows.append(row)

            summary_df = pd.DataFrame(summary_rows)
            final_dashboard = pd.concat(
                [final_dashboard, summary_df], ignore_index=True
            )

            return final_dashboard

        except Exception as e:
            logging.error(f"Failed to create dashboard: {e}", exc_info=True)
            return pd.DataFrame()

    def create_info_sheet(self) -> pd.DataFrame:
        """Creates a DataFrame with detailed information about each scenario."""
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
        return pd.DataFrame(info_data)

    def write_excel_report(
        self,
        path: Path,
        dashboard_df: pd.DataFrame,
        raw_summary_df: pd.DataFrame,
        info_df: pd.DataFrame,
        baseline_df: pd.DataFrame | None,
    ):
        """
        Writes the DataFrames to a styled Excel file using the xlsxwriter engine
        for robust formatting and auto-adjusted column widths.

        Args:
            path (Path): Path to save the Excel report.
            dashboard_df (pd.DataFrame): DataFrame containing the dashboard data.
            raw_summary_df (pd.DataFrame): DataFrame containing the raw summary data.
            info_df (pd.DataFrame): DataFrame containing scenario information.
            baseline_df (pd.DataFrame | None, optional): DataFrame containing baseline data, if available.
        """
        try:
            with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
                # --- Write sheets in the desired order ---
                workbook: Workbook = writer.book  # type: ignore
                # --- Create formats ---
                percent_format = workbook.add_format({"num_format": "0.00%"})

                # 1. Best objective Dashboard
                sheet_name = "BestObjDashboard"
                if not dashboard_df.empty:
                    # Create the multi-level header
                    header = []
                    for col in dashboard_df.columns:
                        if col.startswith(
                            self.relative_percentage_deviation_col_prefix
                        ):
                            header.append(
                                (
                                    "Relative % deviation",
                                    col.replace(
                                        self.relative_percentage_deviation_col_prefix,
                                        "",
                                    ),
                                )
                            )
                        elif col.startswith(self.gap_col_prefix):
                            header.append(
                                ("SolverGap", col.replace(self.gap_col_prefix, ""))
                            )
                        elif col.startswith(
                            self.relative_percentage_difference_col_prefix
                        ):
                            header.append(
                                (
                                    "Relative % difference",
                                    col.replace(
                                        self.relative_percentage_difference_col_prefix,
                                        "",
                                    ),
                                )
                            )
                        elif col == "insName":
                            header.append(("", "insId"))
                        elif col == "baselineObjVal":
                            header.append(("", "baselineObjVal"))
                        elif col == "baselineBound":
                            header.append(("", "baselineBound"))
                        else:
                            header.append(("ObjVal", col))
                    dashboard_df.columns = pd.MultiIndex.from_tuples(header)

                    dashboard_df.to_excel(writer, sheet_name=sheet_name, index=True)

                    worksheet: Worksheet = writer.sheets[sheet_name]
                    data_start_row = 3
                    data_end_row = len(dashboard_df) + data_start_row - 1

                    # --- Apply formatting and set column widths ---

                    # relDiff first_col and last_col
                    rel_diff_first_col: int | None = (
                        None  # Placeholder for first column
                    )
                    rel_diff_last_col = 0

                    # +1 for the index column
                    for col_idx, col_name in enumerate(dashboard_df.columns, 1):
                        # Calculate max width
                        header_l1 = str(col_name[0])
                        header_l2 = str(col_name[1])
                        max_len = (
                            max(
                                len(header_l1),
                                len(header_l2),
                                dashboard_df[col_name].astype(str).map(len).max(),
                            )
                            + 2
                        )  # Add padding

                        worksheet.set_column(col_idx, col_idx, width=max_len)

                        if col_name[0] == "relDiff between baseline":
                            if rel_diff_first_col is None:
                                rel_diff_first_col = col_idx
                            if rel_diff_last_col < col_idx:
                                rel_diff_last_col = col_idx
                            worksheet.set_column(
                                col_idx, col_idx, max_len, percent_format
                            )

                    if rel_diff_first_col is not None:
                        worksheet.conditional_format(
                            data_start_row,
                            rel_diff_first_col,
                            data_end_row,
                            rel_diff_last_col,
                            {
                                "type": "data_bar",
                                "bar_color": "#638EC6",
                                "bar_negative_color": "#F8696B",
                                "bar_axis_position": "middle",
                            },
                        )

                # 2. Scenario_Info
                info_df.to_excel(writer, sheet_name="Scenario_Info", index=False)
                worksheet = writer.sheets["Scenario_Info"]
                for col_idx, col_name in enumerate(info_df.columns):
                    max_len = (
                        max(
                            len(str(col_name)),
                            info_df[col_name].astype(str).map(len).max(),
                        )
                        + 2
                    )
                    if col_name in {"Subroutine Flow", "Stopping Criteria"}:
                        worksheet.set_column(col_idx, col_idx, options={"hidden": True})
                    else:
                        worksheet.set_column(col_idx, col_idx, width=max_len)

                # 3. Raw_Summary
                raw_summary_df.to_excel(writer, sheet_name="Raw_Summary", index=False)
                worksheet = writer.sheets["Raw_Summary"]
                for col_idx, col_name in enumerate(raw_summary_df.columns):
                    max_len = (
                        max(
                            len(str(col_name)),
                            raw_summary_df[col_name].astype(str).map(len).max(),
                        )
                        + 2
                    )
                    if col_name == "methodCallCounts":
                        worksheet.set_column(col_idx, col_idx, options={"hidden": True})
                    else:
                        worksheet.set_column(col_idx, col_idx, width=max_len)
                    if col_name == "improvementRatio":
                        worksheet.set_column(
                            col_idx, col_idx, width=max_len, cell_format=percent_format
                        )

                # 4. Baseline_Data
                if baseline_df is not None and not baseline_df.empty:
                    baseline_df.to_excel(
                        writer, sheet_name="Baseline_Data", index=False
                    )
                    worksheet = writer.sheets["Baseline_Data"]
                    for col_idx, col_name in enumerate(baseline_df.columns):
                        max_len = (
                            max(
                                len(str(col_name)),
                                baseline_df[col_name].astype(str).map(len).max(),
                            )
                            + 2
                        )
                        worksheet.set_column(col_idx, col_idx, width=max_len)
                        if col_name in {"Gap"}:
                            worksheet.set_column(
                                col_idx,
                                col_idx,
                                width=max_len,
                                cell_format=percent_format,
                            )

            logging.info(f"Successfully generated Excel report at: {path}")
        except Exception as e:
            logging.error(f"Failed to write Excel report: {e}", exc_info=True)


def case_ratio(
    dividend: pd.Series,
    divisor: pd.Series,
    both_zero_val: float | None = None,
    divisor_zero_dividend_positive_val: float | None = None,
    divisor_zero_dividend_negative_val: float | None = None,
) -> pd.Series:
    """
    Compute element-wise ratio between two pandas Series with special case handling.

    This function divides `dividend` by `divisor` element-wise and handles
    edge cases where the divisor is zero:

    - If both dividend and divisor are zero, returns `both_zero_val`.
    - If divisor is zero and dividend is positive, returns
      `divisor_zero_dividend_positive_val`.
    - If divisor is zero and dividend is negative, returns
      `divisor_zero_dividend_negative_val`.
    - Otherwise, computes the standard division `dividend / divisor`.

    If any of the special-case values are set to `None` (default),
    the corresponding result will be `NaN`.

    Args:
        dividend (pd.Series): Numerator values for the ratio.
        divisor (pd.Series): Denominator values for the ratio.
        both_zero_val (float | None, optional): Value to assign when both dividend and divisor are zero.
            Defaults to None (which results in NaN).
        divisor_zero_dividend_positive_val (float | None, optional): Value to assign when divisor is zero and dividend is positive.
            Defaults to None (which results in NaN).
        divisor_zero_dividend_negative_val (float | None, optional): Value to assign when divisor is zero and dividend is negative.
            Defaults to None (which results in NaN).

    Returns:
        pd.Series: Element-wise ratio between dividend and divisor with special case handling.
    """
    assert dividend.shape == divisor.shape, (
        "dividend and divisor must have the same shape"
    )

    # Convert to numeric float arrays for safe NumPy operations (pd.NA/non-numeric -> NaN)
    a = pd.to_numeric(dividend, errors="coerce").to_numpy(dtype="float64", copy=False)
    b = pd.to_numeric(divisor, errors="coerce").to_numpy(dtype="float64", copy=False)

    # Output buffer (initialized to all NaN)
    out = np.full(a.shape, np.nan, dtype="float64")

    m_div = b != 0
    np.divide(a, b, out=out, where=m_div)

    # Special cases where b == 0
    m_zero = ~m_div
    m0 = m_zero & (a == 0)  # 0/0
    m1 = m_zero & (a > 0)  # +/0
    m2 = m_zero & (a < 0)  # -/0

    # Replace None with NaN to avoid dtype pollution
    v0 = np.nan if both_zero_val is None else both_zero_val
    v1p = (
        np.nan
        if divisor_zero_dividend_positive_val is None
        else divisor_zero_dividend_positive_val
    )
    v1n = (
        np.nan
        if divisor_zero_dividend_negative_val is None
        else divisor_zero_dividend_negative_val
    )

    out[m0] = v0
    out[m1] = v1p
    out[m2] = v1n

    return pd.Series(out, index=dividend.index, dtype="Float64")


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
