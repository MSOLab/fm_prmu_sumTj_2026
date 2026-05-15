from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xlsxwriter import Workbook
from xlsxwriter.worksheet import Worksheet

from .obj_log_trim import apply_timelimit_trim

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RpdColFormats:
    deviation_prefix: str = "RPDv_"
    deviation_format: str = "RPDv_{}"
    gap_prefix: str = "Gap_"
    gap_format: str = "Gap_{}"
    difference_prefix: str = "RPDf_"
    difference_format: str = "RPDf_{}"


DEFAULT_RPD_FORMATS = RpdColFormats()
DEFAULT_STAT_PAIRS: list[tuple[str, str]] = [
    ("Average", "mean"),
    ("Max", "max"),
    ("Min", "min"),
]


def case_ratio(
    dividend: pd.Series,
    divisor: pd.Series,
    both_zero_val: float | None = None,
    divisor_zero_dividend_positive_val: float | None = None,
    divisor_zero_dividend_negative_val: float | None = None,
) -> pd.Series:
    assert dividend.shape == divisor.shape, (
        "dividend and divisor must have the same shape"
    )

    a = pd.to_numeric(dividend, errors="coerce").to_numpy(dtype="float64", copy=False)
    b = pd.to_numeric(divisor, errors="coerce").to_numpy(dtype="float64", copy=False)

    out = np.full(a.shape, np.nan, dtype="float64")
    m_div = b != 0
    np.divide(a, b, out=out, where=m_div)

    m_zero = ~m_div
    m0 = m_zero & (a == 0)
    m1 = m_zero & (a > 0)
    m2 = m_zero & (a < 0)

    v0 = np.nan if both_zero_val is None else both_zero_val
    v1p = np.nan if divisor_zero_dividend_positive_val is None else divisor_zero_dividend_positive_val
    v1n = np.nan if divisor_zero_dividend_negative_val is None else divisor_zero_dividend_negative_val

    out[m0] = v0
    out[m1] = v1p
    out[m2] = v1n

    return pd.Series(out, index=dividend.index, dtype="Float64")


def build_dashboard_df(
    raw_summary_df: pd.DataFrame,
    *,
    baseline_df: pd.DataFrame | None,
    baseline_instance_col: str = "Instance",
    baseline_obj_val_col: str = "BKS",
    baseline_obj_bound_col: str = "LB",
    base_output_metadata: dict[str, Any] | None = None,
    stat_pairs: list[tuple[str, str]] = DEFAULT_STAT_PAIRS,
    rpd_col_formats: RpdColFormats = DEFAULT_RPD_FORMATS,
) -> pd.DataFrame:
    try:
        best_obj_value_df = raw_summary_df.pivot_table(
            index="insName", columns="scenario", values="bestObj"
        ).reset_index()

        if baseline_df is not None and not baseline_df.empty:
            rename_map = {
                baseline_instance_col: "insName",
                baseline_obj_val_col: "baselineObjVal",
                baseline_obj_bound_col: "baselineBound",
            }
            cols = [
                baseline_instance_col,
                baseline_obj_val_col,
                baseline_obj_bound_col,
            ]
            baseline_subset = baseline_df.loc[:, cols].copy()

            baseline_subset[baseline_instance_col] = baseline_subset[
                baseline_instance_col
            ].astype(str)
            best_obj_value_df["insName"] = best_obj_value_df["insName"].astype(str)
            baseline_subset.rename(columns=rename_map, inplace=True)

            if baseline_subset.columns.duplicated().any():
                dup = baseline_subset.columns[baseline_subset.columns.duplicated()]
                logger.warning(f"Dropping duplicate baseline columns: {list(dup)}")
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
            logger.warning("Baseline data not available. Skipping merge.")
            dashboard_df = best_obj_value_df
            dashboard_df["baselineObjVal"] = None

        metadata = base_output_metadata or {}
        rpdv_col_name_format = metadata.get(
            "rpdv_col_name_format", rpd_col_formats.deviation_format
        )
        gap_col_name_format = metadata.get(
            "gap_col_name_format", rpd_col_formats.gap_format
        )
        rpdf_col_name_format = metadata.get(
            "rpdf_col_name_format", rpd_col_formats.difference_format
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
                rpdv_series = pd.Series(np.nan, index=sc.index, dtype="Float64")
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
                rpdv_series.loc[mask_both_zero] = 0.0
                rpdv_series.loc[mask_bks_zero_sc_pos] = 0.01
                rpdv_series.loc[mask_bks_zero_sc_neg] = -0.01
                rpdv_series.loc[mask_else] = ((sc - bks) / sc)[mask_else].astype("Float64")
                dashboard_df[rpdv_col_name_format.format(scenario)] = rpdv_series

        if has_baseline_bound:
            lb = dashboard_df["baselineBound"]
            for scenario in scenarios:
                ub = dashboard_df[scenario]
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

        ordered_columns = ["insName"]
        obj_val_cols = [col for col in scenarios]
        baseline_obj_val_col_list = (
            ["baselineObjVal"] if "baselineObjVal" in dashboard_df.columns else []
        )
        baseline_bound_col_list = (
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

        final_column_order = (
            ordered_columns
            + obj_val_cols
            + baseline_obj_val_col_list
            + baseline_bound_col_list
            + relative_percentage_deviation_cols
            + gap_cols
            + relative_percentage_difference_cols
        )

        final_dashboard = dashboard_df[final_column_order]

        summary_rows: list[dict[str, Any]] = []
        for stat_name, stat_func in stat_pairs:
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
        logger.error(f"Failed to create dashboard: {e}", exc_info=True)
        return pd.DataFrame()


def build_info_df(scenario_records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(scenario_records)


def write_multi_scenario_excel_report(
    path: Path,
    *,
    dashboard_df: pd.DataFrame,
    raw_summary_df: pd.DataFrame,
    info_df: pd.DataFrame,
    baseline_df: pd.DataFrame | None,
    rpd_col_formats: RpdColFormats = DEFAULT_RPD_FORMATS,
):
    try:
        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            workbook: Workbook = writer.book
            percent_format = workbook.add_format({"num_format": "0.00%"})

            # 1. Best objective Dashboard
            sheet_name = "BestObjDashboard"
            if not dashboard_df.empty:
                header = []
                for col in dashboard_df.columns:
                    if col.startswith(rpd_col_formats.deviation_prefix):
                        header.append(
                            (
                                "Relative % deviation",
                                col.replace(rpd_col_formats.deviation_prefix, ""),
                            )
                        )
                    elif col.startswith(rpd_col_formats.gap_prefix):
                        header.append(
                            ("SolverGap", col.replace(rpd_col_formats.gap_prefix, ""))
                        )
                    elif col.startswith(rpd_col_formats.difference_prefix):
                        header.append(
                            (
                                "Relative % difference",
                                col.replace(rpd_col_formats.difference_prefix, ""),
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

                rel_diff_first_col: int | None = None
                rel_diff_last_col = 0

                for col_idx, col_name in enumerate(dashboard_df.columns, 1):
                    header_l1 = str(col_name[0])
                    header_l2 = str(col_name[1])
                    max_len = (
                        max(
                            len(header_l1),
                            len(header_l2),
                            dashboard_df[col_name].astype(str).map(len).max(),
                        )
                        + 2
                    )
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

        logger.info(f"Successfully generated Excel report at: {path}")
    except Exception as e:
        logger.error(f"Failed to write Excel report: {e}", exc_info=True)


def aggregate_scenario_summaries(
    scenario_dirs: list[Path],
    *,
    out_dir: Path,
) -> pd.DataFrame:
    all_summary_dfs = []
    for scenario_dir in scenario_dirs:
        summary_path = scenario_dir / "multi_instance_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"multi_instance_summary.csv not found in {scenario_dir}"
            )
        df = pd.read_csv(summary_path)
        df["scenario"] = scenario_dir.name
        all_summary_dfs.append(df)

    raw_summary_df = pd.concat(all_summary_dfs, ignore_index=True)
    raw_summary_df.to_csv(
        out_dir / "all_scenarios_summary_endpoint.csv", index=False
    )

    trimmed_df = apply_timelimit_trim(raw_summary_df, out_dir)
    trimmed_df.to_csv(out_dir / "all_scenarios_summary.csv", index=False)
    logger.info(f"Aggregated summary saved to {out_dir}")

    return trimmed_df
