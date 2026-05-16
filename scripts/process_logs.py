import ast
import json
import logging
import re
from pathlib import Path

import pandas as pd
import yaml

from flowshop_tardiness.io_solution import OBJ_LOG_FN_FORMAT, RESULT_DIR_NAME as DEFAULT_RESULTS_DIR

# Default constants
DEFAULT_CONTROLLER_LOG_NAME = "subroutine_controller.log"


def _read_instance_timelimit(instance_dir: Path, instance_id: str) -> float | None:
    """Load ``timelimit`` from ``results/<ins>_summary.csv`` if present.

    The per-instance summary is written by ``FsSingleInstanceRunner`` after
    the run, so it's available by the time ``process_scenario`` is invoked.
    """
    summary_path = instance_dir / DEFAULT_RESULTS_DIR / f"{instance_id}_summary.csv"
    if not summary_path.exists():
        return None
    try:
        df = pd.read_csv(summary_path)
    except Exception as e:
        logging.warning(f"Failed to read summary {summary_path}: {e}")
        return None
    if df.empty or "timelimit" not in df.columns:
        return None
    val = df["timelimit"].iloc[0]
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _last_obj_value_at_or_before(
    obj_data: dict, threshold_sec: float
) -> float | None:
    """Latest numeric obj value whose timestamp key is ``<= threshold_sec``."""
    best_t = float("-inf")
    best_v: float | None = None
    for k, v in obj_data.items():
        try:
            t = float(k)
        except (TypeError, ValueError):
            continue
        if t > threshold_sec or t <= best_t:
            continue
        if _is_missing_obj_value(v):
            continue
        try:
            best_v = float(v)
            best_t = t
        except (TypeError, ValueError):
            continue
    return best_v


def parse_controller_log(log_path: Path) -> list[dict]:
    records = []
    pattern = re.compile(r"INFO - ({'method':.*})")

    if not log_path.exists():
        logging.warning(f"Log file not found: {log_path}")
        return []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                try:
                    data = ast.literal_eval(match.group(1))
                    if (
                        isinstance(data, dict)
                        and "method" in data
                        and "start_sec" in data
                        and "elapsed_sec" in data
                    ):
                        data["start_sec"] = float(data["start_sec"])
                        data["elapsed_sec"] = float(data["elapsed_sec"])
                        data["end_sec"] = data["start_sec"] + data["elapsed_sec"]
                        records.append(data)
                except (ValueError, SyntaxError, TypeError):
                    continue
    return records


def parse_obj_log(yaml_path: Path) -> dict:
    if not yaml_path.exists():
        logging.warning(f"Obj log file not found: {yaml_path}")
        return {"data": {}, "notes": {}}

    with open(yaml_path, "r", encoding="utf-8") as f:
        try:
            content = yaml.safe_load(f)
            # The structure is obj_value: { data: {}, notes: {} }
            if "obj_value" in content:
                return content["obj_value"]
            return {"data": {}, "notes": {}}
        except Exception as e:
            logging.warning(f"Failed to read yaml {yaml_path}: {e}")
            return {"data": {}, "notes": {}}


def get_obj_value_for_method(
    method_prefix: str,
    method_name: str,
    obj_data: dict,
    obj_notes: dict,
    prev_obj_value,
):
    relevant_times = []
    # Prefix match (e.g., "6-" matches "6-pw_cp")
    for time_str, note in obj_notes.items():
        if note.startswith(method_prefix):
            relevant_times.append(time_str)

    if not relevant_times:
        return None

    try:
        max_time_str = max(relevant_times, key=float)
    except ValueError:
        max_time_str = max(relevant_times)

    val = obj_data.get(max_time_str)

    if _is_missing_obj_value(val):
        return prev_obj_value

    return val


def _is_missing_obj_value(val):
    if val is None:
        return True
    if isinstance(val, float) and (pd.isna(val) or val == float("nan")):
        return True
    s_val = str(val).strip()
    if s_val.lower() == "nan" or s_val == "":
        return True
    return False


def get_methods_from_flow(scenario_dir: Path) -> list[tuple[str, str]]:
    flow_path = scenario_dir / "subroutine_flow.yaml"
    if not flow_path.exists():
        logging.warning(f"subroutine_flow.yaml not found in {scenario_dir}.")
        return []

    try:
        with open(flow_path, "r") as f:
            flow = yaml.safe_load(f)
            if isinstance(flow, list):
                methods = []
                for i, item in enumerate(flow):
                    prefix = f"{i + 1}-"
                    m_name = item.get("method", "unknown")
                    methods.append((prefix, m_name))
                return methods
    except Exception as e:
        logging.error(f"Error parsing flow {flow_path}: {e}")
        return []
    return []


def process_instance(instance_dir: Path, methods_list: list[tuple[str, str]]):
    instance_id = instance_dir.name

    log_path = instance_dir / DEFAULT_CONTROLLER_LOG_NAME
    time_records = parse_controller_log(log_path)

    with open(instance_dir / "method_time_log.json", "w", encoding="utf-8") as f:
        json.dump(time_records, f, indent=2)

    method_end_times = {}
    for r in time_records:
        if "call_context" in r:
            # call_context is "1-set_random_seed", match by "1-"
            ctx = r["call_context"]
            for prefix, _ in methods_list:
                if ctx.startswith(prefix):
                    method_end_times[prefix] = r["end_sec"]
                    break

    obj_log_path = (
        instance_dir / DEFAULT_RESULTS_DIR / OBJ_LOG_FN_FORMAT.format(instance_id)
    )
    if not obj_log_path.exists():
        obj_log_path = instance_dir / OBJ_LOG_FN_FORMAT.format(instance_id)

    obj_content = parse_obj_log(obj_log_path)
    obj_data = obj_content.get("data", {})
    obj_notes = obj_content.get("notes", {})

    # Trim method end_sec / obj_value to the configured timelimit so the
    # per-scenario summary CSV (and downstream method-mean charts) reflect the
    # deadline-truncated view, even when the solver wall-clock overran. See
    # docs/TODO.md "Hard cutoff" for the underlying overrun issue.
    timelimit_sec = _read_instance_timelimit(instance_dir, instance_id)

    csv_rows = []
    current_obj_value = None

    for i, (method_prefix, method_name) in enumerate(methods_list):
        end_sec = method_end_times.get(method_prefix)
        in_notes = any(note.startswith(method_prefix) for note in obj_notes.values())

        if in_notes:
            obj_val = get_obj_value_for_method(
                method_prefix, method_name, obj_data, obj_notes, current_obj_value
            )
            final_end_sec = end_sec
            final_obj_val = obj_val
        else:
            successor_in_notes = False
            for j in range(i + 1, len(methods_list)):
                succ_prefix, _ = methods_list[j]
                if any(note.startswith(succ_prefix) for note in obj_notes.values()):
                    successor_in_notes = True
                    break

            if not successor_in_notes:
                final_end_sec = end_sec  # Keep end_sec if it was recorded
                final_obj_val = None
            else:
                final_end_sec = end_sec
                final_obj_val = current_obj_value

        if (
            timelimit_sec is not None
            and final_end_sec is not None
            and float(final_end_sec) > timelimit_sec
        ):
            # Method overran the budget: cap its endpoint at timelimit, and
            # use the obj value recorded at or before that moment.
            obj_at_limit = _last_obj_value_at_or_before(obj_data, timelimit_sec)
            if obj_at_limit is None and not _is_missing_obj_value(current_obj_value):
                obj_at_limit = current_obj_value
            final_end_sec = timelimit_sec
            if obj_at_limit is not None:
                final_obj_val = obj_at_limit

        if not _is_missing_obj_value(final_obj_val):
            current_obj_value = final_obj_val

        csv_rows.append(
            {
                "method_name": method_name,
                "method_end_sec": final_end_sec,
                "objective_value": final_obj_val,
            }
        )

    df = pd.DataFrame(csv_rows)
    df.to_csv(instance_dir / "method_end_time_and_obj_value.csv", index=False)
    return df


def process_scenario(scenario_dir: Path):
    if not scenario_dir.exists():
        logging.warning(f"Scenario directory {scenario_dir} does not exist.")
        return

    logging.info(f"Processing logs for scenario: {scenario_dir.name}")

    methods_list = get_methods_from_flow(scenario_dir)
    if not methods_list:
        logging.warning("No methods found in flow. Skipping summary generation.")
        return

    summary_rows = []

    instance_dirs = [
        p for p in scenario_dir.iterdir() if p.is_dir() and p.name.isdigit()
    ]
    instance_dirs.sort(key=lambda p: int(p.name))

    for instance_dir in instance_dirs:
        try:
            df = process_instance(instance_dir, methods_list)

            row = {"instance_id": int(instance_dir.name)}
            for _, r in df.iterrows():
                m_name = r["method_name"]
                row[f"{m_name}_end_sec"] = r["method_end_sec"]
                row[f"{m_name}_obj_value"] = r["objective_value"]

            summary_rows.append(row)
        except Exception as e:
            logging.error(f"Failed to process {instance_dir.name}: {e}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        cols = ["instance_id"]
        for _, m_name in methods_list:
            cols.append(f"{m_name}_end_sec")
            cols.append(f"{m_name}_obj_value")

        existing_cols = [c for c in cols if c in summary_df.columns]
        summary_df = summary_df.reindex(columns=existing_cols)
        out_path = scenario_dir / "summary_method_end_time_and_obj_value.csv"
        summary_df.to_csv(out_path, index=False)
        logging.info(f"Summary saved to: {out_path}")


def main():
    cwd = Path.cwd()
    if (cwd / "subroutine_flow.yaml").exists():
        process_scenario(cwd)
    else:
        print("Run this script from a scenario output directory, or import it.")


if __name__ == "__main__":
    main()
