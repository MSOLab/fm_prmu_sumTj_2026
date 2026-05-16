from pathlib import Path

from routix.io import load_yaml

START_TIME_MAP_KEY = "start_time_map"
END_TIME_MAP_KEY = "end_time_map"

# Shared constants used across scripts and dashboard modules
OBJ_LOG_FN_FORMAT = "{}_obj_log.yaml"
RESULT_DIR_NAME = "results"


def get_start_time_dict(sol_path: Path, encoding: str = "utf-8") -> dict:
    solution_dict = load_yaml(sol_path, encoding=encoding)
    if START_TIME_MAP_KEY in solution_dict:
        return solution_dict[START_TIME_MAP_KEY]
    raise ValueError(
        f"{START_TIME_MAP_KEY} not found in solution file: {sol_path}"
    )


def get_end_time_dict(sol_path: Path, encoding: str = "utf-8") -> dict:
    solution_dict = load_yaml(sol_path, encoding=encoding)
    if END_TIME_MAP_KEY in solution_dict:
        return solution_dict[END_TIME_MAP_KEY]
    raise ValueError(
        f"{END_TIME_MAP_KEY} not found in solution file: {sol_path}"
    )
