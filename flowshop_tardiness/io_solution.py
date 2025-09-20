from pathlib import Path

import yaml
from routix.io import pyyaml_key_to_tuple


def get_start_time_dict(sol_path: Path, encoding: str = "utf-8") -> dict:
    data = None
    with open(sol_path, "r", encoding=encoding) as f:
        solution_dict = yaml.load(f, Loader=yaml.UnsafeLoader)
        if "start_time_map" in solution_dict:
            data = solution_dict["start_time_map"]
        else:
            raise ValueError(f"'start_time_map' found in solution file: {sol_path}")
    return pyyaml_key_to_tuple(data)


def get_end_time_dict(sol_path: Path, encoding: str = "utf-8") -> dict:
    data = None
    with open(sol_path, "r", encoding=encoding) as f:
        solution_dict = yaml.load(f, Loader=yaml.UnsafeLoader)
        if "end_time_map" in solution_dict:
            data = solution_dict["end_time_map"]
        else:
            raise ValueError(f"'end_time_map' found in solution file: {sol_path}")
    return pyyaml_key_to_tuple(data)
