from pathlib import Path

import yaml
from routix.io import pyyaml_key_to_tuple

START_TIME_MAP_KEY = "start_time_map"
END_TIME_MAP_KEY = "end_time_map"

def get_start_time_dict(sol_path: Path, encoding: str = "utf-8") -> dict:
    data = None
    with open(sol_path, "r", encoding=encoding) as f:
        solution_dict = yaml.load(f, Loader=yaml.UnsafeLoader)
        if START_TIME_MAP_KEY in solution_dict:
            data = solution_dict[START_TIME_MAP_KEY]
        else:
            raise ValueError(f"{START_TIME_MAP_KEY} found in solution file: {sol_path}")
    return pyyaml_key_to_tuple(data)


def get_end_time_dict(sol_path: Path, encoding: str = "utf-8") -> dict:
    data = None
    with open(sol_path, "r", encoding=encoding) as f:
        solution_dict = yaml.load(f, Loader=yaml.UnsafeLoader)
        if END_TIME_MAP_KEY in solution_dict:
            data = solution_dict[END_TIME_MAP_KEY]
        else:
            raise ValueError(f"{END_TIME_MAP_KEY} found in solution file: {sol_path}")
    return pyyaml_key_to_tuple(data)
