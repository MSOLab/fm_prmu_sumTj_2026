import tempfile
from pathlib import Path

from routix.io import dump_yaml, load_yaml


def test_dump_yaml_with_tuple_keys_roundtrip():
    data = {
        "start_time_map": {("job1", "stage1"): 0, ("job2", "stage1"): 5},
        "end_time_map": {("job1", "stage1"): 10, ("job2", "stage1"): 15},
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.yaml"
        dump_yaml(data, path)
        loaded = load_yaml(path)
        assert loaded["start_time_map"] == data["start_time_map"]
        assert loaded["end_time_map"] == data["end_time_map"]


def test_dump_yaml_without_tuple_keys():
    data = {"name": "test", "value": 42, "tags": ["a", "b"]}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.yaml"
        dump_yaml(data, path)
        loaded = load_yaml(path)
        assert loaded == data


def test_load_yaml_returns_tuple_keys():
    yaml_content = """
start_time_map:
  [job1, stage1]: 0
  [job2, stage1]: 5
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.yaml"
        path.write_text(yaml_content)
        loaded = load_yaml(path)
        assert loaded["start_time_map"] == {("job1", "stage1"): 0, ("job2", "stage1"): 5}
