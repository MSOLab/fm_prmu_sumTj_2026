# routix/mbls/schore 최신 버전 대응 계획

## 문제

`uv run main.py` 실행 시 `ImportError: cannot import name 'object_to_yaml' from 'routix.io'` 발생.
`routix`, `mbls`, `schore` 세 패키지의 최신 PyPI 버전 (각각 0.0.17, 0.0.15, 0.0.7)이
기존 editable 설치와 API가 달라졌기 때문. `pyproject.toml`에서 `[tool.uv.sources]`로
editable 경로를 쓰지 않고 PyPI 버전을 직접 사용할 수 있도록 코드 수정이 필요.

## 원인 분석

### routix.io YAML 관련 함수 변경

| 삭제된 함수 | 대체 방법 |
|---|---|
| `object_to_yaml(data, path, encoding)` | `dump_yaml(data, path, encoding=encoding)` |
| `tuple_to_pyyaml_key(dict)` | 불필요. `dump_yaml`의 `PrettyKeyDumper`가 tuple key를 YAML sequence로 자동 변환 |
| `pyyaml_key_to_tuple(dict)` | 불필요. `load_yaml`의 `PrettyKeyLoader`가 YAML list → tuple key 자동 변환 |

### 기존 코드와의 호환성 설명

기존 코드는 아래 순서로 동작했음:

1. `tuple_to_pyyaml_key(dict_with_tuple_keys)` → tuple key를 list로 변환
2. `object_to_yaml(data, path)` → YAML 파일로 저장

이제는 1단계가 필요 없음. `dump_yaml()`이 내부에서 `PrettyKeyDumper`를 사용해
tuple key를 YAML flow sequence로 직접 표현함.

읽을 때도 `load_yaml()`이 `PrettyKeyLoader`를 사용해 YAML sequence(list)를 tuple key로
자동 변환하므로 `pyyaml_key_to_tuple()`이 불필요함.

### 변경 없는 API (영향 없음)

- `mbls.cpsat.CustomCpModel`, `CpsatSolverReport`, `CpsatStatus`, `ObjValueBoundStore`
- `mbls.cpsat.callbacks.ValueBoundPair`
- `schore.parameters_examples.shop.flow.FlowshopDuedateParameters`
- `schore.schedule_examples.shop.flow.FlowshopSchedule`, `FlowshopOperation`
- `routix.DynamicDataObject`, `ElapsedTimer`, `StoppingCriteria`, `SubroutineFlowValidator`
- `routix.util.comparison.float_equals`, `float_a_leq_b`, `float_a_stl_b`
- `routix.report.SubroutineReport`, `SubroutineReportStatistics`
- `routix.solution_manager.SolutionManager`
- `routix.runner.SingleInstanceRunner`, `MultiScenarioRunner`, `MultiInstanceConcurrentRunner`
- `routix.type_defs.RunMode`

## 수정 파일 목록

### 그룹 A: `object_to_yaml` → `dump_yaml` (7개 파일)

import 문과 호출부만 단순 치환.

| # | 파일 | 수정 내용 |
|---|---|---|
| 1 | `main.py` | import 변경, 호출 2건 |
| 2 | `tbb_2018_mhx1_main.py` | import 변경, 호출 2건 |
| 3 | `ga_ctrlr_main.py` | import 변경, 호출 2건 |

### 그룹 B: `object_to_yaml` + `tuple_to_pyyaml_key` 제거 (5개 파일)

`tuple_to_pyyaml_key()`로 dict key를 변환하던 코드를 제거하고,
tuple key를 가진 dict를 `dump_yaml()`에 바로 전달.

| # | 파일 | 수정 내용 |
|---|---|---|
| 4 | `flowshop_tardiness/controller/controller_core.py` | import, export_solution_to_yaml |
| 5 | `flowshop_tardiness/controller/base_flowshop_controller.py` | import, export_solution_to_yaml |
| 6 | `fs_single_instance_runner.py` | import, export_solution_to_yaml |
| 7 | `tbb_2018_mhx1_single_instance.py` | import, export_solution_to_yaml |
| 8 | `ga_ctrlr_single_instance.py` | import, export_solution_to_yaml |

### 그룹 C: `pyyaml_key_to_tuple` → `load_yaml` 대체 (1개 파일)

`yaml.load(f, Loader=yaml.UnsafeLoader)` + `pyyaml_key_to_tuple(data)`를
`load_yaml(path)`로 대체 (`PrettyKeyLoader`가 list→tuple 변환을 자동 처리).

| # | 파일 | 수정 내용 |
|---|---|---|
| 9 | `flowshop_tardiness/io_solution.py` | import, get_start_time_dict, get_end_time_dict |

### 그룹 D: `pyproject.toml` (1개 파일)

`[tool.uv.sources]` 블록 제거.

| # | 파일 | 수정 내용 |
|---|---|---|
| 10 | `pyproject.toml` | editable 경로 참조 삭제 |

## 구현 절차

1. `pyproject.toml` — `[tool.uv.sources]` 블록 제거
2. 그룹 A 파일들 — import + 호출부 수정
3. 그룹 B 파일들 — import + tuple key 처리 코드 제거
4. 그룹 C 파일 — `io_solution.py` 전면 수정
5. `uv sync` 실행 (PyPI 패키지 설치)
6. `uv run main.py` 실행 테스트
7. `uv run pytest` 실행 테스트
