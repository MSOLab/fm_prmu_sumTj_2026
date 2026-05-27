# MCF-LB 호출과 init schedule 등록 분리 + 5개 ablation config 재구성

## Context

`configs_cp_lns/` 의 ablation config 들을 논문 `Juntaek-PhD-Thesis/contents/fc_prmu_sumTj.tex` §4.1.3 표와 대조한 결과 세 가지 불일치 확인:

1. **thread 수 불일치** — C4/C5의 `solve_base_cp_model.solver_thread_cnt` 와 `pw_cp.solver_thread_cnt` 가 1. 본문 `:617-618` 은 "eight threads" 라고 일관 서술. 다른 SOTA 와 동등 조건을 위해 모두 8로 통일.

2. **MCF-LB 호출이 init schedule 생성을 강제** — `compute_preemptive_last_stage_lb` (`flowshop_tardiness/controller/fm_sumtj_cp_lns.py:1222`) 는 `init_by_neh_ms` 로 dispatched 후보냐 NEH-MS 후보냐만 가르고 LBE/LBL/LBA best 를 무조건 `solution_manager.register(...)`. "MCF-LB 를 numerical lower bound 로만 쓰고 initialization 에는 손대지 않는" 경로 부재.

3. **`initialize_by_edd` 가 redundant** — NEH-MS 는 정의상 EDD seed 사용 (NEHedd). 별도 호출 불필요.

세 가지를 한 번에 정리하고 5개 ablation config 를 본문 표와 정확히 대응되게 재정의.

---

## 1. 코드 변경: `compute_preemptive_last_stage_lb` 시그니처

**파일:** `flowshop_tardiness/controller/fm_sumtj_cp_lns.py:1222`

### 1.1 시그니처: `init_by_neh_ms: bool` → `init_method: Literal[...]`

```python
from typing import Literal

InitMethod = Literal["dispatch", "neh-ms", "lb_only"]

def compute_preemptive_last_stage_lb(
    self,
    init_method: InitMethod = "dispatch",
    error_if_infeasible: bool = False,
    draw_gantt: bool = False,
) -> None:
```

세 값 의미:

| 값 | 동작 | 대응되는 기존 동작 |
|---|---|---|
| `"dispatch"` | LBE/LBL/LBA 시퀀스로 dispatched schedule 3개 생성 → best 등록 | 기존 `init_by_neh_ms=False` (default) |
| `"neh-ms"` | LBE/LBL/LBA 시퀀스 각각에 NEH-MS 적용 → best 등록 | 기존 `init_by_neh_ms=True` |
| `"lb_only"` | LB 값만 obj_bound 로 기록, schedule 생성/등록 X (신규) | 없음 |

default `"dispatch"` 이므로 `init_by_neh_ms` 를 명시하지 않던 기존 config (61개, 주로 `configs_600s/`) 동작 보존.

`Literal[...]` 은 runtime 검증되지 않음 (YAML dispatcher 가 `getattr(controller, name)(**kwargs)` 로 호출하므로). 정적 타입 체크 + 사람 가독성 목적.

### 1.2 함수 본문 변경

기존 `:1242-1264` 의 nested `_make_schedule_from` 과 label 분기:

```python
def _make_schedule_from(seq: list[str]) -> FlowshopSchedule:
    if init_by_neh_ms:
        schedule, _ = self._build_neh_schedule(...)
        return schedule
    return self.get_dispatched_schedule(seq)

# ...
label = "NEH-MS" if init_by_neh_ms else "Dispatched"
```

→ `init_method` 기반으로 교체:

```python
def _make_schedule_from(seq: list[str]) -> FlowshopSchedule:
    if init_method == "neh-ms":
        schedule, _ = self._build_neh_schedule(
            seq, tie_breaker="makespan",
            error_if_infeasible=error_if_infeasible,
        )
        return schedule
    return self.get_dispatched_schedule(seq)

# label
label = "NEH-MS" if init_method == "neh-ms" else "Dispatched"
```

### 1.3 신규 분기: `init_method == "lb_only"` early return

`:1240` (obj_bound logging 직후) 와 `:1242` (`_make_schedule_from` 정의) 사이에 삽입:

```python
if init_method == "lb_only":
    log_time = self.timer.elapsed_sec
    report = FsSubroutineReport(
        elapsed_time=sub_timer.elapsed_sec,
        obj_value=None,
        obj_bound=obj_bound,
        is_init=False,
    )
    sub_timer.reset()
    self.solution_manager.register(report, None)

    last_obj_bound = self.obj_store.get_last_obj_bound()
    best_obj_bound = (
        obj_bound
        if last_obj_bound is None or obj_bound > last_obj_bound
        else last_obj_bound
    )
    self.add_obj_bound_log(log_time, best_obj_bound, is_maximize=None)
    _last_timestamp_note = self._get_call_context_of_current_method()
    self.obj_store.add_last_timestamp_note(
        _last_timestamp_note,
        obj_value_is_valid=False,
        obj_bound_is_valid=True,
    )
    return
```

패턴 출처: `set_0_as_lb` (`flowshop_tardiness/controller/fm_sumtj_cp_lns.py:55-69`). 차이점:
- `obj_bound=0.0` 대신 계산된 `obj_bound` 사용
- `obj_value_is_valid=True` (`set_0_as_lb` 는 의도적으로 obj_value=None 인데도 True. 우리는 의미론적으로 `False` 가 정확) → `obj_value_is_valid=False, obj_bound_is_valid=True` 사용
- `is_init=False` (lb_only 는 initialization 아님)

---

## 2. 기존 YAML 마이그레이션

`init_by_neh_ms` 가 사라지므로 명시적으로 쓰던 파일 전부 새 키로 변경. 총 5개:

```
configs_cp_lns/20260514_ablation_c3.yaml   (line 4)
configs_cp_lns/20260514_ablation_c5.yaml   (line 4)
configs_cp_lns/20260514_lb_neh.yaml        (line 4)
configs_cp_lns/20260515_ablation_c3.yaml   (line 6)
configs_cp_lns/20260515_ablation_c5.yaml   (line 6)
```

모두 `init_by_neh_ms: true` → `init_method: neh-ms`.

`init_by_neh_ms: false` 사용 파일 없음 (별도 처리 불요).

명시하지 않던 61개 파일 (`configs_600s/` 등) 은 default `"dispatch"` 가 기존 동작 (`init_by_neh_ms=False`) 과 동일하므로 손대지 않음.

---

## 3. 신규 5개 ablation config 생성

논문 §4.1.3 표 대응:

| Config | NEH-MS | LB-init | SW-CP | II | Base-CP |
|---|---|---|---|---|---|
| C1 | ✓ | – | – | – | ✓ |
| C2 | ✓ | – | – | ✓ | ✓ |
| C3 | ✓ | ✓ | – | ✓ | ✓ |
| C4 (제안) | ✓ | – | ✓ | ✓ | ✓ |
| C5 | ✓ | ✓ | ✓ | ✓ | ✓ |

- "LB-init ✓" = `init_method: neh-ms`
- "LB-init –" = `init_method: lb_only` (LB 계산은 하되 candidate schedule 안 만듦)
- 공통 골격: `set_random_seed → initialize_by_nehms → compute_preemptive_last_stage_lb → set_cp_model_as_base_cp_model → [II / repeat_while_improvement] → solve_base_cp_model`
- LB 호출은 항상 NEH-MS **뒤**. C3/C5 에서 `init_method: neh-ms` 일 때 incumbent 의 job_list 가 NEH seed 로 쓰이므로 NEH-MS 가 먼저 실행되어 있어야 4-candidate semantic (`fc_prmu_sumTj.tex:391-395`) 성립.
- 모든 `solver_thread_cnt: 8` (기존 0515 C4/C5 의 1 에서 변경).

### `20260527_ablation_c1.yaml`

```yaml
- method: set_random_seed
  seed: 42
- method: initialize_by_nehms
- method: compute_preemptive_last_stage_lb
  init_method: lb_only
- method: set_cp_model_as_base_cp_model
- method: solve_base_cp_model
  computational_time: null
  solver_thread_cnt: 8
```

### `20260527_ablation_c2.yaml`

```yaml
- method: set_random_seed
  seed: 42
- method: initialize_by_nehms
- method: compute_preemptive_last_stage_lb
  init_method: lb_only
- method: improve_by_insertion
  subseq_size: 1
  first_improvement: false
- method: set_cp_model_as_base_cp_model
- method: solve_base_cp_model
  computational_time: null
  solver_thread_cnt: 8
```

### `20260527_ablation_c3.yaml`

```yaml
- method: set_random_seed
  seed: 42
- method: initialize_by_nehms
- method: compute_preemptive_last_stage_lb
  init_method: neh-ms
- method: improve_by_insertion
  subseq_size: 1
  first_improvement: false
- method: set_cp_model_as_base_cp_model
- method: solve_base_cp_model
  computational_time: null
  solver_thread_cnt: 8
```

### `20260527_ablation_c4.yaml` (제안 알고리즘)

```yaml
- method: set_random_seed
  seed: 42
- method: initialize_by_nehms
- method: compute_preemptive_last_stage_lb
  init_method: lb_only
- method: set_cp_model_as_base_cp_model
- method: repeat_while_improvement
  n_repeats: 99
  routine_data:
    - method: pw_cp
      added_batch_size: 7
      profile_fixed_cnt: 0
      step_size_on_improve: 7
      step_size_on_no_improve: 1
      max_time_per_add: 10
      solver_thread_cnt: 8
    - method: improve_by_insertion
      subseq_size: 1
      first_improvement: false
      max_passes: 1
- method: solve_base_cp_model
  computational_time: null
  solver_thread_cnt: 8
```

### `20260527_ablation_c5.yaml`

```yaml
- method: set_random_seed
  seed: 42
- method: initialize_by_nehms
- method: compute_preemptive_last_stage_lb
  init_method: neh-ms
- method: set_cp_model_as_base_cp_model
- method: repeat_while_improvement
  n_repeats: 99
  routine_data:
    - method: pw_cp
      added_batch_size: 7
      profile_fixed_cnt: 0
      step_size_on_improve: 7
      step_size_on_no_improve: 1
      max_time_per_add: 10
      solver_thread_cnt: 8
    - method: improve_by_insertion
      subseq_size: 1
      first_improvement: false
      max_passes: 1
- method: solve_base_cp_model
  computational_time: null
  solver_thread_cnt: 8
```

기존 0512/0514/0515 config 는 git history 보존 (삭제하지 말 것).

---

## 4. metadata 파일

`metadata_cp_lns_20260527.yaml` 신규 생성. `metadata_cp_lns_20260514.yaml` 과 동일 형식, 시간 제한 / 인스턴스 셋 / 시드 동일 유지, 새 5개 config 참조.

---

## 5. 검증

### 코드 smoke test (1 인스턴스)

`configs_cp_lns/20260527_ablation_c1.yaml` (또는 임의의 작은 단일 인스턴스) 로 실행:

- C1 (`init_method: lb_only`) 호출 후 `solution_manager.get_incumbent()` 가 NEH-MS 결과 그대로 (LB candidate 가 incumbent 안 바꿈)
- C1 의 `obj_store.get_last_obj_bound()` 가 LB 값으로 갱신됨
- C3/C5 (`init_method: neh-ms`) 로그에 "NEH-MS schedules' total tardiness:" 가 보이고 3개 후보 (start/end/avg) 가 NEH-MS 로 빌드됨

### Thread 확인

12 인스턴스 parallel smoke run 에서:
- CP-SAT 로그의 `Parameters: ... num_workers:8` 확인 (pw_cp, solve_base_cp_model 둘 다)
- 12 instance × 8 workers = 96 → CPU 사용률 ~100% (이전 12 × 1 = 12.5% 대비)

### Regression

기존 default 사용 config (`configs_600s/` 의 60+ 파일 중 임의 1개) 1 인스턴스 실행 → 0515 동일 config 결과와 비교 시 incumbent / obj_bound 동일.

---

## 6. 작업 순서

1. `flowshop_tardiness/controller/fm_sumtj_cp_lns.py:1222` 시그니처 + 본문 변경 (§1).
2. 5개 기존 yaml 마이그레이션 (§2).
3. 신규 5개 ablation yaml + metadata 생성 (§3, §4).
4. 검증 (§5) — 코드 smoke / thread 확인 / regression.
5. (별도 repo) `Juntaek-PhD-Thesis/contents/fc_prmu_sumTj.tex` §4.1.3 표 갱신: "EDD" 컬럼 삭제, "LB-Disp" → "LB-init", caption 에 "All configurations start with NEH-MS(EDD) construction." 추가.

---

## 7. 미해결

- `set_0_as_lb` (`:55-69`) 의 `obj_value_is_valid=True` 가 의도적인지 (obj_value=None 인데 True) — 본 plan 의 lb_only 분기는 `False` 가 정확하다고 판단해 그렇게 작성. 만약 일관성을 위해 `set_0_as_lb` 도 수정해야 한다면 별건.
