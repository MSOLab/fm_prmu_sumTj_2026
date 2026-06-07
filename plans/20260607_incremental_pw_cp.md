# Plan: `incremental_pw_cp` 메서드 추가

> 이 문서는 Sonnet subagent가 단독으로 구현할 수 있도록 정리된 구현 명세다.
> 코드 작성 전 아래 "참조 파일/라인"을 먼저 읽어 현재 시그니처를 확인할 것.

## 목표

`flowshop_tardiness/controller/fm_sumtj_cp_lns.py`에 `incremental_pw_cp` 메서드를 추가한다.
batch size를 `start_batch_size`부터 `end_batch_size`까지 1씩 늘리면서 `pw_cp`를 순차 실행하고,
각 step마다 선택적으로 `improve_by_insertion(subseq_size=1, max_passes=1)`을 실행한다.

- ramp-up 동안에는 improvement 여부를 판단하지 않는다 — 별도 stopping criterion 없이 `end_batch_size`까지 순차 진행한다.
- `repeat_at_end_batch_size_while_improving`가 True면, `end_batch_size` 도달 후 그 step을
  incumbent objective가 strict 개선되는 동안 추가로 반복하는 polish 단계를 수행한다
  (`repeat_while_improvement`와 동일 로직).
- 전역 `is_stopping_condition()`은 ramp-up·polish 양쪽에서 종료 조건으로 사용한다.

## 시그니처

```python
def incremental_pw_cp(
    self,
    start_batch_size: int,
    end_batch_size: int,
    max_time_per_add: float | None = None,
    solver_thread_cnt: int | None = None,
    improvement_by_insertion_after_every_pw_cp: bool = True,
    repeat_at_end_batch_size_while_improving: bool = True,
) -> None:
```

## 입력 검증 (메서드 진입 직후)

- `start_batch_size < 1` 이면 `ValueError` 발생
  - 메시지 예: `f"start_batch_size must be >= 1, got {start_batch_size}"`
- `end_batch_size < start_batch_size` 이면 `ValueError` 발생
  - 메시지 예: `f"end_batch_size ({end_batch_size}) must be >= start_batch_size ({start_batch_size})"`

## 구현 방식 (중요: 직접 호출 금지)

`pw_cp` / `improve_by_insertion`을 **직접 Python 메서드로 호출하면 안 된다.**
이 코드베이스에서 서브루틴은 routix 디스패처(`_run_flow` → `_call_method`)를 통해
method context stack에 push 되어야 하며, 그래야 `pw_cp`가 출력하는
`_obj_log.yaml` 경로가 호출마다 distinct하게 생성된다
(`get_file_path_for_subroutine`가 context stack의 call_count를 사용함).
직접 호출하면 매 반복이 동일 경로를 덮어써서 마지막 반복 로그만 남는다.

따라서 기존 `repeat` / `repeat_while_improvement` 패턴과 동일하게,
`temporarily_extended_context` + `_run_flow`로 디스패치한다.

### 동작 (구현 코드 골격)

step 빌드/실행을 지역 함수로 추출해 ramp-up 루프와 polish 단계가 공유한다(DRY).

```python
subroutine_name = "incr_pw_cp"  # context stack에 push될 이름 (repeat의 "reps"와 동일한 역할)

def build_steps(batch_size: int) -> list[dict]:
    steps: list[dict] = [
        {
            "method": "pw_cp",
            "params": {
                "added_batch_size": batch_size,
                "profile_fixed_cnt": 0,
                "step_size_on_improve": batch_size,
                "step_size_on_no_improve": 1,
                "max_time_per_add": max_time_per_add,
                "solver_thread_cnt": solver_thread_cnt,
            },
        },
    ]
    if improvement_by_insertion_after_every_pw_cp:
        steps.append(
            {
                "method": "improve_by_insertion",
                "params": {"subseq_size": 1, "max_passes": 1},
            }
        )
    return steps

def run_step(batch_size: int) -> None:
    with self.temporarily_extended_context(subroutine_name):
        self._run_flow(DynamicDataObject.from_obj(build_steps(batch_size)))

# --- ramp-up ---
for batch_size in range(start_batch_size, end_batch_size + 1):
    if self.is_stopping_condition():
        logging.info(
            f"[IncrementalPwCp] Stopping condition met at batch_size={batch_size}."
        )
        break
    logging.info(f"[IncrementalPwCp] batch_size={batch_size}")
    run_step(batch_size)

if not repeat_at_end_batch_size_while_improving:
    return

# --- polish: end_batch_size step을 strict 개선되는 동안 반복 ---
# (repeat_while_improvement와 동일하게 obj 전후 비교 + float_a_stl_b 사용)
incumbent_sol = self.solution_manager.get_incumbent()
obj_before = math.inf if incumbent_sol is None else self.get_obj_value(incumbent_sol)
while True:
    if self.is_stopping_condition():
        logging.info("[IncrementalPwCp] Stopping condition met during end-batch repeat.")
        break
    logging.info(f"[IncrementalPwCp] Repeating end_batch_size={end_batch_size} while improving.")
    run_step(end_batch_size)

    incumbent_sol = self.solution_manager.get_incumbent()
    obj_after = math.inf if incumbent_sol is None else self.get_obj_value(incumbent_sol)
    if float_a_stl_b(obj_after, obj_before):
        obj_before = obj_after  # 개선됨 → 계속
    else:
        break  # 개선 없음 → 종료
```

### 디스패치 동작 참고 (subagent가 알아야 할 사실)

- `_run_flow`는 Sequence를 받으면 각 step을 순회하며 실행한다.
- `_run_flow`는 **각 step 실행 전에 `is_stopping_condition()`을 내부적으로 검사**한다
  (`routix/subroutine_controller.py:172`). 따라서 step 사이에 수동 stopping 체크를 넣을 필요가 없다.
  for 루프 상단의 `is_stopping_condition()` 체크는 루프 자체를 조기 종료시키기 위한 것이다.
- step dict 포맷: `{"method": <name>, "params": {<kwargs>}}` (routix `SubroutineFlowKeys.parse_step`).
- `DynamicDataObject.from_obj(list[dict])`는 dict 리스트를 step 시퀀스로 변환한다.

## docstring (Google 스타일, 파일 내 다른 메서드와 동일 포맷으로 추가)

다음 내용을 포함할 것:

- 메서드 요약: incumbent 시퀀스에 대해 batch size를 늘려가며 `pw_cp`를 순차 실행하고
  옵션에 따라 step마다 insertion 개선을 수행한다는 설명.
- Args:
  - `start_batch_size (int)`: 시작 batch size (>= 1).
  - `end_batch_size (int)`: 종료 batch size (포함, >= start_batch_size).
  - `max_time_per_add (float | None)`: 각 `pw_cp` CP solve 1회당 시간 제한. `pw_cp`로 그대로 전달.
  - `solver_thread_cnt (int | None)`: CP 솔버 스레드 수. `pw_cp`로 그대로 전달.
  - `improvement_by_insertion_after_every_pw_cp (bool)`: 각 `pw_cp` 후
    `improve_by_insertion(subseq_size=1, max_passes=1)` 실행 여부. 기본 True.
  - `repeat_at_end_batch_size_while_improving (bool)`: True면 ramp-up 후
    `end_batch_size` step을 incumbent objective가 strict 개선되는 동안 반복. 기본 True.
- Raises:
  - `ValueError`: 입력 검증 실패 시.

## 변경 파일 / 위치

- `flowshop_tardiness/controller/fm_sumtj_cp_lns.py`
  - `repeat_while_improvement` 메서드(`:1568`) 아래에 `incremental_pw_cp` 추가.

## import 확인

- `DynamicDataObject` 가 해당 파일에 이미 import 되어 있는지 확인하고, 없으면 import 추가.
  (`repeat_while_improvement`가 이미 `DynamicDataObject`를 사용하므로 import 되어 있을 가능성 높음 — 확인만.)
- polish 단계에 쓰는 `math`(`math.inf`)와 `float_a_stl_b`도 확인.
  (`repeat_while_improvement`가 이미 둘 다 사용 — 추가 불필요.)
- `logging`은 이미 사용 중.

## YAML 사용 예시

```yaml
- method: incremental_pw_cp
  params:
    start_batch_size: 3
    end_batch_size: 10
    max_time_per_add: 10
    solver_thread_cnt: 1
    improvement_by_insertion_after_every_pw_cp: true
    repeat_at_end_batch_size_while_improving: true
```

## 참조 파일/라인 (구현 전 확인)

- `pw_cp` 메서드: `fm_sumtj_cp_lns.py:1359` (인자: added_batch_size, profile_fixed_cnt,
  step_size_on_improve, step_size_on_no_improve, max_time_per_add, solver_thread_cnt, ...)
- `improve_by_insertion` 메서드: `fm_sumtj_cp_lns.py:1081` (인자: subseq_size, max_passes, ...)
- `repeat_while_improvement` 패턴: `fm_sumtj_cp_lns.py:1568` (구조/네이밍 참고)
- `repeat` 패턴: `routix/subroutine_controller.py:281` (temporarily_extended_context + _run_flow 사용 예)
- `temporarily_extended_context`: `routix/subroutine_controller.py:274`
- `_run_flow`: `routix/subroutine_controller.py:152`
- `is_stopping_condition`: `controller_core.py:161` (상속)

## 검증

- 구현 후 `uv run python -c "import flowshop_tardiness.controller.fm_sumtj_cp_lns"` 로 import 에러 없는지 확인.
- 메서드 시그니처와 입력 검증, docstring이 위 명세와 일치하는지 확인.
