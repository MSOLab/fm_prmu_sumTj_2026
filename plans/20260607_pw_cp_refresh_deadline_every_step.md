# 구현 계획: PW-CP `refresh_deadline_every_step` 옵션 추가

> 모태: `20260607_pw_cp_increasing_step_investigation.md` §6.
> 차이점: §6은 LCT 출처를 "run() 1회 `given_sol`" → "매 iter right-justify"로 **무조건 교체**하려 했다.
> 본 계획은 대신 **`refresh_deadline_every_step: bool = False` optional keyword argument**를 추가해
> - `False`(기본): **현재 동작 그대로 유지**(고정 `given_sol`에서 LCT 산출, sweep 단위 단조성만 보장).
> - `True`: 매 iteration 직전 상태를 right-justify하여 LCT 재계산(iteration 단위 단조성 보장).
> 즉 §6의 (B)를 **opt-in flag**로 제공하고 (A)/기존 실험 재현성을 깨지 않는다.

---

## 0. 작업 분배 (subagent 배정)

각 **파일 1개 = sonnet subagent 1개**. 두 작업 패키지(WP)는 서로 독립적인 텍스트 편집이므로
**병렬 실행 가능**(런타임 통합은 둘 다 끝나야 동작). 각 WP는 자기완결적으로 작성됨 —
subagent에게 해당 WP 섹션 전체를 그대로 전달하면 된다.

| WP | 파일 | 담당 subagent | 핵심 작업 |
| --- | --- | --- | --- |
| **WP-1** | `flowshop_tardiness/controller/pw_cp.py` | sonnet | 파라미터 추가 + LCT 계산 분기(실제 로직) |
| **WP-2** | `flowshop_tardiness/controller/fm_sumtj_cp_lns.py` | sonnet | 파라미터 추가 + 하위 호출로 thread-through |

> 공통 규약(두 WP 모두): 새 파라미터는 **모두 keyword-only 위치(기존 마지막 인자 뒤)**, 기본값 `False`.
> 기본값 경로는 **수정 전과 동작이 완전히 동일**해야 한다(회귀 금지). Python 실행은 항상 `uv run python`.

---

## 1. 배경 (두 WP 공통 컨텍스트)

- LCT(per-stage latest completion 상한)는 right-justified 참조 스케줄 \(S^R\)에서 온다.
- 현재 `S^R`은 `run()` 진입 시 **입력 incumbent 시퀀스로 단 한 번** 계산되어 sweep 내내 고정.
- 결과: pw_cp는 **시작 incumbent 대비 비증가**(sweep 단위)만 보장하고, **내부 iteration 간에는
  단조롭지 않다**. per-step `*-pw_cp_obj_log.yaml`에서 objective 증가가 관찰됨.
- 근본 원인: 고정 `S^R`이 이미 개선된 중간 상태 대비 **slack**을 가지며, CP phase-2(∑ stage makespan
  최소화)가 그 slack을 마지막 stage makespan↑에 써서 greedy tail의 release를 밀어 tail tardiness↑.
- 검증(`check_swcp_rj_refresh_monotonic.py`): 매 step right-justify refresh 시 step 간 증가 0/8000
  (현재 14/8000), 최종 해 품질은 사실상 동등(평균 ≈0).

도입할 인터페이스(세 군데 동일 의미, 모두 기본값 `False`):

```python
refresh_deadline_every_step: bool = False
```

- `False`: LCT를 `run()`에서 1회 만든 고정 `given_sol`(=\(S^R\))에서 읽는다 (= 현행).
- `True`: 매 iteration에서 "직전 상태(committed CP순서 + remaining incumbent순서)"를
  right-justify한 스케줄에서 LCT를 읽는다.

---

## WP-1 — `flowshop_tardiness/controller/pw_cp.py`

> 이 파일이 **실제 동작 변경**을 담는다(LCT 계산 분기). subagent는 아래 4개 편집을 순서대로 적용.
> `given_sol`은 `False` 경로에서 **계속 필요**하므로 제거하지 말 것.

### 편집 1-1) `PwCpConstructor.run()` 시그니처 — `pw_cp.py:211-222`

마지막 인자 `draw_gantt: bool = False,` **다음 줄**에 추가:

```python
    refresh_deadline_every_step: bool = False,
```

docstring `Args:` 끝에 한 항목 추가:

```
refresh_deadline_every_step (bool, optional): When False (default), the per-stage LCT
    upper bound is derived once from the initial right-justified reference schedule S^R
    (current behavior; guarantees only sweep-level non-increase). When True, S^R/LCT is
    recomputed every iteration from the current state (committed CP order + remaining
    incumbent order), right-justified, which yields per-iteration monotonicity. Defaults to False.
```

### 편집 1-2) `run()` → `_run_loop()` 호출 — `pw_cp.py:322-329`

`draw_gantt=draw_gantt,` 다음에 인자 추가:

```python
            return self._run_loop(
                given_sol,
                solver_thread_cnt=solver_thread_cnt,
                max_time_per_add=max_time_per_add,
                error_if_infeasible=error_if_infeasible,
                draw_gantt=draw_gantt,
                refresh_deadline_every_step=refresh_deadline_every_step,
            )
```

> `run()`의 `given_sol` 생성/push_back(`:281-293`)과 draw_gantt 덤프는 **그대로 둔다**(False 경로용).

### 편집 1-3) `_run_loop()` 시그니처 — `pw_cp.py:620-627`

마지막 인자 `draw_gantt: bool = False,` 다음 줄에 추가:

```python
        refresh_deadline_every_step: bool = False,
```

docstring에도 동일 취지 한 줄 추가.

### 편집 1-4) LCT 계산 분기 — `pw_cp.py:707-712` (현재 코드)

```python
# 현재
if not st.last_job_is_included:
    stage_2_lct_map = given_sol.get_stage_2_start_time_map(st.not_added_first_job)
else:
    stage_2_lct_map = {}
```

→ 아래로 교체:

```python
if not st.last_job_is_included:
    if refresh_deadline_every_step:
        # 직전 상태(committed CP순서 + remaining incumbent순서)를 right-justify하여
        # 현재 tardiness를 보존하는 기준에서 LCT를 산출 -> iteration 단위 단조성.
        rj_ref = self._make_all_dispatched(st.time_fixed_sol, st.time_fixed_pool)
        rj_ref.push_back_tail_jobs_keep_tardiness(self.job_cnt)
        stage_2_lct_map = rj_ref.get_stage_2_start_time_map(st.not_added_first_job)
    else:
        # 고정 S^R(run() 1회 계산)에서 LCT -> sweep 단위 보장(현행).
        stage_2_lct_map = given_sol.get_stage_2_start_time_map(st.not_added_first_job)
else:
    stage_2_lct_map = {}
```

**근거(subagent 참고):**
- `_make_all_dispatched(st.time_fixed_sol, st.time_fixed_pool)`(`pw_cp.py:333`)는 이 시점에 commit 전이라
  `committed(CP순서) + remaining(self.job_sequence 순서)` 전체 스케줄을 반환한다(= "직전 schedule").
  검증 스크립트 `check_swcp_rj_refresh_monotonic.py`에서 쓴 로직과 동일.
- `push_back_tail_jobs_keep_tardiness`(`flowshop_tardiness/fm_prmu.py:164`)는 tardiness를 보존하며 우측정렬.
- `given_sol`은 `False` 분기에서만 참조되므로 시그니처에 그대로 유지.

### WP-1 검증 (subagent가 직접 수행)

```bash
uv run python -m py_compile flowshop_tardiness/controller/pw_cp.py
```

- 통과해야 함. 동작 검증(brute-force/416)은 WP-2 완료 후 §통합 검증에서 수행.

---

## WP-2 — `flowshop_tardiness/controller/fm_sumtj_cp_lns.py`

> 이 파일은 **파라미터를 하위로 전달(thread-through)만** 한다. 동작 로직 없음.
> WP-1이 `run()`에 `refresh_deadline_every_step`를 받도록 만든다는 전제 위에서 전달 코드를 작성한다.
> subagent는 아래 4개 편집을 적용.

### 편집 2-1) `pw_cp()` 시그니처 — `fm_sumtj_cp_lns.py:1359-1369`

마지막 인자 `draw_gantt: bool = False,` 다음 줄에 추가:

```python
        refresh_deadline_every_step: bool = False,
```

docstring `Args:`에 한 줄 추가(WP-1의 문구와 동일 취지: False=현행 sweep 단위, True=iteration 단위 단조).

### 편집 2-2) `pw_cp()` → `constructor.run(...)` 전달 — `fm_sumtj_cp_lns.py:1415-1425`

`draw_gantt=draw_gantt,` 다음에 추가:

```python
            refresh_deadline_every_step=refresh_deadline_every_step,
```

### 편집 2-3) `incremental_pw_cp()` 시그니처 — `fm_sumtj_cp_lns.py:1632-1640`

마지막 인자 `repeat_at_end_batch_size_while_improving: bool = True,` 다음 줄에 추가:

```python
        refresh_deadline_every_step: bool = False,
```

docstring `Args:`에 한 줄 추가:

```
refresh_deadline_every_step (bool): Passed through to each ``pw_cp`` step. When True,
    pw_cp recomputes the per-stage LCT bound every iteration for per-iteration
    monotonicity. Defaults to False (current behavior).
```

### 편집 2-4) `incremental_pw_cp()` → `build_steps`의 pw_cp params 전달 — `fm_sumtj_cp_lns.py:1688-1701`

`build_steps` 내부 `"method": "pw_cp"` step의 `"params"` dict에 키 1개 추가:

```python
            {
                "method": "pw_cp",
                "params": {
                    "added_batch_size": batch_size,
                    "profile_fixed_cnt": 0,
                    "step_size_on_improve": batch_size,
                    "step_size_on_no_improve": 1,
                    "max_time_per_add": max_time_per_add,
                    "solver_thread_cnt": solver_thread_cnt,
                    "refresh_deadline_every_step": refresh_deadline_every_step,
                },
            },
```

> `build_steps`는 `incremental_pw_cp`의 클로저이므로 인자 `refresh_deadline_every_step`를
> 캡처해서 그대로 쓸 수 있다(별도 인자 추가 불필요).

### WP-2 검증 (subagent가 직접 수행)

```bash
uv run python -m py_compile flowshop_tardiness/controller/fm_sumtj_cp_lns.py
```

---

## 통합 검증 (WP-1·WP-2 둘 다 완료 후, 메인 세션에서)

1. 컴파일 일괄:
   ```bash
   uv run python -m py_compile flowshop_tardiness/controller/pw_cp.py flowshop_tardiness/controller/fm_sumtj_cp_lns.py
   ```
2. brute-force 단조성:
   ```bash
   uv run python -m checks.check_swcp_rj_refresh_monotonic
   ```
   → refresh 경로 0 증가 확인.
3. **기본값 회귀(중요)**: flag 미지정 호출이 수정 전과 동일 결과인지.
   - 인스턴스 416을 `refresh_deadline_every_step` 미지정으로 재실행 → 기존 결과와 일치 확인.
4. opt-in 효과: 416을 `refresh_deadline_every_step=True`로 재실행 →
   `subroutine_controller.log`에서 `FULL OBJECTIVE INCREASED` 소멸 + per-step obj_log 단조 감소 확인,
   최종 obj가 `False` 대비 악화되지 않는지 비교.

진단 로깅(investigation §4) 정합성:
- `window_violations`는 새 LCT 기준(`True`)에서도 0이어야 함.
- `FULL OBJECTIVE INCREASED` WARNING은 `True`에서 **발화하지 않아야 정상**(검증 1차 신호).
  `False`에서는 기존대로 가끔 발화 가능(설계상 허용).

---

## 회귀 / 주의

- **기본값 `False`이므로 기존 실험/thesis 수치는 변하지 않는다**(§6 "무조건 교체"와의 핵심 차이).
- `True` 사용 시에만 pw_cp committed sequence가 달라져 수치 변동 → 해당 실험만 재실행.
- 비용(`True`): 매 iter `_make_all_dispatched`(O(n·c)) + `push_back`(O(n·c)) 추가.
  n=350,c=50에서 CP solve(수 초) 대비 무시 가능. (선택 최적화: 직전 iter 로깅용 full_sol 캐시 재사용 →
  KISS상 우선 매 iter 재계산으로 구현 권장.)
- `profile_fixed_cnt>0` 경로도 동일하게 `not_added_first_job` 기준을 쓰므로 구조 변화 없음(현 설정은 0).

---

## 변경 파일 / 라인 요약

| WP | 파일 | 위치 | 변경 |
| --- | --- | --- | --- |
| WP-1 | `controller/pw_cp.py` | `run()` `:211-222`, `:322-329` | 파라미터 추가 + `_run_loop`로 전달 |
| WP-1 | `controller/pw_cp.py` | `_run_loop()` `:620-627`, `:707-712` | 파라미터 추가 + LCT 계산 분기 |
| WP-2 | `controller/fm_sumtj_cp_lns.py` | `pw_cp()` `:1359-1369`, `:1415-1425` | 파라미터 추가 + `run()`으로 전달 |
| WP-2 | `controller/fm_sumtj_cp_lns.py` | `incremental_pw_cp()` `:1632-1640`, `:1688-1701` | 파라미터 추가 + `build_steps` params로 전달 |

---

## 참조

- 조사/근거: `plans/20260607_pw_cp_increasing_step_investigation.md` (§3 원인, §5 검증, §6 원안)
- `PermutationFlowshopScheduleLite.push_back_tail_jobs_keep_tardiness`: `flowshop_tardiness/fm_prmu.py:164`
- `PwCpConstructor._make_all_dispatched`: `flowshop_tardiness/controller/pw_cp.py:333`
- `PwCpRunState.not_added_first_job`: `flowshop_tardiness/controller/pw_cp.py:125`
- 알고리즘 명세(논문): `~/code/Juntaek-PhD-Thesis/contents/fc_prmu_sumTj.tex` (SW-CP, \(S^R\), \([EST, LCT]\))
