# 조사 + 수정 계획: PW-CP step 간 objective 증가

> 작성 경위: `pw_cp`(SW-CP)의 per-step `*-pw_cp_obj_log.yaml`에서 objective(total tardiness)가
> **감소만 해야 하는데 증가하는** 현상을 발견하여 조사. 조사용 스크립트(`checks/`)와
> `pw_cp.py`의 진단 로깅을 먼저 작성/stage한 뒤, 사후적으로 본 문서로 정리한다.
> 실제 **수정 구현은 별도 대화에서** 수행한다 (§6이 그 구현 명세).

## 1. 현상

- 설정: `Outputs_scenarios/.../20260607_03/subroutine_flow.yaml` (incremental_pw_cp, batch 5→7,
  `improvement_by_insertion_after_every_pw_cp: true`, `repeat_at_end_batch_size_while_improving: true`),
  인스턴스 416 (n=350, c=50).
- `4-incremental_pw_cp.1-incr_pw_cp.1-pw_cp_obj_log.yaml`의 "ObjVal after dispatch"가
  단조 감소하지 않고 여러 지점에서 증가:
  - note 135→140: 2082856 → 2082877 (+21)
  - note 224→229: 2081885 → 2081924 (+39)
  - note 265→270: 2081541 → 2081628 (+87)
- 이 obj_log는 **단일 pw_cp 호출 내부**의 sliding-window iteration 기록(`result.sub_obj_store`)이다.
  즉 한 pw_cp 안에서 commit이 진행되며 full objective가 step 간 증가한다.

## 2. 조사 방법 및 근거 (`checks/`)

repo root에서 `uv run python -m checks.<module>`로 실행(파일 직접 실행 `python checks/x.py`는
sys.path 문제로 실패). `PermutationFlowshopScheduleLite`만 사용한 무작위 검증.

| 스크립트 (`checks/`) | 검증 내용 | 결과 |
| --- | --- | --- |
| `check_push_back_keeps_tardiness.py` | `push_back_tail_jobs_keep_tardiness`가 각 job의 tardiness를 보존하는가 | 2000/2000 보존 (정상) |
| `check_swcp_single_step_monotonic.py` | **단일** batch를 window 안에서 모든 순열로 재정렬했을 때 full tardiness가 incumbent보다 증가하는가 | 0/20000 (단일 step은 안전) |
| `check_swcp_multistep_monotonic.py` | **여러 iteration**에 걸친 sliding-window 전체를 모사하여 step 간 증가가 있는가 | **4/4000 trajectory에서 증가 → 현상 재현** |
| `check_swcp_increase_trace.py` | 재현된 한 사례(trial 764)를 stage별 makespan·LCT까지 상세 추적 | 아래 §3의 최소 사례 |
| `check_swcp_rj_refresh_monotonic.py` | §5 수정안(매 step right-justify refresh) vs 현재 방식의 step 간 증가 비교 | 현재 14/8000 vs **refresh 0/8000** |

핵심: **단일 step은 안전(0/20000)한데 multi-step에서 증가(4/4000)** → 누적(여러 iteration) 상호작용이 원인.

## 3. 근본 원인 (확정)

### 최소 재현 사례 (trial 764, n=8, c=3, B=3)

- it1: batch `[j0,j1,j2]`→`[j1,j0,j2]`, FULL **5** (incumbent 10에서 개선)
- it2: batch `[j3,j4,j5]`→`[j3,j5,j4]`, FULL **6** ← **증가**, 모든 stage에서 makespan ≤ LCT (**window 위반 없음**)

수동 계산:

- `[j3,j4,j5]`와 `[j3,j5,j4]` **둘 다 batch tardiness = 0**.
- CP **phase-2(∑ stage makespan 최소화)** 가 `[j3,j5,j4]`(합 113)를 `[j3,j4,j5]`(합 114)보다 선택.
- 그런데 `[j3,j5,j4]`는 **마지막 stage makespan을 45→47로 올림**(앞 stage를 내리고 마지막을 올려 합은 ↓).
  LCT(s2)=50이라 slack 안이라 합법.
- 마지막 stage makespan↑ → greedy tail(j6,j7)의 release가 늦어짐 → **tail tardiness 0→1** → FULL +1.

### 메커니즘

1. LCT(per-stage 마감)는 right-justified 참조 스케줄 \(S^R\)에서 온다. 그런데 \(S^R\)은
   **원래 incumbent로 단 한 번**(`run()`에서) 계산된다 → 이미 개선된 중간 상태 대비 **slack**을 가진다.
2. CP phase-2는 **합** ∑C_i를 최소화하므로, 앞 stage를 줄이는 대가로 **마지막 stage makespan을
   slack 안에서 올리는** batch 순서를 고를 수 있다.
3. 마지막 stage makespan이 그 step의 직전 상태보다 커지면, 고정 sequence인 tail을 greedy(semi-active,
   그 release에 대해 최적)로 깔아도 release 자체가 밀려 tail tardiness가 증가한다.

### 무엇이 버그가 아닌가 (검증 완료)

- **window bound 강제는 정상**: greedy ≤ CP해 ≤ LCT라 commit된 makespan은 항상 LCT 이내(`window_violations=0`).
- `push_back`은 정상(tardiness 보존).
- prec→sequence 디코드(topological sort)도 정상.
- 즉 **bound·push_back·decode 버그가 아니라, "한 번 계산한 \(S^R\)의 slack을 phase-2가
  마지막 stage에 쓰는" 설계상 비단조성**이다.

### 보장 범위 정리

- pw_cp는 **시작 incumbent에 대해서는** 비증가(≤ incumbent)이다(증명·brute-force 일치).
- 그러나 **자기 내부 iteration 간에는 단조롭지 않다**. obj_log는 내부 iteration을 기록하므로 증가가 보인다.
- 과거 분석에서 두 가지를 정정함: (a) "tail 고정→증가 불가"는 release time이 slack만큼 움직이는 점을 놓침,
  (b) "≤ incumbent" 증명을 "단조 감소"로 착각했던 것.

## 4. 이미 적용한 변경 (staged) — `flowshop_tardiness/controller/pw_cp.py`

조사용 진단 로깅. **알고리즘 동작은 바꾸지 않음** (관찰만).

- `_run_loop` 진입부: `prev_full_obj_val` 추적 변수 추가.
- partition update: 새로 commit된 작업 목록 `newly_committed_jobs` 캡처.
- `--- Monotonicity diagnostics ---` 블록:
  - 매 iter INFO: `full_obj=[committed + tail]  cp(committed+window)=...  window_violations=N`
  - `WINDOW BOUND VIOLATED` WARNING: makespan>LCT면 발화(=진짜 버그 신호, 정상이면 안 나옴)
  - `FULL OBJECTIVE INCREASED` WARNING: 같은 commit 작업들의
    **incumbent-순서 vs CP-순서 마지막 stage makespan vs LCT**를 찍어 "slack을 얼마 썼는지" 노출.
- `_log_snapshot`: 미사용 인자(`iter_report`,`draw_gantt`) 제거, `bound_val`(=committed tardiness)을
  "ObjVal before dispatch" series에 기록(플롯에서 tail 기여 = after−before 가 보이게).

> 참고: 416 재실행 로그(`20260608T015907_.../416/subroutine_controller.log`)에서 증가 iter의
> `makespan(last) X->Y within LCT=Z`가 모두 Y≤Z(마지막 stage 위반 없음)로 확인되어 §3과 일치.

## 5. 제안 수정 방식 검토 — "직전 schedule을 매 step right-justify"

사용자 제안: **매 pw_cp step마다 직전 schedule을 right-justify하여 그로부터 LCT window를 계산.**

- "직전 schedule" = 그 step 시작 시점의 현재 상태 = `committed`(CP 순서) + `remaining`(incumbent 순서), left-justified.
- 이를 right-justify하면 \(S^R\)이 **현재 상태의 tardiness를 보존**하는 기준이 된다.
  따라서 "batch makespan ≤ LCT ⇒ greedy tail ≤ \(S^R\) ⇒ tail tardiness ≤ **직전 상태**의 tail tardiness"가 되어
  **step 간 단조성**이 성립한다 (기존엔 기준이 "원래 incumbent"라 ≤ incumbent만 보장).

### 타당성 검증 (brute-force, `uv run python` 인라인)

8000 trials, 현재 방식 vs 제안(refresh) 비교:

`check_swcp_rj_refresh_monotonic.py` (seed 7, 8000 trials) 출력:

| 방식 | step 간 증가 trajectory 수 |
| --- | --- |
| 현재 (\(S^R\) 1회) | 14 / 8000 |
| **제안 (매 step refresh)** | **0 / 8000** ✅ 단조 |

최종 해 품질 영향: refresh가 현재보다 나쁜 경우 19/8000(≈0.24%), 나머지는 같거나 더 나음.
별도 seed 검증에서도 better 15 / worse 11 / equal 7974, 평균 ≈ 0 (max worse +3, max better −5)로
**단조성을 사실상 품질 손해 없이 획득**.

> 정정: 직전 답변에서 "refresh는 해결 못 함"이라 했으나 **틀렸다.** refresh는 기준 tardiness를
> 현재 상태로 바꾸므로 slack이 "현재 tardiness를 유지하는 양"으로 보정되어 단조성을 준다.

### trade-off / 주의

- refresh는 LCT가 더 tight해져 CP 탐색 자유도가 약간 감소(드물게 최종 해가 ±소량 변동). 위 통계상 무시 가능.
- 비용: 매 iter `_make_all_dispatched`(O(n·c)) + `push_back`(O(n·c)) 추가. n=350,c=50에서 pw_cp당 ~수백만 연산
  → CP solve(10s) 대비 무시 가능.

## 6. 코드 변경 계획 (별도 대화에서 구현)

> 목표: \(S^R\)/LCT 출처를 **"run() 1회 계산한 `given_sol`"** 에서 **"매 iteration 직전 상태를 right-justify"** 로 바꾼다.
> 알고리즘의 나머지(EST, phase1/2, slide 규칙)는 그대로.

### 변경 파일: `flowshop_tardiness/controller/pw_cp.py`

**(A) `_run_loop`의 LCT 계산 교체 — 현재 `:708-713`**

```python
# 현재
if not st.last_job_is_included:
    stage_2_lct_map = given_sol.get_stage_2_start_time_map(st.not_added_first_job)
else:
    stage_2_lct_map = {}
```

→

```python
# 제안: 직전 상태(committed + remaining, incumbent 순서)를 right-justify하여 LCT 산출
if not st.last_job_is_included:
    rj_ref = self._make_all_dispatched(st.time_fixed_sol, st.time_fixed_pool)
    rj_ref.push_back_tail_jobs_keep_tardiness(self.job_cnt)
    stage_2_lct_map = rj_ref.get_stage_2_start_time_map(st.not_added_first_job)
else:
    stage_2_lct_map = {}
```

- `_make_all_dispatched(st.time_fixed_sol, st.time_fixed_pool)`(`:333`)는 이 시점에 commit 전이라
  `committed(CP순서) + remaining(self.job_sequence 순서)` 전체 스케줄을 반환한다(= "직전 schedule").
- 이를 `push_back_tail_jobs_keep_tardiness(self.job_cnt)`로 right-justify 후
  `not_added_first_job`의 stage별 start를 LCT로 읽는다. (검증에서 쓴 로직과 동일.)

**(B) `run()`의 1회성 `given_sol` 정리 — 현재 `:281-293`, `:324`**

- `given_sol`은 더 이상 LCT 출처가 아니다. `run()`의 `given_sol.push_back_tail_jobs_keep_tardiness(...)`(`:287`)와
  `_run_loop(given_sol, ...)`(`:324`) 인자 전달 제거 가능.
- 단 `draw_gantt`일 때 `_0_pushed_back_solution.yaml` 덤프(`:288-293`)는 디버그 산출물이므로 유지하려면
  그 용도로만 `given_sol`을 남기고, `_run_loop` 시그니처에서 `given_sol` 파라미터를 제거(미사용)한다.
- 권장: `_run_loop`의 `given_sol` 파라미터/도크스트링(`:622`,`:635`) 삭제, `run()`에서 draw_gantt 분기 안에서만
  필요시 right-justify 스냅샷을 만들도록 정리(YAGNI: draw_gantt가 False면 불필요).

**(C) 진단 로깅(§4)과의 정합성**

- `window_violations` 체크는 새 LCT 기준으로도 그대로 유효(여전히 0이어야 함).
- `FULL OBJECTIVE INCREASED` WARNING은 수정 후 **발화하지 않아야 정상**(단조성 확보).
  수정 검증의 1차 신호로 사용 가능.

### (선택) 최적화

- (A)의 `_make_all_dispatched`는 iter 끝에서 로깅용으로 한 번 더 계산된다. 직전 iter의 full_sol을
  캐시해 재사용하면 push_back만 추가하면 된다. 단 KISS상 우선 매 iter 재계산으로 구현 권장.

### 검증 (구현 후)

1. `uv run python -m py_compile flowshop_tardiness/controller/pw_cp.py`
2. `uv run python -m checks.check_swcp_rj_refresh_monotonic` → 0 증가 확인
3. 인스턴스 416 재실행 → `subroutine_controller.log`에 `FULL OBJECTIVE INCREASED`가 사라지고
   per-step obj_log가 단조 감소인지 확인. 최종 obj가 수정 전 대비 악화되지 않는지 비교.

### 회귀/주의

- pw_cp 결과(committed sequence)가 바뀌므로 **thesis 실험 수치가 변동**한다 → 재실행 필요.
- profile_fixed_cnt>0 경로도 동일하게 `not_added_first_job` 기준을 쓰므로 구조 변화 없음(현 설정은 0).

## 7. `checks/` 파일명 (적용 완료) 및 실행법

의도 기반 이름으로 `git mv` 완료(공통 접두사 `check_swcp_*`):

| 변경 전 | 변경 후 |
| --- | --- |
| `check_push_back_invariant.py` | `check_push_back_keeps_tardiness.py` |
| `check_full_tardiness_after_window_rearrange.py` | `check_swcp_single_step_monotonic.py` |
| `check_full_tardiness_after_window_rearrange_2.py` | `check_swcp_multistep_monotonic.py` |
| `check_full_tardiness_after_window_rearrange_2_1.py` | `check_swcp_increase_trace.py` |
| `check_full_tardiness_after_window_rearrange_3.py` | `check_swcp_rj_refresh_monotonic.py` |

실행: **repo root에서 `uv run python -m checks.<module>`** (파일 직접 실행은 sys.path 문제로 실패).

5개 모두 드라이버/요약 print 보완 완료. `uv run python -m checks.<module>`로 §2 결과 재현 확인:
push_back 0/2000, single_step 0/20000, multistep 4/4000, refresh 현재 14 vs refresh 0 /8000.

> 공통 접두사 `check_swcp_*`로 묶으면 SW-CP 관련 검증임이 한눈에 보인다.

## 8. 참조 (구현 전 확인)

- `pw_cp.py` `run()`: `:211` / `given_sol` 생성·push_back `:281-293` / `_run_loop` 호출 `:324`
- `pw_cp.py` `_run_loop()`: `:620` / LCT 계산 `:708-713` / `_make_all_dispatched` `:333`
- `PwCpRunState.not_added_first_job`: `:125`
- `PermutationFlowshopScheduleLite.push_back_tail_jobs_keep_tardiness`: `flowshop_tardiness/fm_prmu.py:164`
- 알고리즘 명세(논문): `~/code/Juntaek-PhD-Thesis/contents/fc_prmu_sumTj.tex` (SW-CP, \(S^R\), 윈도우 \([EST, LCT]\))
