# Prefix-Window CP (PW-CP) Algorithm

## 개요

PW-CP는 순열 플로우숍(Permutation Flowshop)에서 총 지연(total tardiness)을 최소화하기 위한 **슬라이딩 윈도우 CP 구성 휴리스틱**이다. 초기 작업 순서(시드 솔루션)를 받아, 앞쪽 작업들을 점진적으로 확정해 나가면서 윈도우 내 작업들에 대해 CP(Constraint Programming) 최적화를 수행한다.

## 전제 (입력)

| 기호 | 의미 |
| --- | --- |
| $J = \{1, \ldots, n\}$ | 작업 집합 |
| $I = \{1, \ldots, m\}$ | 스테이지(기계) 집합 |
| $P_{ij}$ | 스테이지 $i$에서 작업 $j$의 처리 시간 |
| $D_j$ | 작업 $j$의 납기일 |
| $\pi_0$ | 초기 작업 순서 (시드) |

목표: $\sum_j T_j$ 최소화 (단, $T_j = \max(0, C_j^{\text{last}} - D_j)$).

## 작업 분류 (파티션)

알고리즘 실행 중 전체 작업은 세 그룹으로 분류되며, 이 분류가 매 이터레이션마다 갱신된다.

```text
[ time-fixed | profile-fixed | remaining ]
     (확정)      (순서 고정)     (미확정)
```

- **time-fixed**: 시작/종료 시각이 완전히 확정된 작업. 이후 이터레이션에서 CP 모델에 포함되지 않는다.
- **profile-fixed**: CP 윈도우 내에 포함되지만 상대적 순서가 고정된 작업. 절대 시각은 CP가 결정한다.
- **remaining**: 아직 처리되지 않은 후보 작업들.

## 주요 파라미터

| 파라미터 | 기본값 | 설명 |
| --- | --- | --- |
| `added_batch_size` | 1 | 매 이터레이션마다 윈도우에 추가되는 작업 수 |
| `profile_fixed_cnt` | 0 | profile-fixed로 유지할 최대 작업 수 |
| `step_size_on_improve` | `added_batch_size` | CP가 순서를 개선했을 때 확정할 작업 수 |
| `step_size_on_no_improve` | `added_batch_size` | CP가 순서를 개선 못 했을 때 확정할 작업 수 |
| `max_time_per_add` | None | 이터레이션당 CP 시간 제한 |

## 알고리즘 절차

### 사전 처리

1. 초기 순서 $\pi_0$를 받아 `PermutationFlowshopScheduleLite`로 시뮬레이션한다.
2. `push_back_tail_jobs_keep_tardiness`를 적용해 후미 작업들의 LCT(Latest Completion Time) 상한 추정치를 얻는다. 이 값은 이후 `stage_2_lct_map`으로 쓰인다.

### 메인 루프

각 이터레이션 $t$에서:

#### ① 윈도우 구성

```text
CP 대상 = profile_fixed_jobs + added_job_list
         (앞 pf개)              (다음 b개)
```

- `added_job_list` = `remaining_jobs[:added_batch_size]`

#### ② 경계 조건 산출

- `stage_2_est_map`: time-fixed 솔루션의 각 스테이지 완료 시각 → CP의 Earliest Start Time(EST)
- `stage_2_lct_map`: 초기 push-back 솔루션에서 다음 미추가 작업(`not_added_first_job`)의 스테이지 시작 시각 → CP의 Latest Completion Time(LCT) 상한
- `sumTj_offset`: time-fixed 작업들의 누적 tardiness → CP의 총 tardiness 하한 오프셋

#### ③ CP 2단계 풀이 (Lexicographic)

> `_solve_cp_model_lexico_for_batch`

- **Phase 1 — 총 tardiness 최소화**
  - 초기 순서의 tardiness가 이미 0이면 CP 풀이 생략.
  - CP 변수: 간접 선행 변수(indirect precedence) $\text{prec}_{j_1, j_2} \in \{0,1\}$ (positional variable 대신).
  - 목적: $\min \sum_j T_j + \text{sumTj\_offset}$
  - profile-fixed 작업 쌍에는 $\text{prec}_{j,j'} = 1$ 제약을 추가한다.
  - 힌트: 현재 순서(앞→뒤)를 초기 힌트로 제공.
  - 전체 작업이 포함된 마지막 이터레이션에서는 전역 best obj bound로 하한을 추가할 수 있다.

- **Phase 2 — 스테이지별 최대 완료 시각 합 최소화** (부 목적, 마지막 이터레이션 제외)
  - Phase 1의 최적 tardiness를 등호 제약으로 고정한다: $\sum T_j = \text{best\_sumTj}$
  - 목적: $\min \sum_i C_i$ (각 스테이지의 최대 완료 시각 합)
  - 이유: 후속 작업들의 EST를 낮춰 다음 이터레이션의 solution quality 향상.
  - Phase 2가 실패하면 Phase 1 결과를 사용한다.

#### ④ 개선 여부 판단 및 step size 결정

```python
improved = (solver_seq != base_seq)
step_size = step_size_on_improve if improved else step_size_on_no_improve
```

#### ⑤ 파티션 갱신

```python
new_profile_fixed = solver_seq[: prev_pf_len + step_size]

overflow = len(new_profile_fixed) - profile_fixed_cnt
if overflow > 0:
    time_fixed += new_profile_fixed[:overflow]   # 확정
    profile_fixed = new_profile_fixed[overflow:]  # 유지
remaining = remaining - time_fixed - profile_fixed
```

즉, `profile_fixed_cnt` 초과분이 time-fixed로 이동(dispatch)된다.

#### ⑥ 로깅

- 현재 time-fixed + remaining 작업들을 원래 순서로 dispatch한 전체 스케줄의 tardiness를 기록한다.

### 종료 조건

- `remaining_jobs`가 소진되었거나,
- 남은 시간이 0 이하이거나,
- `added_job_list`가 비어 있으면 루프 종료.

### 최종 결과 조합

```python
final_seq = committed_time_fixed_jobs + profile_fixed_jobs + remaining_jobs
```

`FlowshopSchedule`으로 변환 후 반환.

## CP 모델 구조 (cpsat_model_2 / indirect_prec)

| 변수 | 설명 |
| --- | --- |
| $s_{ij}$, $e_{ij}$ | 스테이지 $i$, 작업 $j$의 시작/종료 시각 |
| $\text{prec}_{j_1,j_2} \in \{0,1\}$ | $j_1$이 $j_2$ 앞에 오면 1 (간접 선행) |
| $T_j \geq 0$ | 작업 $j$의 tardiness |
| $C_i$ | 스테이지 $i$의 최대 완료 시각 |

핵심 제약:

- 같은 작업 내 스테이지 연속성: $s_{i,j} + P_{ij} \leq s_{i+1,j}$
- 선행 관계 ↔ 시각: $\text{prec}_{j_1,j_2} = 1 \Rightarrow s_{i,j_2} \geq e_{i,j_1}$
- 상호 배타성: $\text{prec}_{j_1,j_2} + \text{prec}_{j_2,j_1} = 1$
- Tardiness: $T_j \geq e_{\text{last},j} - D_j$

## 다른 문제에 적용 시 고려 사항

1. **목적함수 교체**: `_define_objectives`와 `_solve_cp_model_lexico_for_batch`의 Phase 1/2 목적을 변경하면 다른 기준(weighted tardiness, makespan 등)에 적용 가능하다.

2. **EST/LCT 경계**: `stage_2_est_map` / `stage_2_lct_map`은 플로우숍 구조에 특화되어 있다. 다른 스케줄링 구조에서는 이 경계 추정 방식을 교체해야 한다.

3. **profile_fixed_cnt = 0 (기본값)**: 모든 이터레이션에서 최적화된 작업이 즉시 time-fixed로 이동한다. `profile_fixed_cnt > 0`으로 설정하면 경계 근처 작업들을 다음 이터레이션에서 재최적화할 여지를 남긴다.

4. **Lexicographic Phase 2의 역할**: "앞쪽 스케줄이 빨리 끝날수록 뒷 윈도우에 더 많은 여유"라는 직관을 구현한다. 이 부 목적이 필요 없으면 `last_job_is_included = True`처럼 강제해 Phase 2를 항상 건너뛸 수 있다.

5. **선행 변수 타입**: position 변수($\pi_k$) 대신 간접 선행 변수($\text{prec}_{j_1,j_2}$)를 사용한다. 이는 profile-fixing 제약을 직접 `prec` 값으로 표현할 수 있어 구현이 간결하다.
