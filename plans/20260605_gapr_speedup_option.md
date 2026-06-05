# GAPR speedup 옵션화: `vr2010` / `fv2020` / `none`

## Context

현재 저장소의 GAPR (`genetic_algorithm.py:518` `gapr()`, Vallada & Ruiz 2010) 은
local search / NEH 삽입 평가에 **Fernandez-Viagas et al. (2020)** 의 가속법
(`PermutationFlowshopSubseqEvaluator`, `flowshop_batch_eval.py:49`) 을 사용한다.
정방향 DP + **역방향(tail) DP** + Fig.10 경계워크로, 한 job 의 전체 위치 스윕을 싸게 처리한다.

그러나 원논문(*"(2010 Omega) Vallada and Ruiz - Genetic algorithms with path relinking
for the minimum tardiness permutation flowshop problem"*) §3.3 (p.59) 이 기술한 speedup 은
훨씬 단순한 **prefix 부기(bookkeeping)** 다: job 을 빼고 1..n 위치에 순서대로 넣을 때
**삽입점 앞 prefix 의 완료시간만 재사용**하고 **suffix 는 매 위치마다 재계산**한다.
역방향 DP 가 없고, 논문이 명시한 복잡도는 한 job 스윕당
`O(n²m − ((n²−3n+2)/2)·m) ≈ O(n²m/2)` — no-speedup 대비 상수배(~2x) 절감일 뿐
점근적 개선이 아니다 (FV2020 의 핵심 기여가 바로 이 점근 개선).

### 왜 옵션화가 의미 있는가 — 종료 기준이 시간 기반

`gapr()` 의 main loop 는 `while not (is_timeover or is_optimal)` (`genetic_algorithm.py:606`,
종료 체크 `:719` `self.time_is_up()`) 이고, 시간한도는 n×m 에 비례한다
(`configs_vr_2010_gapr/stopping_criteria.yaml: timelimit_n_by_m_multiplier: 0.045`,
논문의 `n·(m/2)·t` 예산과 같은 형태).

→ 세 speedup 은 **출력(각 삽입 위치의 정확한 total tardiness)이 동일**하고 **CPU 비용만 다르다.**
시간예산 하에서는 speedup 이 빠를수록 같은 벽시계 시간에 GA 세대·LS 스텝을 더 많이 돌려
**더 좋은 해**를 낸다. 즉 현재 구현은 "논문 그대로의 GAPR" 이 아니라
"현대 가속법으로 부스트된 GAPR" 이다. 충실 재현을 위해 원논문 speedup 을 선택 가능하게 한다.
(반복 횟수 기반 종료였다면 speedup 은 런타임만 바꾸고 결과는 불변 → 작업 자체가 무의미.)

### 결정 사항 (사용자 확정)

- **3 모드 노출**: `vr2010`(논문 충실), `fv2020`(현재, 기본값), `none`(speedup 없음 = 논문 "GAPR" vs "GAPRsu" 비교용).
- **로컬서치 + NEH 구성 모두**에 적용.
- 기본값 `fv2020` → 기존 config 전부 동작 보존.
- **config 는 변형 디렉토리 생성** (기존 `configs_vr_2010_gapr/` 덮어쓰지 않음).

speedup 이 GA loop 에 들어가는 지점은 두 곳뿐이다 (PR 은 `_evaluate` 전체평가, mutation 은 평가 없음):
- 로컬서치 `_search_insertion_neighborhood` (`genetic_algorithm.py:978`, 호출 `:1005`)
- NEH 구성 `_get_best_pos_list_and_metric_new_acc` (`:821`, 호출 `:850`; `_build_neh_edd_solution` `:808` 가 사용)

---

## 1. 옵션 파라미터 & 디스패치

### 1.1 `gapr()` 시그니처

**파일:** `flowshop_tardiness/controller/genetic_algorithm.py:518`

```python
from typing import Literal

InsertionSpeedup = Literal["fv2020", "vr2010", "none"]

def gapr(
    self,
    P_size: int | None = None,
    div: float | None = None,
    pressure: float | None = None,
    P_m: float | None = None,
    P_ls: float | None = None,
    speedup: InsertionSpeedup = "fv2020",
):
```

본문 defaults 블록(`:540-550`) 뒤에 검증 + 저장:

```python
if speedup not in ("fv2020", "vr2010", "none"):
    raise ValueError(f"Unknown speedup option: {speedup!r}")
self._insertion_speedup: InsertionSpeedup = speedup
```

`Literal` 은 runtime 검증되지 않으므로 (routix dispatcher 가
`getattr(self, name)(**kwargs)` 로 호출 — `subroutine_controller.py:212`) 명시 검증을 둔다.
로그(`:555`)에 speedup 값도 출력.

### 1.2 YAML 디스패치

routix 가 `method: gapr` step 의 나머지 키를 그대로 kwargs 로 넘기므로
**config 로더/파서 변경 불필요.** YAML 에 `speedup: vr2010` 한 줄 추가하면 전달된다.

```yaml
- method: gapr
  P_size: 30
  div: 0.4
  pressure: 0.3
  P_m: 0.02
  P_ls: 0.15
  speedup: vr2010      # 추가
```

### 1.3 영향 범위

`speedup` 은 `gapr()` 전용. `ga_edd()` (Ta et al. 2018) 및 CP-LNS 경로는 불변.

---

## 2. 평가기 인터페이스 통일 & 팩토리

### 2.1 공통 시그니처

세 평가기 모두 기존 FV 와 동일한 계약을 따른다
(`flowshop_batch_eval.py:331`):

```python
def get_best_position(
    self, pi: Sequence[int], subseq: Sequence[int] | int,
    tie_breaker: str = "default",
) -> tuple[list[int], int]:   # (best_pos_list, best_total_tardiness)
```

- 콜러는 `best_pos_list[0]`(= 최소 pos, 동률 시) 만 사용 (`:1012`, `:816`).
- 타이브레이크 규칙도 동일하게: pos 를 0..L 오름차순 순회, strict `<` 면 reset, `==` 면 append
  → `best_pos_list` 는 오름차순, `[0]` 가 최소 pos. (FV 의 `:401-405` 와 일치.)
- GAPR(LS/NEH) 은 `tie_breaker="default"` 만 사용한다. 신규 평가기는 default 만 지원하고
  `"makespan"` 은 `NotImplementedError` (YAGNI; CP-LNS 만 makespan 타이브레이크 사용).

### 2.2 팩토리 `_get_insertion_evaluator()`

**파일:** `genetic_algorithm.py` — 기존 `_get_new_acc_evaluator()` (`:853`) **유지**(fv2020 경로),
그 옆에 신규 팩토리 추가. 반환 형태는 기존과 동일한
`(evaluator, job_id_to_idx, idx_to_job_id)` 튜플로 맞춰 콜러 변경을 최소화.

```python
def _get_insertion_evaluator(self):
    mode = getattr(self, "_insertion_speedup", "fv2020")
    cache = getattr(self, "_insertion_eval_cache", None)
    if cache is None:
        cache = {}
        self._insertion_eval_cache = cache
    if mode in cache:
        return cache[mode]

    # p[m][n], due[n], id<->idx 구성: 기존 _get_new_acc_evaluator 의 :865-886 재사용
    ... (job_id_to_idx, idx_to_job_id, p, due 빌드) ...

    if mode == "fv2020":
        evaluator = PermutationFlowshopSubseqEvaluator(p, due)
    elif mode == "vr2010":
        evaluator = VR2010InsertionEvaluator(p, due)
    elif mode == "none":
        evaluator = NaiveInsertionEvaluator(p, due)
    else:
        raise ValueError(f"Unknown speedup option: {mode!r}")

    cache[mode] = (evaluator, job_id_to_idx, idx_to_job_id)
    return cache[mode]
```

p/due/idx 빌드 로직은 `_get_new_acc_evaluator` 와 중복되므로 **공용 헬퍼**
`_build_pdue_idx()` 로 추출해 둘 다 호출(DRY).

### 2.3 콜러 교체

- `_search_insertion_neighborhood` (`:987`): `self._get_new_acc_evaluator()` → `self._get_insertion_evaluator()`
- `_get_best_pos_list_and_metric_new_acc` (`:843`): 동일 교체

이 두 줄만 바꾸면 LS 와 NEH 가 동시에 선택된 speedup 을 사용한다.

---

## 3. 신규 평가기 2종 (single-job 삽입)

GAPR/NEH 는 항상 **1개 job** 만 삽입한다 (NEH 루프 `:814-817`, LS `:1005` 의 `[job_idx]`).
따라서 신규 평가기는 `len(subseq) == 1` 만 지원하고, 그 외엔 `NotImplementedError`.

### 3.1 `VR2010InsertionEvaluator` (prefix 부기)

**신규 파일:** `flowshop_tardiness/controller/flowshop_vr2010_eval.py`

생성자는 FV 와 동형: `__init__(self, p: Sequence[Sequence[int]], due: Sequence[int])`
(`p[i][j]` = machine i, job idx j; `due[j]`).

```python
def get_best_position(self, pi, subseq, tie_breaker="default"):
    if tie_breaker != "default":
        raise NotImplementedError("vr2010 supports tie_breaker='default' only")
    sigma = subseq[0] if not isinstance(subseq, int) else subseq
    p, due, m, L = self.p, self.due, self.m, len(pi)

    C_prefix = [0] * m          # 마지막 prefix job 의 머신별 완료시간
    prefix_tard = 0
    best_pos_list, best_obj = [], None

    for pos in range(L + 1):
        # (1) prefix 직후에 sigma 삽입 — O(m)
        c = [0] * m
        c[0] = C_prefix[0] + p[0][sigma]
        for i in range(1, m):
            c[i] = (C_prefix[i] if C_prefix[i] > c[i-1] else c[i-1]) + p[i][sigma]
        total = prefix_tard + (c[m-1] - due[sigma] if c[m-1] > due[sigma] else 0)

        # (2) suffix pi[pos:] 를 c 에서부터 재계산 — O((L-pos)·m)
        prev = c
        for job in pi[pos:]:
            cur = [0] * m
            cur[0] = prev[0] + p[0][job]
            for i in range(1, m):
                cur[i] = (prev[i] if prev[i] > cur[i-1] else cur[i-1]) + p[i][job]
            if cur[m-1] > due[job]:
                total += cur[m-1] - due[job]
            prev = cur

        # best 추적 (동률 위치 수집)
        if best_obj is None or total < best_obj:
            best_obj, best_pos_list = total, [pos]
        elif total == best_obj:
            best_pos_list.append(pos)

        # (3) prefix 에 pi[pos] 편입 — O(m)
        if pos < L:
            job = pi[pos]
            nc = [0] * m
            nc[0] = C_prefix[0] + p[0][job]
            for i in range(1, m):
                nc[i] = (C_prefix[i] if C_prefix[i] > nc[i-1] else nc[i-1]) + p[i][job]
            if nc[m-1] > due[job]:
                prefix_tard += nc[m-1] - due[job]
            C_prefix = nc

    return best_pos_list, best_obj
```

- 복잡도: 외부 `L+1` × suffix 합 `Σ(L-pos) = L(L+1)/2` job × m = **`O(L²m/2)`**, prefix 편입 `O(Lm)`.
  → 논문 §3.3 의 `O(n²m/2)` 와 일치. **역방향 tail DP 의도적으로 미사용.**
- prefix 만 재사용, suffix 는 매번 재계산하는 것이 FV2020(역방향 DP 로 각 위치 저비용) 과의 핵심 차이.

### 3.2 `none` — `NaiveInsertionEvaluator`

같은 파일 또는 `flowshop_naive_eval.py`. 기존 테스트 헬퍼
`naive_sum_tardiness` / `naive_completion_times` (`tests/test_flowshop_batch_eval.py:46-70`) 와
동일한 전체 재계산을 매 위치마다 수행:

```python
def get_best_position(self, pi, subseq, tie_breaker="default"):
    if tie_breaker != "default":
        raise NotImplementedError
    sigma = subseq[0] if not isinstance(subseq, int) else subseq
    best_pos_list, best_obj = [], None
    for pos in range(len(pi) + 1):
        seq = list(pi[:pos]) + [sigma] + list(pi[pos:])
        total = self._full_total_tardiness(seq)   # O((L+1)·m)
        if best_obj is None or total < best_obj:
            best_obj, best_pos_list = total, [pos]
        elif total == best_obj:
            best_pos_list.append(pos)
    return best_pos_list, best_obj
```

`_full_total_tardiness` 는 naive_completion_times 와 동일한 DP. 복잡도 `O(L²m)` (가장 느림).

---

## 4. 동치성 테스트 (TDD: red → green)

**신규 파일:** `tests/test_insertion_speedup_equivalence.py`

세 평가기는 *출력이 같아야 하는* 알고리즘이므로 강한 정확성 보증이 가능하다.

- 다양한 `(n, m)` 와 due 분포(빡빡/느슨)의 **무작위 인스턴스 다수** 생성 (seed 고정).
- 각 인스턴스에서 임의 `pi` + 삽입 job `sigma` 에 대해:
  - `fv2020.get_best_position(pi, [sigma])`, `vr2010...`, `none...` 의
    **best_total_tardiness 가 모두 동일**한지 assert.
  - 콜러가 쓰는 `best_pos_list[0]` 가 일치하는지 assert (동률 규칙 동일성).
- ground truth: 기존 `naive_sum_tardiness` 로 `pi[:pos]+[sigma]+pi[pos:]` 전수 계산과 대조.
- 진행: 먼저 `vr2010` 미구현 상태에서 import/실패 확인(red) → 구현 후 green.

(참고: 기존 `tests/test_flowshop_batch_eval.py`, `test_flowshop_new_acc.py` 가 FV 평가기를
naive 와 대조하는 패턴을 이미 갖고 있어 그대로 차용.)

---

## 5. 변형 config 디렉토리

기존 `configs_vr_2010_gapr/` (7× `subroutine_flow_*.yaml` + `stopping_criteria.yaml`,
현재 모두 speedup 키 없음 = 기본 `fv2020`) 는 **변경하지 않는다**.

신규 변형 디렉토리 2개 생성 — 기존 디렉토리 구조 그대로 복제하되 각 `method: gapr` step 에
`speedup:` 한 줄을 추가:

| 디렉토리 | speedup | 논문 대응 |
|---|---|---|
| `configs_vr_2010_gapr/` (기존) | `fv2020` (미명시 기본) | 현대 가속 GAPR |
| `configs_vr_2010_gapr_vr2010/` (신규) | `vr2010` | 논문 "GAPRsu" (충실 재현) |
| `configs_vr_2010_gapr_none/` (신규) | `none` | 논문 "GAPR" (speedup 없음) |

각 신규 디렉토리:
- `stopping_criteria.yaml` 복사 (동일 `timelimit_n_by_m_multiplier`).
- 7개 `subroutine_flow_*.yaml` 복사 + gapr step 에 `speedup: vr2010` / `speedup: none` 추가.
- `ga_ctrlr_metadata.yaml` 에 대응 엔트리 추가
  (`stopping_criteria_rel_path` / `subroutine_flow_rel_path` / `output_dir` 3-튜플,
  기존 `:160-165` 형식; output_dir 는 변형 디렉토리 하위로).

> 이름은 제안값. `_vr2010`/`_none` 대신 `_su`/`_nosu` 등 선호 시 변경 가능.

---

## 6. 검증

- `uv run pytest tests/test_insertion_speedup_equivalence.py` (+ 기존 evaluator 테스트 회귀).
- 소형 단일 인스턴스에 `uv run python` 으로 `gapr(speedup="vr2010")`, `gapr(speedup="none")`
  스모크 1회씩 — 정상 종료 + best obj 기록 확인.
- 회귀: `speedup` 미지정(기본 `fv2020`) config 1개 실행 → 변경 전과 동일 결과(기본값 보존 확인).
- (선택) 동일 소형 인스턴스에서 세 모드의 LS 호출당 시간·세대수 차이 로깅으로
  "vr2010/none 이 더 느려 반복수↓" 가시화.

---

## 7. 작업 순서

1. `flowshop_vr2010_eval.py` (+ naive) 평가기 작성 (§3).
2. `tests/test_insertion_speedup_equivalence.py` 작성 → red 확인 → green (§4).
3. `genetic_algorithm.py`: `gapr()` 시그니처/검증 (§1), `_build_pdue_idx` 추출 +
   `_get_insertion_evaluator()` 팩토리 (§2.2), 두 콜러 교체 (§2.3).
4. 변형 config 디렉토리 2개 + metadata 엔트리 (§5).
5. 검증 (§6).

---

## 8. 미해결 / 주의

- **품질 영향(의도된 결과):** 시간예산 하에서 `vr2010`(LS 한 스윕 `O(L²m/2)`) 와 `none`(`O(L²m)`) 은
  FV2020 보다 느려 반복수↓ → GAPR 결과가 현재보다 나빠진다. 충실 재현의 정확한 귀결.
  특히 `none` 은 대형 인스턴스(n=350)에서 실행시간 주의(논문 비교/디버그용).
- **언어/하드웨어 차이:** 알고리즘(speedup)을 맞춰도 Delphi/Pentium-IV(논문) vs Python/현대 CPU 의
  상수배 차이로 절대 반복수는 논문과 불일치. "충실" = 같은 알고리즘적 speedup 사용이지 논문 수치 재현 아님.
- **변형 디렉토리 명명** (§5) 은 제안값 — 확정 필요.
- 신규 평가기의 `makespan` 타이브레이크 미지원(`NotImplementedError`) — GAPR 미사용이라 의도적.
  추후 CP-LNS 등에서 재사용하려면 별건으로 확장.
