# 정수 완성시간 DP(ΣTj 평가) 가속 계획 — numba-JIT(CPU) 우선, CUDA(batch) 선택

작성일: 2026-06-13
관련 선행 계획: [`20260605_gapr_speedup_option.md`](20260605_gapr_speedup_option.md)
(fv2020/vr2010/none 삽입 speedup 도입 — 본 계획은 그 *알고리즘 선택*과 직교하는 *구현 가속*이다)

---

## 0. 한 줄 요약

`Fm|prmu|ΣTj` 평가의 hot op은 정수 완성시간 점화식
`C[i] = max(prev[i], C[i-1]) + p[i][job]` 이다. **현재 병목은 산술량이 아니라
파이썬 인터프리터 오버헤드**(dict/문자열 키 조회, 객체 할당)이므로,
**CUDA 이전에 (1) 동일 DP를 단일 numba `@njit` 커널로 통합하고 (2) CP-LNS 핫패스의
dict/string 표현을 정수 인덱스로 바꾸는 것**이 가장 싸고 큰 win이다. GPU(순열-간
batch)는 Phase 1 측정 후 *필요할 때만* 진행한다.

**불변식(invariant): 모든 변경은 결과를 1정수도 바꾸지 않는다.** 속도만 바뀐다.
기존 등가성 테스트가 oracle다.

---

## 1. Context — 코드에서 확인한 사실

### 1.1 같은 DP가 4곳 이상에 중복 구현됨 (DRY 위반)

| 위치 | 표현 | 비용/특성 |
|---|---|---|
| `fm_prmu.py:69-126` `simulate_append` | `dict[str,int]`, stage 이름 키 | 가장 느린 per-op |
| `fm_prmu.py:312-346` `get_total_tardiness` | dict | full ΣTj |
| `fm_sumtj_cp_lns.py:444-467` `_simulate_append` | `dict[str,int]` | CP-LNS 핫패스 |
| `fm_sumtj_cp_lns.py:573-673` `_eval_insert_with_criteria` | dict | **position마다 suffix 전체 재계산(O(L²m)) + position마다 `ScheduleMetric.p_ij` dict 재생성(660-669)** |
| `flowshop_new_acc.py` | `list[list[int]]`, 정수 인덱스 | FV2020 가속, full sweep O(Lm) |
| `flowshop_batch_eval.py` | list, 정수 인덱스 | 위를 subseq(block) 삽입으로 일반화. CP-LNS도 사용(`:736,:775`) |
| `flowshop_vr2010_eval.py:43-195` | list | naive O(L²m) / VR2010 O(L²m/2) |
| `tests/test_insertion_speedup_equivalence.py:_full_total_tardiness` | list | brute-force 기준값 |

→ 어떤 가속이든 **단일 소스로 통합**하지 않으면 여러 곳에 중복 적용해야 한다.

### 1.2 스택

- 순수 파이썬. `numpy`는 전이 의존성으로만 존재하고 evaluator는 numpy array조차
  안 쓴다. **numba / cupy / torch / jax 전부 없음** (`pyproject.toml`).
- `cprofile_main_simulate_append.py`가 `_simulate_append`를 콕 집어 프로파일 →
  **본인들이 이미 이 DP를 의심 핫스팟으로 보고 있음.**
- 정수 전용 확인됨(사용자 확인). int32 max는 numba/GPU 모두에서 1-instruction.

### 1.3 진단

정수 `max+add`는 가장 싼 연산이다. 비용의 정체는 dict 조회·문자열 해싱·list
인덱싱·객체 할당 = **원소당 파이썬 오버헤드**. 같은 알고리즘을 컴파일만 해도
보통 10–100×. GPU의 이점(산술 throughput)은 *단일 컴파일 코어 성능을 넘어선
다음*에야 의미가 생긴다. → **컴파일 먼저, GPU는 그 다음.**

---

## 2. 설계 원칙 (전 단계 공통)

1. **결과 불변(behavioral identity).** 모든 커널은 동일 `(best_pos, ΣTj)` /
   동일 `best_pos_list`(오름차순, 최소 position 우선) / 동일 full ΣTj를 정수-정확히
   반환. tie-break 순서까지 보존.
2. **단일 소스(DRY, single source of truth).** DP를 한 모듈에 모으고 모든 호출자가
   그것을 쓴다(§3.1). 가속/검증/GPU 포팅을 한 곳에서만 한다.
3. **opt-in, default 불변(reproducibility).** 진행 중 검증 동안은 토글 가능하게
   두고, 등가성+속도 확인 후 njit을 기본 구현으로 승격. 기존 `speedup`
   selector(fv2020/vr2010/none)는 *알고리즘* 선택이고 njit은 *구현*이라 직교 —
   세 알고리즘 각각에 njit이 적용된다.
4. **KISS/YAGNI.** CPU 컴파일 경로를 소진·측정하기 전에는 GPU를 만들지 않는다.

---

## 3. Phase 1 — CPU 컴파일 + de-dict (주력, GPU 불필요)

### 3.1 단일 DP 커널 모듈 신설 (single source of truth)

신규 `flowshop_tardiness/controller/flowshop_dp.py` (또는 `_kernels.py`):
모든 입력은 numpy 정수 배열. 순수 함수 + `@njit(cache=True)`.

```python
# 계약(서명) — 구현 아님, 합의용
def total_tardiness(p, due, seq) -> int
    # p: int[m,n], due: int[n], seq: int[L]; rolling O(m) 벡터, ΣTj 누적

def best_insertion_naive(p, due, pi, sigma) -> (best_obj, best_pos_earliest, tie_count)
    # position-독립 full recompute. 가장 단순 → 등가성 oracle + Phase 2 GPU seed

def best_insertion_fv2020(p, due, pi, sigma) -> (best_obj, best_pos_earliest, tie_count)
    # flowshop_new_acc.py 의 precompute+find_i_star+calculate_OF_fig10 포팅 (O(Lm))

def best_insertion_subseq(p, due, pi, subseq) -> (best_obj, best_pos_list_buf, n_best)
    # flowshop_batch_eval.py 포팅 (block 삽입)
```

설계 메모:

- **dtype:** 누적은 `int64`(CPU에서 공짜, overflow 0 위험). VR2010/Taillard 규모
  (n≲500, m≲20)면 ΣTj는 int32 여유 내지만 검증으로 확인(§6).
- **tie 리스트:** njit은 가변 list 반환이 번거로움 → `best_pos_earliest`(GAPR가
  읽는 유일 값, `flowshop_vr2010_eval.py:22-23` 참고) + `tie_count`를 반환하거나,
  호출자가 preallocate한 버퍼에 채움. CP-LNS가 전체 tie 리스트를 쓰면 best_obj로
  값싼 2차 패스 수집.
- list↔ndarray 변환은 **호출자 진입 시 1회**. 커널 내부는 ndarray만.

### 3.2 호출자를 단일 커널로 라우팅

| 파일 | 조치 |
|---|---|
| `flowshop_new_acc.py`, `flowshop_batch_eval.py`, `flowshop_vr2010_eval.py` | `get_best_position`은 thin wrapper로 — ndarray 변환 후 §3.1 커널 호출. 클래스 API 유지 |
| `fm_sumtj_cp_lns.py:444-518,573-673` | **de-dict**: `dict[str,int]` stage 맵을 정수 인덱스 배열로. position마다 `p_ij` dict 재생성(660-669) 제거. 가능하면 `_eval_insert_with_criteria`를 이미 있는 subseq 커널로 흡수(중복 제거) |
| `fm_prmu.py` | `get_total_tardiness`/`simulate_append`이 핫패스로 남으면 내부를 §3.1 `total_tardiness`로. (string-키 공개 API는 보존하되 내부 계산만 교체) |

### 3.3 의존성

- `numba`(+`llvmlite`)를 추가. 무게가 있으니 `pyproject.toml`의 **optional extra**
  로(`[project.optional-dependencies] jit = ["numba"]`) 두고, 모듈은 numba 부재 시
  순수-파이썬 reference로 graceful fallback (import 가드).
- py3.11 지원 numba 버전 고정. 첫 호출 컴파일 지연 → `cache=True` + 1회 warm-up.

### 3.4 Phase 1 테스트 (oracle = 기존 테스트, 추가는 등가성)

- **무조건 green 유지:** `tests/test_insertion_speedup_equivalence.py`,
  `test_flowshop_new_acc.py`, `test_flowshop_batch_eval.py`, `test_fm_prmu.py`.
  (이들이 이미 fv2020/vr2010/none/subseq를 brute-force와 정수-정확 대조 → njit
  포팅의 완벽한 안전망)
- **신규** `tests/test_flowshop_dp_kernels.py`: njit 커널 vs 순수-파이썬 reference를
  랜덤 인스턴스(다양한 n,m,due tightness) 다수에서 exact-match. fallback 경로도 테스트.
- **신규** `tests/test_cp_lns_dedict_equivalence.py`: de-dict 전/후 `_eval_insert_with_criteria`
  (또는 통합된 경로)가 동일 `(best_pos, metric.sumTj, makespan)` 반환.

---

## 4. Phase 0 — 측정 (Phase 1 착수 전, 로직 변경 0)

Phase 1의 기대 이득과 Phase 2 필요성을 *수치로* 고정한다.

1. `uv run python cprofile_main_simulate_append.py` 를 대표 인스턴스(작/중/대 n)로
   실행 → `_simulate_append` cumtime 비중 vs CP solve 비중 기록. (이게 작으면 Phase 1
   효과도 작다는 뜻 → 기대치 보정)
2. 마이크로벤치 하니스(`scripts/bench_dp_eval.py`, 신규): 랜덤 인스턴스에서
   `get_best_position`(3 evaluator) 평균 시간을 (n,m) 격자로. Phase 1 전/후 비교 기준선.

---

## 5. Phase 2 — GPU 순열-간 batch (선택, Phase 1 측정 결과에 게이트)

**전제:** Phase 1 후에도 GA population/multistart throughput이 부족할 때만.

- **모델:** thread 1개 = 순열 1개. `p`/`due`는 constant/shared(읽기 전용 공유),
  thread당 길이 m rolling 벡터(레지스터), ΣTj streaming 누적, 마지막에 grid
  reduction으로 argmin. (지난 논의의 커널 구조 그대로)
- **알고리즘↔하드웨어 trade-off:** 삽입 이웃에서는 **FV2020(boundary-walk)이
  position-병렬이 아님**(`calculate_OF_fig10`이 cp/cbar를 suffix 따라 순차 재사용,
  subseq판은 `i_star` 이동). GPU에선 일부러 **position-독립 naive**(`best_insertion_naive`)를
  태운다 — 독립 position × 다수 순열로 GPU 포화, 늘어난 산술량은 공짜.
- **도구:** `numba.cuda`(스택 유지 + Phase 1 정수 커널 재사용, 1순위) >
  `cupy RawKernel` > JAX. numba.cuda면 §3.1 커널과 분기 최소.
- **적용처:** GA offspring/multistart 동시 채점. **CP-SAT(OR-Tools)·CPLEX solve는
  대상 아님** — propagation/search는 GPU 직접 가속 불가. CP-LNS/PW-CP에서 GPU가
  닿는 건 솔버를 감싸는 평가/insertion-improvement 단계뿐.
- **테스트:** GPU 커널 vs Phase 1 njit CPU 커널 exact-match(랜덤 인스턴스).
  CUDA 디바이스 없으면 `pytest.skip`.

---

## 6. 리스크 / 주의

- **int overflow:** 누적은 int64. int32 GPU 경로는 최대 ΣTj(최대 인스턴스 기준)로
  헤드룸 검증 후 채택(빠름). 검증 전엔 int64.
- **tie-break 보존:** `best_pos_list` 오름차순·최소 position 우선이 깨지면 결과가
  미세하게 달라짐 → 테스트로 핀.
- **numba 무게:** llvmlite 포함. optional extra + fallback로 격리.
- **dict 버전을 njit하지 말 것:** numba는 `dict[str]` 비우호적 → **de-dict가 정답**.
- **재현성:** 진행 중 실험과 섞이지 않게 별도 branch(아래)에서만.

---

## 7. Branch & 작업 순서

```bash
# 현재 결과-실험 branch 오염 방지: 현재 HEAD에서 분기(평가 코드가 벤치 대상과 일치)
git switch -c 20260613_numba_dp_eval   # 또는 perf/jit-dp-eval
```

순서:

1. **Phase 0** 측정 → 기준선 수치 확보.
2. **Phase 1.1** `flowshop_dp.py` 순수-파이썬 reference + `test_flowshop_dp_kernels.py`
   (먼저 reference로 등가성 골격, TDD: red→green).
3. **Phase 1.1** 동일 커널에 `@njit` 부여 → 등가성 테스트 그대로 통과 확인.
4. **Phase 1.2** evaluator wrapper 라우팅 → 기존 4개 테스트 green 유지.
5. **Phase 1.2** CP-LNS de-dict → `test_cp_lns_dedict_equivalence` green.
6. **Phase 0 벤치 재실행** → 가속 배수 기록(수용 기준).
7. (게이트) Phase 1 후 throughput 부족 시에만 **Phase 2**.

---

## 8. 수용 기준 (Acceptance)

- 기존 전체 테스트 green (`uv run pytest`).
- 신규 등가성 테스트 green (njit↔reference, de-dict 전후).
- **default 동작(플래그 미지정) 결과 불변** — 한 인스턴스 full 파이프라인 obj_value가
  branch 전후 동일.
- 벤치로 가속 배수 문서화(Phase 0 대비).
- Phase 2 미착수 시에도 Phase 1만으로 닫힌 PR이 되도록 범위 분리.
