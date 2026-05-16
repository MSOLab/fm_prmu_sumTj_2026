# P2 (Fm|prmu|sumTj) — Phase Breakdown Data Spec

본 문서는 박사 논문(`Juntaek-PhD-Thesis`)의 P2 장 §"Time–quality trade-off"
그림(`fig/method_avg_time_and_rpdf_p2.pdf`)을 갱신하기 위해, 이 저장소
(`flowshop-tardiness`)에서 **가공해 내보내야 할 phase별 데이터**의 명세이다.

작성일: 2026-05-16
관련: 논문 저장소 `Juntaek-PhD-Thesis/plans/20260516_p2_analysis_data_and_scripts.md`

---

## 1. 목적

§"Time–quality trade-off" 그림은 제안 알고리즘(C4)의 **파이프라인 phase별
(평균 누적 경과시간, 평균 RPDf)** 산점도이다. phase 순서:

```
EDD → NEH-MS → Local search (SW-CP + II 반복) → Base-CP
```

각 phase 종료 시점의 누적 경과시간과 incumbent 목적값이 필요하다. RPDf는
**논문 저장소에서** per-instance Best와 join하여 계산하므로(§4 참조), 이 저장소는
**원시 목적값(obj_value)까지만** 내보낸다.

> **SW-CP와 II를 분리하지 않는다.** 두 연산은 `repeat_while_improvement`
> 안에서 개선이 없을 때까지 **번갈아 반복**된다. 따라서 둘을 분리된 순차
> phase로 두면 누적 시간축에서 왜곡된다 — 예컨대 instance 237은 마지막 II
> 호출이 ~682초, 마지막 SW-CP 호출이 ~787.9초에 끝나, 분리해 기록하면 II가
> 대부분의 시간을 쓴 것처럼 보인다(실제로는 그 반대). 인터리빙되는 두 연산에
> disjoint한 누적 위치를 줄 수 없으므로, **loop 전체를 하나의 phase
> (`local_search`)로 기록한다.** SW-CP의 기여는 ablation(C2 vs C4)에서 이미
> 분리된다.

---

## 2. 대상 실험

- 시나리오: `Outputs_scenarios/20260513T142520_492897/20260512_ablation_c4/`
  (C4 = 제안 알고리즘, 540 인스턴스).
- 인스턴스별 디렉터리: `<insName>/` (`insName = 1..540`).

---

## 3. 내보낼 데이터

### 3.1 출력 파일

`phase_obj_c4.csv` — tidy long 형식, 최대 4×540 = 2160행.

| 컬럼 | 설명 |
|------|------|
| `insName` | 인스턴스 인덱스 (1–540) |
| `phase` | `edd`, `neh_ms`, `local_search`, `base_cp` 중 하나 |
| `end_sec` | 해당 phase 종료 시점의 누적 wall-clock 경과시간(초) |
| `obj_value` | 해당 phase 종료 시점 incumbent 총 지연시간 |

### 3.2 소스 — single source of truth 원칙

이 저장소에는 phase 경계 시각과 best incumbent 목적값을 **둘 다** 정확히
담은 단일 record가 없다. 따라서 컬럼별로 **권위 source를 하나씩** 지정하여
single source of truth 원칙을 유지한다.

| 컬럼 | 권위 source | 이유 |
|------|-------------|------|
| `end_sec` | `<insName>/method_end_time_and_obj_value.csv`의 `method_end_sec` | 개선 여부와 무관하게 **모든 호출의 종료 시각**을 기록 → phase 경계의 권위 source |
| `obj_value` | `<insName>/results/<insName>_obj_log.yaml`의 `obj_value` trace | incumbent가 **개선된 시점만** 기록하는 단조 감소 trace → best incumbent의 권위 source |

두 source는 각자 다른 한 컬럼에만 권위를 가지며, 한 컬럼을 두 source에서
교차 보정하지 않는다.

**`end_sec` 매핑** — `method_end_time_and_obj_value.csv` (컬럼:
`method_name, method_end_sec, objective_value`)에서 phase별로 한 행:

| phase | `method_name` 행 |
|-------|------------------|
| `edd` | `initialize_by_edd` |
| `neh_ms` | `initialize_by_nehms` |
| `local_search` | `repeat_while_improvement` |
| `base_cp` | `solve_base_cp_model` |

- `end_sec` ← 해당 행의 `method_end_sec`.
- 이 CSV의 `objective_value` 컬럼은 **사용하지 않는다** (loop 종료 시점의
  working solution이라 best incumbent와 다를 수 있음 — §3.3 참조).
- `set_random_seed`, `set_cp_model_as_base_cp_model` 행은 sub-초 단위라
  그림에서 제외한다.
- 해당 행이 없거나 `method_end_sec`가 빈 값이면 그 `(insName, phase)`는
  도달하지 않은 것이다 → 행을 생략한다(예: 시간 제한으로 `base_cp` 미도달).

### 3.3 `obj_value` 추출 규칙

`results/<insName>_obj_log.yaml`은 다음 구조다.

- `obj_value.data` — `{경과시간(초, 문자열): 목적값}`. incumbent가 개선된
  시점만 기록하므로 시간순 **단조 비증가** trace다.
- `obj_value.notes` — `{경과시간(초, 문자열): 호출 context}`. 같은 키로
  각 trace 점이 어느 메서드 호출에서 나왔는지 알려준다.

각 trace 점을 `notes` 문자열로 phase에 귀속한다.

| `notes` 문자열에 포함된 토큰 | phase |
|------------------------------|-------|
| `initialize_by_edd` | `edd` |
| `initialize_by_nehms` | `neh_ms` |
| `repeat_while_improvement` | `local_search` |
| `solve_base_cp_model` | `base_cp` |

phase 순위를 `edd < neh_ms < local_search < base_cp`로 두면,

> **phase P의 `obj_value`** = trace 점들 중 **귀속 phase 순위가 P 이하**인
> 점들의 마지막(= 최대 경과시간) 값.

trace가 단조 비증가이므로 이는 곧 **phase P 종료 시점의 best incumbent**다.

**왜 `notes` 귀속이고 timestamp 비교가 아닌가.** CSV의 `method_end_sec`와
YAML trace의 timestamp는 미세하게 어긋난다 — 예: instance 237의
`local_search`는 CSV `method_end_sec`가 787.5(시간제한 값)이지만, loop의
마지막 obj 평가는 YAML에서 787.9에 기록된다. 단순
`timestamp ≤ end_sec` cutoff는 787.9 점을 놓쳐 잘못된 값(900709)을
잡는다. `notes` 귀속은 이 불일치에 영향받지 않고 올바른 값(900258)을 준다.

**점이 없는 phase.** 어떤 phase가 incumbent를 한 번도 개선하지 못하면 그
phase의 trace 점이 없다(예: `base_cp`가 loop best를 못 깸). 위 규칙은
"순위 P 이하 마지막 값"이므로 자동으로 직전 phase의 best incumbent를
준다 — trace 단조성 덕분에 별도 처리가 필요 없다. 단 `edd`는 항상 trace의
첫 점이므로 모든 인스턴스에서 최소 한 점은 존재한다.

> **근본 해법(향후).** `repeat_while_improvement`가 working solution이
> 아니라 best incumbent를 기록하도록 로깅을 고치면 CSV 한 source만으로
> 충분해진다. 현재는 §3.2의 컬럼별 권위 분리로 해결한다.

---

## 4. 다운스트림 (논문 저장소)

- `phase_obj_c4.csv`를 `Juntaek-PhD-Thesis/data/fm_prmu_sumTj/`로 복사·commit.
- 논문 저장소 스크립트가 `phase_obj_c4.csv`를 per-instance Best
  (`algorithm_results.csv` 기반, pool = {C4, GAPR, GA_EDD, MH_X1})와 join하여
  phase별 RPDf를 계산하고, phase별 `(mean end_sec, mean rpdf)`를 그려
  `fig/method_avg_time_and_rpdf_p2.pdf`를 생성한다.
- RPDf 공식: 대칭식 `(Z-Best)/((Z+Best)/2)×100`, `Z=Best=0`이면 0.
- 그림의 phase 라벨은 `Local search`가 SW-CP와 II의 반복임을 캡션에 명시.

---

## 5. 구현 메모

- 신규 스크립트는 `scripts/`에 두고 `uv run`으로 실행.
- 의존성 격리 영역(pandas/numpy/yaml만) 준수. 본 추출은 CSV(end_sec)와
  YAML(obj_value)을 읽으므로 pandas + pyyaml이 필요하다.
- 540 인스턴스 중 일부가 특정 phase에 도달하지 못할 수 있다(예: 시간 제한
  으로 `solve_base_cp_model` 미도달). 그런 경우 해당 `(insName, phase)`
  행은 생략하고, 다운스트림에서 phase별 `n_instances`로 집계 모수를 함께
  보고한다.
