# 2026-06-07 실험 정리: 제안 알고리즘 vs VR2010 Best 비교 및 튜닝 시도

> 작성일: 2026-06-08
>
> 관련 커밋: `c903fed` (cp-lns-006nc run config/runner), `b187cce` (incremental_pw_cp), `f1cbce7` (LB-init)
>
> 실험 환경: 본 리포트의 **모든 계산은 `calop4` 서버에서 수행됨**.

## 0. 요약 (TL;DR)

- **제안 알고리즘이 VR2010 Best Solution (Total Tardiness)를 이긴 instance는 0개.** 비기는 경우만 존재하며, 비기는 것은 전부 objective value = 0인 instance.
- 게다가 obj = 0 instance 개수에서도 밀린다: **VR2010은 540개 중 85개가 0, 우리는 80개**(5개 부족).
- 시간 제약 상향(0.045 → 0.06·n·m), batch size 조정(7 vs 8), incremental PW-CP(ISW-CP) 신규 구현까지 시도했으나, 가장 어려운 350×50 그룹에서 **여전히 VR2010 대비 만 단위(평균 약 3.3%) 격차**가 남는다.
- 개선폭은 미미하여 현재로서는 알고리즘 우수성을 주장하기 어려운 상황.

---

## 1. 배경 및 비교 기준

### 1.1 제안 알고리즘

- 설정 파일: [`configs_cp_lns/20260527_ablation_c4.yaml`](../../../configs_cp_lns/20260527_ablation_c4.yaml)
  - 사실상 `20260512_ablation_c4.yaml`과 결과 동일.
- 구성: `NEH-MS 초기해 → preemptive last-stage LB → repeat_while_improvement( pw_cp(batch=7) → improve_by_insertion )`

### 1.2 비교 대상 (VR2010 Best)

- 2010년 논문의 결과로 추정되는 엑셀 파일에 기록된 **Best Solution (Total Tardiness)** 값.
- 출처: <https://web.archive.org/web/20220120203337/http://soa.iti.es/files/Eva_Instances.zip>
- 본 리포트용으로 해당 column만 추출: [`Eva_Instances_EarlinessTardiness.csv`](./Eva_Instances_EarlinessTardiness.csv)
- **추정**: Best column은 *15종 알고리즘 × 3종 time limit(0.03 / 0.045 / 0.06 · n · m)* = **45 trial 중 best 값**으로 보임. → 공정 비교를 위해 우리도 가장 관대한 0.06을 채택(아래 2.1).

### 1.3 벤치마크 구조

전체 540 instance, n×m 12개 그룹 × 각 45개:

| n \ m | 10 | 30 | 50 |
|------:|---:|---:|---:|
| 50  | 45 | 45 | 45 |
| 150 | 45 | 45 | 45 |
| 250 | 45 | 45 | 45 |
| 350 | 45 | 45 | 45 |

---

## 2. 문제 진단: VR2010을 못 이긴다

### 2.1 obj = 0 instance 개수 비교

| | obj = 0 instance 수 |
|---|---:|
| VR2010 Best | **85 / 540** |
| 제안 알고리즘 | **80 / 540** |

- 비기는(=동률) 경우는 모두 objective = 0인 instance에 한정됨.
- 0이 아닌 instance에서는 VR2010 Best를 단 한 번도 이기지 못함.
- obj = 0 개수마저 5개 부족 → "최소한 동급"이라는 주장조차 흔들리는 상황.

---

## 3. 튜닝 시도

테스트 대상은 **objective value가 180만(1.8M)을 넘는 가장 어려운 10개 instance**(전부 350×50 그룹: T=0.6, R∈{0.2, 0.6})로 제한.

### 3.1 시간 제약 상향: 0.045·n·m → 0.06·n·m

- 근거: VR2010 Best가 45 trial의 best이므로(2.1), 가장 긴 time limit인 0.06과 맞추는 것이 공정.
- 설정: [`configs_cp_lns_006nc/stopping_criteria_t120.yaml`](../../../configs_cp_lns_006nc/stopping_criteria_t120.yaml) (`timelimit_n_by_m_multiplier: 0.06`)

### 3.2 batch size 7 vs 8 (preliminary_1)

논문은 batch size 7을 사용. 시간 제약이 늘었으니 8도 시험.
데이터: [`20260607_preliminary_1.csv`](./20260607_preliminary_1.csv)

| insName | n | m | T | R | SW-CP(7) | SW-CP(8) | Best_VR2010 | Δ(7) | Δ(8) |
|---|---|---|---|---|---|---|---|---|---|
| 416 | 350 | 50 | 0.6 | 0.2 | 2,036,495 | 2,045,185 | 1,945,815 | 90,680 | 99,370 |
| 417 | 350 | 50 | 0.6 | 0.2 | 1,990,095 | 2,021,983 | 1,935,135 | 54,960 | 86,848 |
| 418 | 350 | 50 | 0.6 | 0.2 | 1,961,728 | 1,990,082 | 1,897,063 | 64,665 | 93,019 |
| 419 | 350 | 50 | 0.6 | 0.2 | 2,000,514 | 2,019,975 | 1,936,200 | 64,314 | 83,775 |
| 420 | 350 | 50 | 0.6 | 0.2 | 1,902,655 | 1,932,992 | 1,835,320 | 67,335 | 97,672 |
| 476 | 350 | 50 | 0.6 | 0.6 | 1,949,213 | 1,970,143 | 1,884,667 | 64,546 | 85,476 |
| 477 | 350 | 50 | 0.6 | 0.6 | 1,939,652 | 1,952,205 | 1,857,069 | 82,583 | 95,136 |
| 478 | 350 | 50 | 0.6 | 0.6 | 1,886,362 | 1,933,338 | 1,841,663 | 44,699 | 91,675 |
| 479 | 350 | 50 | 0.6 | 0.6 | 1,907,190 | 1,944,000 | 1,838,435 | 68,755 | 105,565 |
| 480 | 350 | 50 | 0.6 | 0.6 | 1,871,750 | 1,899,227 | 1,828,421 | 43,329 | 70,806 |

**요약 통계 (10개 instance)**

| 방법 | 평균 Δ | 최소 Δ | 최대 Δ | 평균 gap(%) |
|---|---:|---:|---:|---:|
| SW-CP(7) | 64,587 | 43,329 | 90,680 | **3.43%** |
| SW-CP(8) | 90,934 | 70,806 | 105,565 | 4.84% |

- **결론: batch size 7이 8보다 일관되게 우수**(10/10). 시간 제약을 늘려도 8은 도움이 안 됨.
- 두 방법 모두 VR2010과 **만 단위 격차** 유지.

### 3.3 incremental PW-CP 구현 + batch ramp 탐색 (preliminary_2)

`incremental_pw_cp` 메서드 신규 구현(커밋 `b187cce`): start_batch_size → end_batch_size로 batch를 점증시키며 각 단계마다 pw_cp 수행, 단계마다 선택적 `improve_by_insertion`, 마지막에 end_batch_size에서 개선되는 동안 반복.

첫 instance(insName=1, 50×10)에서 (start, end) 조합 9종 비교.
데이터: [`20260607_preliminary_2.csv`](./20260607_preliminary_2.csv) — 참고로 이 instance의 VR2010 Best = 1,876.

| start | end | objValue | seconds |
|---:|---:|---:|---:|
| 6 | 7 | 2,401 | 4.38 |
| **5** | **7** | **1,997** | 8.73 |
| 4 | 7 | 2,034 | 5.50 |
| 3 | 7 | 2,034 | 7.26 |
| **3** | **8** | **1,997** | 25.52 |
| 4 | 8 | 2,034 | 20.50 |
| 5 | 8 | 2,017 | 30.83 |
| 6 | 8 | 2,058 | 26.49 |
| 7 | 8 | 2,050 | 18.51 |

- **(5,7)과 (3,8)이 best (obj 1,997)**. 단 (3,8)은 ~3배 더 느림.
- best조차 VR2010 Best(1,876) 대비 Δ121 (≈6.4%) — 작은 instance에서도 격차 존재.
- → 이 두 조합으로 10 instance 확대 시험.

### 3.4 ISW-CP 10 instance 확대 (preliminary_3)

데이터: [`20260607_preliminary_3.csv`](./20260607_preliminary_3.csv) (SW-CP(7)/(8) 열은 3.2의 결과 재사용)

**요약 통계 (10개 instance, 350×50)**

| 방법 | 평균 Δ | 최소 Δ | 최대 Δ | 평균 gap(%) |
|---|---:|---:|---:|---:|
| SW-CP(7) | 64,587 | 43,329 | 90,680 | 3.43% |
| SW-CP(8) | 90,934 | 70,806 | 105,565 | 4.84% |
| **ISW-CP(5~7)** | **62,178** | 42,690 | 93,497 | **3.30%** |
| ISW-CP(3~8) | 63,939 | 27,678 | 85,260 | 3.39% |

**방법별 10개 중 best 횟수**

| 방법 | best 횟수 |
|---|---:|
| SW-CP(7) | 4 |
| ISW-CP(5~7) | 4 |
| ISW-CP(3~8) | 2 |
| SW-CP(8) | 0 |

- ISW-CP(5~7)이 평균적으로 가장 좋지만 SW-CP(7) 대비 개선폭은 평균 **2,409 (전체의 약 0.13%p)**에 불과.
- 특정 instance(예: 476, 480)에서는 ISW-CP(3~8)이 Δ를 크게 줄임(최소 Δ 27,678) → instance별 편차가 큼.
- 그럼에도 **모든 방법이 여전히 VR2010 대비 만 단위 격차.**

### 3.5 LB-init NEH-MS 초기해 품질 (preliminary_4)

개선 단계(pw_cp 등) 없이 **초기화 방식 자체**의 효과를 분리해서 본 실험.
시나리오 `20260607_00` ([`Outputs_scenarios/20260608T011058_692458`](../../../Outputs_scenarios/20260608T011058_692458)): `NEH-MS → compute_preemptive_last_stage_lb(init_method=neh-ms)`.

각 instance마다 **4개의 schedule**이 생성되고, 그중 가장 좋은 것으로 incumbent를 초기화:

1. **NEH-MS(EDD)** — 베이스라인 (`initialize_by_nehms`)
2. **LBinit_start** — LB-init NEH-MS, start time sequence
3. **LBinit_completion** — LB-init NEH-MS, completion time sequence
4. **LBinit_average** — LB-init NEH-MS, average time sequence

데이터: [`20260607_preliminary_4.csv`](./20260607_preliminary_4.csv) — 파서: [`parse_preliminary_4.py`](./parse_preliminary_4.py)

| insName | T | R | NEH-MS(EDD) | best LB-init | seq | Δ(개선) | 개선% |
|---|---|---|---:|---:|---|---:|---:|
| 416 | 0.6 | 0.2 | 2,083,008 | 2,056,623 | completion | **26,385** | 1.27% |
| 417 | 0.6 | 0.2 | 2,073,735 | 2,074,616 | start | −881 | −0.04% |
| 418 | 0.6 | 0.2 | 2,032,458 | 2,018,754 | start | **13,704** | 0.67% |
| 419 | 0.6 | 0.2 | 2,068,657 | 2,075,456 | average | −6,799 | −0.33% |
| 420 | 0.6 | 0.2 | 1,985,440 | 1,972,423 | average | **13,017** | 0.66% |
| 476 | 0.6 | 0.6 | 2,031,355 | 2,037,901 | completion | −6,546 | −0.32% |
| 477 | 0.6 | 0.6 | 2,016,270 | 1,999,386 | average | **16,884** | 0.84% |
| 478 | 0.6 | 0.6 | 2,005,921 | 1,962,728 | average | **43,193** | 2.15% |
| 479 | 0.6 | 0.6 | 2,016,615 | 1,979,995 | completion | **36,620** | 1.82% |
| 480 | 0.6 | 0.6 | 1,966,006 | 1,973,845 | average | −7,839 | −0.40% |

**요약**

- best LB-init이 NEH-MS(EDD)를 이긴 경우: **6 / 10**. 나머지 4개(417·419·476·480)는 오히려 LB-init이 더 나쁨 → 이 경우 incumbent는 NEH-MS(EDD)로 유지.
- 이기는 경우의 개선폭도 작음: **평균 Δ 12,774 (평균 0.63%)**, 최대도 478의 2.15%.
- best 시퀀스 분포: average 5, completion 3, start 2 → **지배적인 시퀀스가 없고 instance별로 갈림**.
- 결론: LB-init NEH-MS가 NEH-MS(EDD)보다 더 좋은 초기해를 주는 경우는 분명 있으나, **그 폭이 작고 일관적이지 않아** 단독 효과로는 아쉬움. VR2010 대비 격차(11만~15만)에는 거의 영향 없음.

---

## 4. 종합 및 다음 단계

### 4.1 결론

- 어떤 튜닝(시간 상향, batch size, incremental ramp)도 가장 어려운 그룹에서 VR2010 Best를 따라잡지 못함.
- ISW-CP가 약간 우수하나 차이는 통계적으로도 의미가 작고(≈0.1%p), 일관되지도 않음.
- batch size 8은 명확히 열등. **batch size 7이 여전히 기본값으로 타당.**

### 4.2 한계 / 가설

- 비교 대상 VR2010 Best가 45 trial의 envelope(best-of-many)라면, 단일 알고리즘·단일 trial로 이를 이기는 것은 구조적으로 불리. → **공정 비교 프레이밍 재검토 필요**(우리도 multi-start/multi-config envelope로 비교하거나, 동일 trial 수 기준으로 비교).
- 큰 instance(350×50)에서 만 단위 격차 → CP 서브문제 크기/시간 배분이 병목일 가능성.

### 4.3 다음 단계 제안

1. **공정 비교 기준 정리**: VR2010 Best가 실제로 몇 trial의 best인지 문헌으로 확정하고, 동일 조건의 비교 테이블 작성.
2. obj ≠ 0 instance에서 격차의 원인 분해(초기해 품질 vs 개선 단계 한계).
3. 부족한 5개 obj=0 instance가 어느 그룹인지 식별 → 해당 그룹 집중 분석.
4. ISW-CP의 instance별 편차가 큰 점을 활용한 adaptive batch ramp 검토.

---

## 부록: 파일 목록

| 파일 | 설명 |
|---|---|
| `Eva_Instances_EarlinessTardiness.csv` | VR2010 Best/Worst (540 instance) |
| `20260607_preliminary_1.csv` | batch 7 vs 8, 10 instance |
| `20260607_preliminary_2.csv` | incremental PW-CP batch ramp 탐색 (instance 1) |
| `20260607_preliminary_3.csv` | SW-CP(7/8) + ISW-CP(5~7 / 3~8), 10 instance |
| `20260607_preliminary_4.csv` | LB-init NEH-MS 초기해 품질 (4개 schedule), 10 instance |
| `parse_preliminary_4.py` | preliminary_4 로그 파서 |
