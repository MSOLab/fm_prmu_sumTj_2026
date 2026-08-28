# `pw_cp` → `sw_cp` 이름 통일 계획 (flowshop-tardiness)

- **날짜**: 2026-07-03 (수행: 2026-08-26)
- **상태**: ✅ 완료 — 실제 수행 결과와 계획 대비 차이는 §7 참조.
- **동기**: 논문에서 이 알고리즘을 **sliding-window CP (`sw_cp`)** 로 표기. 형제 저장소
  `ffc_dw_wET_2026`에서 먼저 수행한 동일 리네임과 이름을 맞춘다.
- **참고**: `~/code/ffc_dw_wET_2026/plans/20260703/pw_cp_to_sw_cp_rename.md` (선행 완료 사례),
  `~/code/hybridflowshop/plans/20260703/pw_cp_to_sw_cp_rename.md`.

> 선행 저장소에서 얻은 교훈: **(a) 내용만 치환하고 파일/디렉토리명을 안 바꾸면 임포트가 깨진다.
> (b) 리터럴 `pw_cp`만 치환하면 CamelCase `PwCp*` 클래스명이 남아 불일치한다. (c) 외부 레포
> 경로·결과물 이름·과거 실기록 인용은 치환하면 사실관계가 틀어진다.** 아래 계획은 이 세 가지를
> 처음부터 반영한다.

---

## 0. 이 저장소의 구조적 특징 (중요)

- **`pw_cp/` 패키지 디렉토리가 없다.** 단일 모듈 `flowshop_tardiness/controller/pw_cp.py`(≈999줄).
  → 디렉토리 `git mv`가 아니라 **파일** `git mv`.
- YAML `method:` 문자열은 외부 `routix` 패키지가 `getattr`로 컨트롤러 메서드에 디스패치한다.
  → `def pw_cp`를 바꾸면 이를 참조하는 **모든 config의 `method: pw_cp` 문자열을 함께 바꿔야**
  런타임에 메서드를 찾는다 (안 바꾸면 "unknown method"로 조용히 skip/실패).
- **ruff가 아예 설정돼 있지 않다** (`pyproject.toml`에 `[tool.ruff]` 없음). isort 활성화 시
  최초로 규칙이 생기므로 리포 전역에서 대량의 import 재정렬이 발생할 수 있다.
- 흥미로운 선행 상태: **문서/일부 config는 이미 "SW-CP"/"ISW-CP"로 개념을 리네임**해 둠
  (`docs/report/20260607/README.md`, `cp_lns_006nc_metadata_20260607.yaml`,
  `configs_cp_lns/20260512_ablation.md`). 코드만 `pw_cp`로 남아있는 상태 → 이 작업이 그 간극을 메움.

## 1. 리네임 대상 (load-bearing — 반드시 함께)

**소스 (파일명 + 내용):**
- `flowshop_tardiness/controller/pw_cp.py` → **`sw_cp.py`** (`git mv`)
  - 내부 클래스: `PwCpContext`, `PwCpRunState`, `PwCpResult`, `PwCpConstructor` → `SwCp*`
  - 독스트링/로그의 `PW-CP` → `SW-CP`
- `flowshop_tardiness/controller/fm_sumtj_cp_lns.py` (파일명 유지, 내용만):
  - `:1417` `from .pw_cp import PwCpConstructor, PwCpResult` → `from .sw_cp import SwCpConstructor, SwCpResult`
  - `:1419` `PwCpConstructor(self)` → `SwCpConstructor(self)`
  - `:1359` `def pw_cp(...)` → `def sw_cp(...)`
  - `:1638` `def incremental_pw_cp(...)` → `def incremental_sw_cp(...)`
  - 로그 태그 `[IncrementalPwCp]` (≈:1729–1767, 6곳) → `[IncrementalSwCp]`
  - `:1433` `logging.info(f"PW-CP done ...")` → `SW-CP`

**설정 (내용 — `method:` 디스패치 문자열, 반드시):**
- `method: pw_cp` — `configs_600s/*.yaml`(34) + `configs_cp_lns/*.yaml`(4) + `configs_cp_lns_006nc/*.yaml` 등 **약 38개 파일**.
- `method: incremental_pw_cp` — `configs_cp_lns_006nc/{20260607_03,04,07,08}.yaml:6`.
- 일괄: `grep -rl 'method: *pw_cp\|method: *incremental_pw_cp' configs*/` 로 목록화 후 치환.
- 함께 바뀌는 keyword argument: `improvement_by_insertion_after_every_pw_cp` →
  `improvement_by_insertion_after_every_sw_cp` (시그니처와 config 양쪽).

**대시보드/분석 (문자열 키 — 기존 `pw_cp` 결과와 새 `sw_cp` 결과를 둘 다 다뤄야 함):**
- `flowshop_tardiness/report/dashboards/_chart_internals.py:42` `"pw_cp": "circle"`
  → **키 추가** 권장 (`"pw_cp"`와 `"sw_cp"` 둘 다), 단순 치환 시 기존 결과 플롯이 마커를 잃음.
  → **실제로는 불필요**했다: 미등록 서브루틴은 렌더 시 `"circle"`로 폴백하는데 `pw_cp`의 값이
  원래 `"circle"`이었다. 단순 치환해도 과거 결과 플롯의 마커는 그대로다.
- `scripts/analysis_metadata.py:19` `("6-pw_cp", "pw_cp")` → 동일하게 둘 다 대응.
  → **실제로는 불필요**했다: `obj_log_top_level_methods`는 소비처가 없는 데드 필드이고
  (`analyzer.py`는 `reactive_loop_report_rel_path`만 사용), 실제 과거 로그 라벨은
  `5-repeat_while_improvement.N-reps.1-pw_cp` 형태라 `6-pw_cp`와 애초에 매칭되지 않았다.
- `flowshop_tardiness/report/dashboards/obj_log_loader.py` — 스텝 라벨 파싱 docstring 예시(4곳).

**테스트 (파일명 + 내용):**
- `tests/test_pw_cp.py` → **`test_sw_cp.py`** (`git mv`); 내부 fixture `def pw_cp`, `PwCp*` import/assert → `sw_cp`/`SwCp*`.

## 2. isort 규칙 추가

`pyproject.toml`에 신규 섹션 추가 (기본 규칙 유지 + isort):
```toml
[tool.ruff.lint]
# Keep ruff's default rules (E4, E7, E9, F) and add isort import sorting.
extend-select = ["I"]
```
- 적용 후 `uv run ruff check --fix` → 리포 전역 import 정렬. **이 저장소는 ruff 최초 도입이므로
  isort 외 default 규칙(F841/E402 등) 위반도 함께 표면화될 수 있음.** rename과 무관한 위반은
  이번 커밋에서 건드리지 말고 별도 처리(문서에 남길 것).
- `sw_cp`는 알파벳순으로 위치가 바뀌므로 import 블록 재정렬 필수.

## 3. 절대 건드리지 말 것 (사실관계 보존)

- **`Outputs_scenarios/`** (≈7GB, `*-pw_cp_obj_log.yaml` 등 38,386개) — 과거 실행 결과물.
  런타임에 당시 `method:`명으로 파일명이 생성됨. 향후 실행분만 `sw_cp`로 나온다.
- **외부 레포 경로 인용**: `plans/20260607_pw_cp_increasing_step_investigation.md:216`,
  `plans/20260607_pw_cp_refresh_deadline_every_step.md:267` →
  `~/code/Juntaek-PhD-Thesis/contents/fc_prmu_sumTj.tex`. 손대지 않음.
- **과거 서술형 plan/보고서** (`plans/20260607_*.md`, `docs/report/20260607/README.md` — 커밋 해시
  `c903fed` 등 인용) — 역사 기록. 인용된 코드/로그 발췌 안의 식별자는 그대로 둔다.
- **주석처리된 블록**: `main_metadata.yaml`의 `output_dir: "output_600s/pw_cp"` 등.
  → 단, 같은 블록의 `subroutine_flow_rel_path`는 실제 config 파일을 가리키므로 파일 리네임과
  **함께** 바꿔야 한다 (파일만 옮기고 경로를 안 바꾸면, 혹은 그 반대면 깨진다).
- `checks/check_swcp_multistep_monotonic.py` — 이미 `swcp`로 이름지어진 독립 검증 스크립트.
  `pw_cp` 모듈을 import하지 않음 → 그대로 둔다. (`checks/check_swcp_*.py` 4개 모두 동일.)

## 4. 문서 (선택)

`docs/algorithm/pw_cp.md`, `plans/20260607_pw_cp_*.md`의 **파일명/내용**은 선택 사항. 대부분 이미
프로즈상 `PW-CP`/`SW-CP` 혼재 → 급하지 않으면 보존, 정리 원하면 별도 단계로.

## 5. 절차 (권장 순서)

1. `git switch -c 20260703_pw_to_sw`
2. `git mv flowshop_tardiness/controller/pw_cp.py flowshop_tardiness/controller/sw_cp.py`
   `git mv tests/test_pw_cp.py tests/test_sw_cp.py`
3. 코드 내용 치환: 위 §1 대상 파일에서 `PwCp`→`SwCp`, `PW-CP`→`SW-CP`, `def pw_cp`→`def sw_cp`,
   `def incremental_pw_cp`→`def incremental_sw_cp`, import 경로 `.pw_cp`→`.sw_cp`.
4. config `method:` 문자열 치환 (§1의 grep 목록 한정).
5. `pyproject.toml`에 §2 isort 추가 → `uv run ruff check --fix` → `uv run ruff format`.
6. 검증: `uv run pytest tests/` (특히 `tests/test_sw_cp.py`),
   샘플 config 1개로 `uv run python main.py` 짧게 실행해 `method: sw_cp` 디스패치 확인.
   **모듈 import 스모크 테스트는 이 리네임의 검증으로 불충분하다** — `fm_sumtj_cp_lns.py`의
   `from .sw_cp import ...`는 메서드 본문 안의 지연 import라, 경로가 깨져 있어도
   `import flowshop_tardiness.controller.fm_sumtj_cp_lns`는 성공한다.

## 6. 착수 전 결정 사항 (2026-08-26 확정)

- **P의 의미**: 문서가 "Prefix-Window"와 "sliding-window"를 혼용 → **`sw_cp`(sliding-window)로 확정.**
  코드/문서의 "Prefix-Window CP" 프로즈도 "Sliding-Window CP"로 통일.
- **대시보드 키**: 양쪽 키 병행 **불필요**(§1 참조). 단순 치환으로 확정.
- **문서 파일명**: `docs/algorithm/pw_cp.md` → `sw_cp.md` **리네임함**.
  `plans/20260607_*.md`는 날짜 기반 역사 기록이므로 **파일명 유지**.

## 7. 실제 수행 결과 (2026-08-26)

**수행한 것:**
- `git mv` 4건: `controller/pw_cp.py`→`sw_cp.py`, `tests/test_pw_cp.py`→`test_sw_cp.py`,
  `configs_600s/subroutine_flow_pw_cp.yaml`→`subroutine_flow_sw_cp.yaml`,
  `docs/algorithm/pw_cp.md`→`sw_cp.md`.
- `PwCp*`→`SwCp*` 클래스 4종, `PW-CP`→`SW-CP`, "Prefix-Window CP"→"Sliding-Window CP" 치환.
- config `method:`/keyword argument 치환 (§1).
- 과거 문서(`plans/20260607_*.md`, `plans/20260527_*.md`, `docs/report/20260607/README.md`)는
  **본문 서술만 새 이름으로 갱신하고, 상단에 리네임 주석 배너를 병기**했다. 당시 생성된 결과
  파일명·커밋 내용 인용은 옛 이름(`pw_cp`)으로 되돌렸다.
- 검증: `uv run pytest tests/` 77 passed.

**하지 않은 것:**
- **§2 isort 도입 안 함.** 리네임과 무관한 리포 전역 import 재정렬을 같은 커밋에 섞지 않기 위해
  별도 작업으로 미룸. `pyproject.toml`은 손대지 않았다.
- `Outputs_scenarios/`의 과거 결과 파일명(`*-pw_cp_obj_log.yaml` 38,386개)은 그대로 둔다.
