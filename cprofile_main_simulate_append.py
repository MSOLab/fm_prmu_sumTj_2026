import cProfile
import pstats
import time
from functools import wraps

from flowshop_tardiness.controller import FlowshopTardinessCpLnsController
from main import main  # entrypoint의 main()


def _dump_profile(
    profile: cProfile.Profile, base_name: str = "profile_simulate_append"
):
    ts = time.strftime("%Y%m%d_%H%M%S")
    prof_path = f"{base_name}_{ts}.prof"
    txt_path = f"{base_name}_{ts}.txt"

    # .prof 저장 (나중에 다른 정렬로 재분석 가능)
    profile.dump_stats(prof_path)

    # TXT 덤프 (cumtime 내림차순 + callers/callees)
    with open(txt_path, "w") as f:
        ps = pstats.Stats(profile, stream=f).strip_dirs().sort_stats("cumulative")
        ps.print_stats()  # 상위 N개만 보고 싶으면 print_stats(N)
        ps.print_callers(r"_simulate_append")
        ps.print_callees(r"_simulate_append")

    print(f"[cProfile] wrote: {prof_path}, {txt_path}")


def patch_profile_simulate_append(base_name: str = "profile_simulate_append"):
    """
    _simulate_append만 집계하도록 클래스 메서드를 임시 래핑.
    프로그램 전체 실행 동안 해당 메서드 호출들을 하나의 Profile로 '누적' 수집.
    """
    pr = cProfile.Profile()

    # 원본 메서드 보관
    orig = FlowshopTardinessCpLnsController._simulate_append

    @wraps(orig)
    def wrapped(self, *args, **kwargs):
        # 이 호출 1번을 pr에 누적 수집
        return pr.runcall(orig, self, *args, **kwargs)

    # 클래스에 임시 패치
    FlowshopTardinessCpLnsController._simulate_append = wrapped

    def finalize():
        # 패치 원복 + 결과 덤프
        FlowshopTardinessCpLnsController._simulate_append = orig
        _dump_profile(pr, base_name=base_name)

    return finalize


if __name__ == "__main__":
    # 패치 적용
    finalize = patch_profile_simulate_append(base_name="profile_simulate_append")

    try:
        # 🚀 여기서 평소처럼 메인 진입
        main()
    finally:
        # 실행이 끝나면 누적된 _simulate_append 프로파일을 파일로 저장
        finalize()
