from .fs_subroutine_report import FsSubroutineReport

__all__ = [
    "FsCpsatSolverReport",
    "FsSubroutineReport",
    "FsSubroutineReportStatistics",
]


def __getattr__(name: str):
    if name == "FsCpsatSolverReport":
        from .fs_cpsat_subroutine_report import FsCpsatSolverReport

        return FsCpsatSolverReport
    if name == "FsSubroutineReportStatistics":
        from .fs_subroutine_report_statistics import FsSubroutineReportStatistics

        return FsSubroutineReportStatistics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
