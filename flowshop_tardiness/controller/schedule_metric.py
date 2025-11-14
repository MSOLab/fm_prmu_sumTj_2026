class ScheduleMetric:
    sumTj: int
    """Total tardiness."""

    Cmax_list: list[int]
    """Makespan at each stage."""

    p_ij: dict[tuple[str, str], int]
    """Processing time map: (stage_id, job_id) -> processing time."""

    def __init__(
        self, sumTj: int, Cmax_list: list[int], p_ij: dict[tuple[str, str], int]
    ):
        self.sumTj = sumTj
        self.Cmax_list = Cmax_list
        self.p_ij = p_ij
        self.sum_pij = sum(p_ij.values())
        """Sum of all processing times."""

    @property
    def makespan(self) -> int:
        """Makespan at the last stage."""
        return self.Cmax_list[-1]

    def get_total_idle_time(self) -> int:
        return sum(self.Cmax_list) - self.sum_pij
