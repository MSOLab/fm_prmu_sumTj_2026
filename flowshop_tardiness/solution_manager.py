from routix.solution_manager import SolutionManager
from routix.util.comparison import float_equals

from .report import FsSubroutineReport
from .scheduling.flowshop_schedule import (
    FlowshopSchedule,
)


class FsSolutionManager(SolutionManager[FsSubroutineReport, FlowshopSchedule]):
    """
    A concrete solution manager for Flowshop Scheduling.

    This class specializes the abstract manager by implementing the comparison
    logic specific to Flowshop Scheduling, which is based on minimizing the total tardiness.
    """

    job_2_duedate_map: dict[str, int]
    """Job names -> due dates"""

    # Start abstract methods

    def _get_obj_value(self, solution: FlowshopSchedule) -> float:
        return float(solution.get_total_tardiness(self.job_2_duedate_map))

    def _a_is_better_obj_value(self, value_a: float, value_b: float | None) -> bool:
        if value_b is None:
            return True
        # False if close enough (considering floating point precision)
        if float_equals(value_a, value_b):
            return False
        # A smaller objective value is better (minimization).
        return value_a < value_b

    def _a_is_better_obj_bound(self, bound_a: float, bound_b: float | None) -> bool:
        if bound_b is None:
            return True
        # False if close enough (considering floating point precision)
        if float_equals(bound_a, bound_b):
            return False
        # For a minimization problem, a higher lower bound is better.
        return bound_a > bound_b

    # End abstract methods

    # Start setters

    def set_job_2_duedate_map(self, job_2_duedate_map: dict[str, int]) -> None:
        self.job_2_duedate_map = job_2_duedate_map.copy()

    # End setters
