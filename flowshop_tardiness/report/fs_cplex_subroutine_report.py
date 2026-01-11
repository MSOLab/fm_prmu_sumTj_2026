from __future__ import annotations

from dataclasses import dataclass

from .fs_solver_report import FsSolverReport


@dataclass(frozen=True)
class FsCplexSolverReport(FsSolverReport):
    """
    Report for subroutines that either initializes a solution
    or improves incumbent solution, specifically using CPLEX solver.
    """

    status: str
    """Solver status as a string."""

    def to_string_dict(self) -> dict[str, str]:
        """
        Return a dictionary with string representations of each field, suitable for CSV export.

        - All values are converted to strings.
        - The status is exported as is.
        - Progress logs are wrapped in double quotes to ensure they are treated as strings in CSV.
          - If the log is empty, the string is empty.

        Returns:
            dict[str, str]: String representations of all report fields.
                - "elapsed_time"
                - "obj_value"
                - "obj_bound"
                - "status"
                - "obj_value_records"
                - "obj_bound_records"
        """
        d = super().to_string_dict()
        d["status"] = self.status
        return d
