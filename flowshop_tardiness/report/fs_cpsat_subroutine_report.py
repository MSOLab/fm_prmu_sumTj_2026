from __future__ import annotations

from dataclasses import dataclass

from mbls.cpsat import CpsatSolverReport, CpsatStatus

from .fs_solver_report import FsSolverReport


@dataclass(frozen=True)
class FsCpsatSolverReport(FsSolverReport):
    """
    Report for subroutines that either initializes a solution
    or improves incumbent solution, specifically using CP-SAT solver.
    """

    status: CpsatStatus
    """Solver status as a CpsatStatus enum."""

    def to_string_dict(self) -> dict[str, str]:
        """
        Return a dictionary with string representations of each field, suitable for CSV export.

        - All values are converted to strings.
        - The status is exported as the standardized status string
          (e.g., "OPTIMAL"), not the enum representation.
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
        d["status"] = self.status.to_solver_status_enum().value
        return d

    @classmethod
    def from_other(
        cls, other: CpsatSolverReport, is_init: bool = False
    ) -> FsCpsatSolverReport:
        """
        Create an FsCpsatSolverReport from a generic CpsatSolverReport.

        Args:
            other (CpsatSolverReport): The original report to convert.
            is_init (bool, optional): Whether this report corresponds to initialization. Defaults to False.

        Returns:
            FsCpsatSolverReport: A new instance of FsCpsatSolverReport created from another CpsatSolverReport.
        """
        return cls(
            elapsed_time=other.elapsed_time,
            obj_value=other.obj_value,
            obj_bound=other.obj_bound,
            obj_value_records=other.obj_value_records,
            obj_bound_records=other.obj_bound_records,
            status=other.status,
            is_init=is_init,
        )

    def copy(self, **kwargs) -> FsCpsatSolverReport:
        """Create a copy of the report, optionally updating fields with new values.

        Args:
            **kwargs: Keyword arguments to update specific fields.

        Returns:
            FsCpsatSolverReport: A new instance of FsCpsatSolverReport with copied or updated fields.
        """
        return FsCpsatSolverReport(
            elapsed_time=kwargs.get("elapsed_time", self.elapsed_time),
            obj_value=kwargs.get("obj_value", self.obj_value),
            obj_bound=kwargs.get("obj_bound", self.obj_bound),
            obj_value_records=kwargs.get("obj_value_records", self.obj_value_records),
            obj_bound_records=kwargs.get("obj_bound_records", self.obj_bound_records),
            status=kwargs.get("status", self.status),
            is_init=kwargs.get("is_init", self.is_init),
        )

    @property
    def is_feasible(self) -> bool:
        """Check if the solution is feasible.

        Returns:
            bool: True if an objective value is available and the status indicates feasibility.
        """
        if self.obj_value is None:
            return False
        return self.status.is_feasible
