from __future__ import annotations

from dataclasses import dataclass

from .fs_subroutine_report import FsSubroutineReport


@dataclass(frozen=True)
class FsSolverReport(FsSubroutineReport):
    """
    Report for subroutines that either initializes a solution
    or improves incumbent solution, specifically using CP-SAT solver.
    """

    obj_value_records: list[tuple[float, float]]
    """
    List of (elapsed time, objective value)

    - Each entry records the state of the solver at a given time.
    - The list may not have the last entry.
    """

    obj_bound_records: list[tuple[float, float]]
    """
    List of (elapsed time, objective bound)

    - Each entry records the state of the solver at a given time.
    - The list may not have the last entry.
    """

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
        d["obj_value_records"] = (
            f'"{self.obj_value_records}"' if self.obj_value_records else ""
        )
        d["obj_bound_records"] = (
            f'"{self.obj_bound_records}"' if self.obj_bound_records else ""
        )
        return d

    def copy(self, **kwargs) -> FsSolverReport:
        """Create a copy of the report, optionally updating fields with new values.

        Args:
            **kwargs: Keyword arguments to update specific fields.

        Returns:
            FsCpsatSolverReport: A new instance of FsCpsatSolverReport with copied or updated fields.
        """
        return FsSolverReport(
            elapsed_time=kwargs.get("elapsed_time", self.elapsed_time),
            obj_value=kwargs.get("obj_value", self.obj_value),
            obj_bound=kwargs.get("obj_bound", self.obj_bound),
            obj_value_records=kwargs.get("obj_value_records", self.obj_value_records),
            obj_bound_records=kwargs.get("obj_bound_records", self.obj_bound_records),
            is_init=kwargs.get("is_init", self.is_init),
        )
