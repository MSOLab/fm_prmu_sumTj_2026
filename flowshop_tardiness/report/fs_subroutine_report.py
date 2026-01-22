from dataclasses import dataclass
from typing import TypeVar

from routix.report import SubroutineReport


@dataclass(frozen=True)
class FsSubroutineReport(SubroutineReport):
    """
    Report for subroutines that either initializes a solution
    or improves incumbent solution.

    - is_init: If this report corresponds to initialization, not improvement.
    """

    is_init: bool
    """True if this report corresponds to solution initialization, False otherwise."""

    def to_string_dict(self) -> dict[str, str]:
        """
        Return a dictionary with string representations of each field, suitable for CSV export.

        Returns:
            dict[str, str]: String representations of all report fields.
                - "elapsed_time"
                - "obj_value"
                - "obj_bound"
                - "obj_progress_log"
                - "is_init"
        """
        result = super().to_string_dict()
        result["is_init"] = str(self.is_init)
        return result


FsSubroutineReportT = TypeVar("FsSubroutineReportT", bound=FsSubroutineReport)
"""
Type variable for FsSubroutineReport, allowing methods to specify
that they return or accept an instance of FsSubroutineReport or its subclasses.
"""
