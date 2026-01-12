from dataclasses import dataclass

@dataclass(frozen=True)
class TrajectoryRecord:
    timestamp: float
    """The time (in seconds) at which the record was created."""

    obj_value: float
    """The objective value at the time of the record."""

    generation: int
    """The generation number at the time of the record."""

    source: str
    """The source or method that generated this record."""
