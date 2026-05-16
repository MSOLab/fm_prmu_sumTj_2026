from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    # Indices & Parameters

    j_2_job_name_map: dict[int, str]
    """Mapping from job index (j) to job name"""

    job_name_2_j_map: dict[str, int]
    """Mapping from job name to job index (j)"""

    j_list: list[int]
    """$J$: job index (j) list; 0..(n-1)"""

    j_first: int
    """Index of the first job (0)"""

    j_last: int
    """Index of the last job (n-1)"""

    i_2_stage_name_map: dict[int, str]
    """Mapping from stage index (i) to stage name"""

    i_list: list[int]
    """$I$: stage index (i) list; 0..(m-1)"""

    P: dict[tuple[int, int], int]
    """$P_{ij}$: processing time at stage i for job j (P[i, j])"""

    stage_start_time_lb: dict[int, int]
    """i -> lower bound on the start time of the stage."""

    stage_end_time_ub: dict[int, int]
    """i -> upper bound on the makespan of the stage."""

    D: dict[int, int]
    """$D_j$: due date of job j"""
