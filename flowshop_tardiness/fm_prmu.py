from itertools import pairwise
from typing import Iterable


class PermutationFlowshopScheduleLight:
    def __init__(
        self,
        stage_name_list: Iterable[str],
        job_2_stage_2_p_map: dict[str, dict[str, int]],
        job_2_due_map: dict[str, int] | None = None,
    ):
        """
        Initialize a PermutationFlowshopScheduleSimple instance.

        Args:
            stage_name_list (list[str]): A list of stage names in the flowshop.
            job_2_stage_2_p_map (dict[str, dict[str, int]]): A nested dictionary mapping
                job names to stage names to processing times.
            job_2_due_map (dict[str, int] | None): An optional dictionary mapping job names to due dates.
                If None, defaults to an empty dictionary.
        """
        self._stage_name_list: list[str] = list(stage_name_list)
        """A list of stage names in the flowshop."""

        self._job_2_stage_2_p_map: dict[str, dict[str, int]] = job_2_stage_2_p_map
        """A nested dictionary mapping job names to stage names to processing times."""

        self._job_2_due_map: dict[str, int] = (
            job_2_due_map if job_2_due_map is not None else {}
        )
        """A dictionary mapping job names to due dates."""

        self._job_seq: list[str] = []
        """A list of job names representing the processing sequence."""

        self._stage_2_job_2_end_map: dict[str, dict[str, int]] = {
            stage_name: {} for stage_name in stage_name_list
        }
        """A nested dictionary mapping job names to stage names to end times."""

    def clear(self) -> None:
        """Clear the schedule."""
        self._job_seq.clear()
        for job_2_end_map in self._stage_2_job_2_end_map.values():
            job_2_end_map.clear()

    def simulate_append(self, job_name: str) -> list[int]:
        """
        Simulate appending a job to the schedule and return the resulting total tardiness.

        Args:
            job_name (str): The name of the job to simulate appending.

        Returns:
            list[int]: The completion time of the appended job at each stage.
        """
        end_time_list: list[int] = []
        last_j_before_append: str | None = self._job_seq[-1] if self._job_seq else None

        i0: str = self._stage_name_list[0]
        this_stage_est: int = (
            0
            if last_j_before_append is None
            else self._stage_2_job_2_end_map[i0][last_j_before_append]
        )
        end_time_list.append(this_stage_est + self._job_2_stage_2_p_map[job_name][i0])

        for prev_i, this_i in pairwise(self._stage_name_list):
            prev_stage_est: int = end_time_list[-1]
            this_stage_est = (
                0
                if last_j_before_append is None
                else self._stage_2_job_2_end_map[this_i][last_j_before_append]
            )
            est: int = (
                this_stage_est if this_stage_est > prev_stage_est else prev_stage_est
            )
            end_time_list.append(est + self._job_2_stage_2_p_map[job_name][this_i])

        return end_time_list

    def append_job(self, job_name: str) -> None:
        """
        Append a job to the schedule.

        Args:
            job_name (str): The name of the job to append.
        """
        end_time_list: list[int] = self.simulate_append(job_name)
        for stage_name, end_time in zip(self._stage_name_list, end_time_list):
            self._stage_2_job_2_end_map[stage_name][job_name] = end_time
        self._job_seq.append(job_name)

    def extend_jobs(self, job_name_list: Iterable[str]) -> None:
        """
        Extend the schedule by appending multiple jobs.

        Args:
            job_name_list (Iterable[str]): An iterable of job names to append.
        """
        for job_name in job_name_list:
            self.append_job(job_name)

    def push_back_tail_jobs_keep_total_tardiness(self, tail_job_cnt: int) -> dict[str, int]:
        """
        Push back the last 'tail_job_cnt' jobs in the schedule
        while keeping the total_tardiness unchanged.mary_

        Args:
            tail_job_cnt (int): The number of jobs to push back from the end of the schedule.

        Raises:
            ValueError: If the tail_job_cnt is invalid.
            ValueError: If the due date map is not provided.
            ValueError: If inconsistent end times are detected.

        Returns:
            dict[str, int]: stage id -> start time of the last-pushed job at that stage
        """
        if tail_job_cnt <= 0 or tail_job_cnt > len(self._job_seq):
            raise ValueError(f"Invalid tail_job_cnt value: {tail_job_cnt}")
        if self._job_2_due_map is None:
            raise ValueError("Due date map is not provided.")

        tail_jobs: list[str] = self._job_seq[-tail_job_cnt:]

        # Iterate over tail_jobs in reverse order
        for j_idx in range(len(tail_jobs) - 1, -1, -1):
            this_j: str = tail_jobs[j_idx]
            next_j: str | None = (
                tail_jobs[j_idx + 1] if j_idx + 1 < len(tail_jobs) else None
            )
            stage_2_p_this_j: dict[str, int] = self._job_2_stage_2_p_map[this_j]
            stage_2_p_next_j: dict[str, int] | None = (
                self._job_2_stage_2_p_map[next_j] if next_j is not None else None
            )

            # Iterate over stages in reverse order
            for i_idx in range(len(self._stage_name_list) - 1, -1, -1):
                this_i: str = self._stage_name_list[i_idx]
                next_i: str | None = (
                    self._stage_name_list[i_idx + 1]
                    if i_idx + 1 < len(self._stage_name_list)
                    else None
                )

                # Distance between this_j and next_j at this stage
                dist_to_next_j: int | None = None
                if next_j is not None and stage_2_p_next_j is not None:
                    dist_to_next_j = (
                        self._stage_2_job_2_end_map[this_i][next_j]
                        - stage_2_p_next_j[this_i]
                        - self._stage_2_job_2_end_map[this_i][this_j]
                    )
                    if dist_to_next_j < 0:
                        raise ValueError(
                            "Inconsistent end times detected when pushing back tail jobs."
                        )

                # Distance between this_i completion time and next_i start time of this job
                # If the last stage, max(0, due date - completion time)
                dist_to_next_i: int
                if next_i is not None:
                    dist_to_next_i = (
                        self._stage_2_job_2_end_map[next_i][this_j]
                        - self._stage_2_job_2_end_map[this_i][this_j]
                        - stage_2_p_this_j[next_i]
                    )
                else:
                    d_j: int = self._job_2_due_map[this_j]
                    if d_j is not None:
                        due_date_room = (
                            d_j - self._stage_2_job_2_end_map[this_i][this_j]
                        )
                        dist_to_next_i = due_date_room if due_date_room >= 0 else 0

                # New end time calculation
                dist_to_next = None
                if dist_to_next_j is not None:
                    dist_to_next = dist_to_next_j
                if dist_to_next is None or dist_to_next_i < dist_to_next:
                    dist_to_next = dist_to_next_i
                self._stage_2_job_2_end_map[this_i][this_j] += dist_to_next

        # Return the start time of the last-pushed job at each stage
        stage_2_start_time_map: dict[str, int] = {}
        for stage_name in self._stage_name_list:
            last_job_name: str = tail_jobs[0]
            end_time: int = self._stage_2_job_2_end_map[stage_name][last_job_name]
            p_time: int = self._job_2_stage_2_p_map[last_job_name][stage_name]
            stage_2_start_time_map[stage_name] = end_time - p_time
        return stage_2_start_time_map
