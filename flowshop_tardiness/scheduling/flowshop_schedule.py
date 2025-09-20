from __future__ import annotations

from collections import defaultdict

from .exception import SchedulingFailureException
from .flowshop_operation import FlowshopOperation
from .flowshop_stage import FlowshopStage


class FlowshopSchedule:
    def __init__(self) -> None:
        self._stages: dict[str, FlowshopStage] = {}
        """Stage name -> FlowshopStage instance."""

        self._last_stage_name: str | None = None
        """Name of the last stage in the flowshop."""

        # Internal states

        self.job_2_last_oper_end_time_map: dict[str, int] = defaultdict(int)
        """Job name -> end time of the last operation scheduled for that job."""

        self.job_2_scheduled_oper_count_map: dict[str, int] = defaultdict(int)
        """Job name -> count of operations scheduled for that job."""

    @classmethod
    def from_stage_name_list(cls, stage_name_list: list[str]) -> FlowshopSchedule:
        """
        Create a FlowshopSchedule from a list of stage names. Each stage is
        represented by a single-machine `FlowshopStage`.

        Args:
            stage_name_list (list[str]): Ordered list of stage names.

        Returns:
            FlowshopSchedule: A new instance of FlowshopSchedule.
        """
        assert len(stage_name_list) > 0, "Stage name list cannot be empty"

        schedule = cls()
        for stage_name in stage_name_list:
            schedule._stages[stage_name] = FlowshopStage(stage_name)
        schedule._last_stage_name = stage_name_list[-1]
        return schedule

    def deepcopy(self) -> FlowshopSchedule:
        """
        Returns a deep copy of this FlowshopSchedule,
        including all stages and internal state.
        """
        from copy import deepcopy

        new_schedule = FlowshopSchedule()
        # Deep copy stages
        new_schedule._stages = {
            k: v.deepcopy() if hasattr(v, "deepcopy") else deepcopy(v)
            for k, v in self._stages.items()
        }
        # Deep copy internal states
        new_schedule.job_2_last_oper_end_time_map = deepcopy(
            self.job_2_last_oper_end_time_map
        )
        new_schedule.job_2_scheduled_oper_count_map = deepcopy(
            self.job_2_scheduled_oper_count_map
        )
        return new_schedule

    # Start getters

    @property
    def makespan(self) -> int:
        """
        Calculate the makespan of the entire schedule.

        Returns:
            int: The maximum makespan across all stages.
        """
        return max((stage.makespan for stage in self._stages.values()), default=0)


    def get_stage_by_name(self, stage_name: str) -> FlowshopStage:
        """
        Get a stage by its name.

        Args:
            stage_name (str): The name of the stage to retrieve.

        Raises:
            ValueError: If the stage with the given name does not exist in the schedule.

        Returns:
            FlowshopStage: The stage instance with the specified name.
        """
        if stage_name not in self._stages:
            raise ValueError(f"Stage {stage_name} not found in schedule")
        return self._stages[stage_name]

    def get_start_time_map(self) -> dict[tuple[str, str], int]:
        """
        Get a map of (job_name, stage_name) to start time for all operations in the schedule.

        Returns:
            dict[tuple[str, str], int]: A dictionary mapping (job_name, stage_name) to start time.
        """
        return_dict: dict[tuple[str, str], int] = {}
        for stage in self._stages.values():
            stage_start_time_map = stage.get_start_time_map()
            for k, v in stage_start_time_map.items():
                if k in return_dict:
                    raise ValueError(
                        f"Duplicate start time entry for key {k} across stages"
                    )
                return_dict[k] = v
        return return_dict

    def get_end_time_map(self) -> dict[tuple[str, str], int]:
        """
        Get a map of (job_name, stage_name) to end time for all operations in the schedule.

        Returns:
            dict[tuple[str, str], int]: A dictionary mapping (job_name, stage_name) to end time.
        """
        return_dict: dict[tuple[str, str], int] = {}
        for stage in self._stages.values():
            stage_end_time_map = stage.get_end_time_map()
            for k, v in stage_end_time_map.items():
                if k in return_dict:
                    raise ValueError(
                        f"Duplicate end time entry for key {k} across stages"
                    )
                return_dict[k] = v
        return return_dict

    def get_tardiness_map(self, job_2_duedate_map: dict[str, int]) -> dict[str, int]:
        """Calculate the tardiness for each job based on the due dates provided.

        Args:
            job_2_duedate_map (dict[str, int]): A mapping of job names to their due dates.

        Returns:
            dict[str, int]: A dictionary of job names to their tardiness (end_time - due_date)
                for jobs that are late. Jobs that meet or beat their due dates are not included.
        """
        job_2_tardiness_map: dict[str, int] = {}

        assert self._last_stage_name is not None, "Last stage name is not set"

        last_stage = self.get_stage_by_name(self._last_stage_name)
        for operation in last_stage.operations:
            job_name = operation.job_name
            if job_name in job_2_duedate_map:
                due_date = job_2_duedate_map[job_name]
                if operation.end > due_date:
                    tardiness = operation.end - due_date
                    job_2_tardiness_map[job_name] = tardiness
        return job_2_tardiness_map

    def get_total_tardiness(self, job_2_duedate_map: dict[str, int]) -> int:
        """
        Calculate the total tardiness of all jobs in the schedule.

        Args:
            duedate_dict (dict[str, int]): A mapping of job names to their due dates.

        Returns:
            int: The total tardiness across all jobs.
        """
        return sum(self.get_tardiness_map(job_2_duedate_map).values())

    def get_stage_2_job_list_map(self) -> dict[str, list[str]]:
        """
        Get a mapping of stage names to the list of job names scheduled in each stage.

        Returns:
            dict[str, list[str]]: A dictionary mapping stage names to lists of job names.
        """
        return {
            stage_name: [activity.job_name for activity in stage.activity_list]
            for stage_name, stage in self._stages.items()
        }

    # End getters

    # Start setters

    def schedule_operation(
        self, operation: FlowshopOperation, force_add: bool = False
    ) -> FlowshopOperation | None:
        return self.get_stage_by_name(operation.stage_name).add_operation(
            operation, force_add=force_add
        )

    def dispatch_operation_earliest(
        self, job_name: str, stage_name: str, p: int, release_t: int = 0
    ) -> FlowshopOperation:
        """
        Dispatch an operation to the (single machine) stage at the earliest available time.

        Args:
            job_name (str): The name of the job this operation belongs to.
            stage_name (str): The name of the stage this operation belongs to.
            p (int): Processing time.
            release_t (int, optional): Earliest time the operation can start. Defaults to 0.

        Raises:
            SchedulingFailureException: If the operation cannot be scheduled.

        Returns:
            FlowshopOperation: The scheduled operation.
        """
        stage = self.get_stage_by_name(stage_name)
        _release_t = max(release_t, self.job_2_last_oper_end_time_map[job_name])
        start_time = int(stage.get_earliest_start_time(p, _release_t))
        # integer casting to ensure start_time is an integer
        # (not np.int64 for YAML compatibility)
        end_time = int(start_time + p)
        operation = stage.add_operation(
            FlowshopOperation(
                job_name=job_name, stage_name=stage_name, start=start_time, end=end_time
            )
        )
        if operation is None:
            raise SchedulingFailureException(
                f"{job_name}.{stage_name}", p, stage_name, start_time
            )

        # Update internal states
        self.job_2_last_oper_end_time_map[job_name] = operation.end
        self.job_2_scheduled_oper_count_map[job_name] += 1

        return operation

    def dispatch_job_by_stages(
        self,
        job_name: str,
        stage_name_list: list[str],
        stage_name_2_p_map: dict[str, int],
        release_t: int = 0,
    ) -> list[FlowshopOperation]:
        """Dispatch a job across multiple stages, scheduling each stage's operation
        on the earliest available time (respecting job precedence).

        Args:
            job_name (str): The name of the job to be dispatched.
            stage_name_list (list[str]): List of stage names in the order they should be processed.
            stage_name_2_p_map (dict[str, int]): Stage name -> processing time.
            release_t (int, optional): The earliest time the job can start processing at the 1st stage.
                Defaults to 0.

        Returns:
            list[FlowshopOperation]: A list of scheduled operations for the job across all stages.
        """
        operations: list[FlowshopOperation] = []
        last_end_time = max(self.job_2_last_oper_end_time_map[job_name], release_t)

        for stage_name in stage_name_list:
            p = stage_name_2_p_map[stage_name]
            op = self.dispatch_operation_earliest(
                job_name, stage_name, p, last_end_time
            )
            operations.append(op)
            last_end_time = op.end

        return operations

    def remove_operation_from_stage_by_job_name(
        self, stage_name: str, job_name: str
    ) -> bool:
        """Remove an operation from a specific stage by job name.

        Args:
            stage_name (str): The name of the stage to remove the operation from.
            job_name (str): The job name of the operation to remove.

        Returns:
            bool: True if the operation was found and removed, False otherwise.
        """
        return self.get_stage_by_name(stage_name).remove_operation_by_job_name(job_name)

    def remove_operations_by_list_of_job_stage_mc_names(
        self, job_stage_names: list[tuple[str, str]]
    ) -> bool:
        """
        Remove a list of operations from stages by job name & stage name.

        Args:
            job_stage_names (list[tuple[str, str]]): A list of tuples containing
                (job_name, stage_name) for the operations to remove.

        Returns:
            bool: True if all operations were removed, False if any were not found.
        """
        all_removed = True
        for job_name, stage_name in job_stage_names:
            removed = self.remove_operation_from_stage_by_job_name(stage_name, job_name)
            if not removed:
                all_removed = False
        return all_removed

    # End setters
