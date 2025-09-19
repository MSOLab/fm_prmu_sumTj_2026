from __future__ import annotations
from schore.schedule.abstract import Resource

from .flowshop_operation import FlowshopOperation


class FlowshopStage(Resource[FlowshopOperation]):
    def __init__(self, name: str) -> None:
        """Initialize a FlowshopStage.

        Args:
            name (str): The name of the stage.
        """
        super().__init__()
        self._name: str = name
        """The name of the stage."""

    def deepcopy(self) -> FlowshopStage:
        """
        Returns a deep copy of this FlowshopStage,
        including all assigned operations.
        """
        new_ins = FlowshopStage(self._name)
        # Deep copy all assigned operations
        new_ins._activity_list = [op.copy() for op in self.activity_list]
        return new_ins

    # Start required getters

    @property
    def name(self) -> str:
        """Returns the name of the stage."""
        return self._name

    # End required getters

    # Start getters

    @property
    def operations(self) -> list[FlowshopOperation]:
        """Returns the list of operations assigned to this stage.

        Returns:
            list[FlowshopOperation]: The list of operations.
        """
        return self.activity_list

    def get_start_time_map(self) -> dict[tuple[str, str], int]:
        """Get a map of (job_name, stage_name, mc_name) to start time.

        Returns:
            dict[tuple[str, str], int]: A dictionary mapping (job_name, stage_name) to start time.
        """
        return_dict: dict[tuple[str, str], int] = {}
        for operation in self.operations:
            key = (operation.job_name, self.name)
            if key in return_dict:
                raise ValueError(
                    f"Duplicate start time entry for key {key} in stage {self.name}."
                )
            return_dict[key] = operation.start
        return return_dict

    def get_end_time_map(self) -> dict[tuple[str, str], int]:
        """Get a map of (job_name, stage_name) to end time.

        Returns:
            dict[tuple[str, str], int]: A dictionary mapping (job_name, stage_name) to end time.
        """
        return_dict: dict[tuple[str, str], int] = {}
        for operation in self.operations:
            key = (operation.job_name, self.name)
            if key in return_dict:
                raise ValueError(
                    f"Duplicate end time entry for key {key} in stage {self.name}."
                )
            return_dict[key] = operation.end
        return return_dict

    # End getters

    # Start setters

    def add_operation(
        self, operation: FlowshopOperation, force_add: bool = False
    ) -> FlowshopOperation | None:
        """Add an operation to the stage.

        Args:
            operation (FlowshopOperation): The operation to add.
            force_add (bool, optional): If True, force add the operation even if it conflicts with existing operations.
                Defaults to False.

        Raises:
            ValueError: If the operation's stage name does not match this stage's name.

        Returns:
            FlowshopOperation | None: The operation if added successfully, otherwise None.
        """
        if operation.stage_name != self.name:
            raise ValueError(
                f"Operation's stage name {operation.stage_name} does not match"
                f" this stage's name {self.name}."
            )
        if self.add_activity(operation, force_add=force_add) is None:
            return None
        return operation

    def remove_operation_by_job_name(self, job_name: str) -> bool:
        """Remove an operation from the stage by job name.

        Args:
            job_name (str): The job name of the operation to remove.
        """
        for operation in self._activity_list:
            if operation.job_name == job_name:
                self._activity_list.remove(operation)
                return True
        return False

    # End setters
