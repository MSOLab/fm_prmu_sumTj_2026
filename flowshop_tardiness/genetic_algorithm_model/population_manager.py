from routix import ElapsedTimer

from .trajectory_record import TrajectoryRecord


class PopulationManager:
    def __init__(self, pop_size: int, timer: ElapsedTimer | None = None):
        self.pop_size: int = pop_size
        self.timer: ElapsedTimer = timer or ElapsedTimer()

        self._population: dict[tuple[str, ...], float] = {}
        """Mapping from solution to fitness score."""

        self.generation: int = 0
        self.best_sol: tuple[str, ...] | None = None
        self.best_fitness: float | None = None

        self._trajectory: list[TrajectoryRecord] = []
        """
        List of trajectory records for best solution found so far.
        Each record corresponds to an improvement in the best solution.
        `clear` method does not clear this list.
        """

        self.cumulative_best_sol: tuple[str, ...] | None = None
        """
        Best solution found since the creation of the PopulationManager.
        `clear` method does not reset this value.
        """

        self.cumulative_best_fitness: float | None = None
        """
        Fitness of the best solution found since the creation of the PopulationManager.
        `clear` method does not reset this value.
        """

    # Start getters

    def get_this_population_list(self) -> list[tuple[str, ...]]:
        return list(self._population.keys())

    def get_best_solution(self) -> tuple[str, ...] | None:
        return self.best_sol

    def get_best_fitness(self) -> float | None:
        return self.best_fitness

    def get_last_trajectory_record(self) -> TrajectoryRecord | None:
        if not self._trajectory:
            return None
        return self._trajectory[-1]

    def get_best_obj_series(self) -> list[tuple[float, float]]:
        return [(record.timestamp, record.obj_value) for record in self._trajectory]

    def get_population_set(self) -> set[tuple[str, ...]]:
        return set(self._population.keys())

    def get_worst_fitness(self) -> float | None:
        if not self._population:
            return None
        return max(self._population.values())

    # End getters

    # Start setters

    def clear(self) -> None:
        self._population.clear()
        self.generation = 0
        self.best_sol = None
        self.best_fitness = None

    def add_individual(
        self, solution: tuple[str, ...], fitness: float, source: str
    ) -> None:
        self._population[solution] = fitness
        self._update_best(solution, fitness, source)

    def _update_best(
        self, solution: tuple[str, ...], fitness: float, source: str
    ) -> None:
        if self.best_fitness is None or fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_sol = solution
            self._trajectory.append(
                TrajectoryRecord(
                    timestamp=self.timer.elapsed_sec,
                    obj_value=fitness,
                    generation=self.generation,
                    source=source,
                )
            )
            if (
                self.cumulative_best_fitness is None
                or fitness < self.cumulative_best_fitness
            ):
                self.cumulative_best_fitness = fitness
                self.cumulative_best_sol = solution

    def elitist_replace(self) -> None:
        """From current population leave pop_size number of solutions having the smallest fitness score."""
        sorted_population = sorted(
            self._population.items(), key=lambda item: item[1], reverse=False
        )
        self._population = dict(sorted_population[: self.pop_size])
