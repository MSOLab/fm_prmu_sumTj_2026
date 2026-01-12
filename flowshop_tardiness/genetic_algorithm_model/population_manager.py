from routix import ElapsedTimer

from .trajectory_record import TrajectoryRecord


class PopulationManager:
    def __init__(self, pop_size: int, timer: ElapsedTimer | None = None):
        self.pop_size: int = pop_size
        self.timer: ElapsedTimer = timer or ElapsedTimer()

        self._population: dict[tuple[str, ...], float] = {}
        self._trajectory: list[TrajectoryRecord] = []
        self.generation: int = 0
        self.best_sol: tuple[str, ...] | None = None
        self.best_fitness: float | None = None

    # Start getters

    def get_best_solution(self) -> tuple[str, ...] | None:
        return self.best_sol

    def get_best_fitness(self) -> float | None:
        return self.best_fitness

    def get_best_obj_series(self) -> list[tuple[float, float]]:
        return [(record.timestamp, record.obj_value) for record in self._trajectory]

    # End getters

    # Start setters

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

    def elitist_replace(self) -> None:
        """From current population leave pop_size number of solutions having the smallest fitness score."""
        sorted_population = sorted(
            self._population.items(), key=lambda item: item[1], reverse=False
        )
        self._population = dict(sorted_population[: self.pop_size])
