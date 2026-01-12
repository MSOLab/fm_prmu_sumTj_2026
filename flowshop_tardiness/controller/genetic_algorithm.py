from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Sequence

from routix import DynamicDataObject, StoppingCriteria
from schore.parameters_examples.shop.flow import FlowshopDuedateParameters
from schore.schedule_examples.shop.flow import FlowshopOperation, FlowshopSchedule

from flowshop_tardiness.fm_prmu import PermutationFlowshopScheduleLite
from flowshop_tardiness.report import FsSubroutineReport

from ..genetic_algorithm_model import PopulationManager
from .base_flowshop_controller import BaseFlowshopController


class FlowshopTardinessGeneticAlgorithmController(BaseFlowshopController):
    def __init__(
        self,
        instance: FlowshopDuedateParameters,
        shared_param_dict: dict,
        subroutine_flow: Sequence[DynamicDataObject] | DynamicDataObject,
        stopping_criteria: StoppingCriteria,
    ):
        super().__init__(
            instance=instance,
            shared_param_dict=shared_param_dict,
            subroutine_flow=subroutine_flow,
            stopping_criteria=stopping_criteria,
        )
        self.method_names_to_run_before_resume = {"set_random_seed"}

    # Start subalgorithm definition

    def ga_edd(self, pop_size: int, cross_size: int, mut_size: int):
        _cross_cnt = int(cross_size / 2)  # two children per crossover
        # Initialize population manager
        self.population_manager = PopulationManager(pop_size, timer=self.timer)
        pop_mgr = self.population_manager
        # job_set = set(self.instance.job_id_list)  # for sanity check

        # Initialize population
        self._initialize_population()
        self._log_best_fitness_to_obj_store("1st")
        pop_mgr.generation = 1
        record = pop_mgr.get_last_trajectory_record()
        if record is None:
            raise RuntimeError("No trajectory record found after initialization.")
        logging.info(
            f"Gen {record.generation} at {record.timestamp:.2f}: "
            f"Best obj (sumTj) = {record.obj_value} by {record.source}"
        )

        time_over = self.time_is_up()
        # Start main loop
        while not time_over:
            pop_mgr.generation += 1
            P_prev = pop_mgr.get_this_population_list()
            if not P_prev:
                raise RuntimeError("Population is empty, cannot proceed with GA.")

            # Crossover
            if len(P_prev) >= 2:
                for _ in range(_cross_cnt):
                    # Select two parents
                    parent_sols = random.sample(P_prev, 2)
                    sol1 = parent_sols[0]
                    sol2 = parent_sols[1]

                    cross_type = random.choice(["X1", "LOX"])
                    if cross_type == "X1":
                        # One-point crossover (X1)
                        child_sol1, child_sol2 = self._crossover_one_point(sol1, sol2)

                        # Optional sanity checks (can be commented out for performance)
                        # assert set(child_sol1) == job_set, (
                        #     "Invalid child solution generated."
                        # )
                        # assert set(child_sol2) == job_set, (
                        #     "Invalid child solution generated."
                        # )
                        # assert len(set(child_sol1)) == len(child_sol1), (
                        #     "Duplicate jobs in child solution."
                        # )
                        # assert len(set(child_sol2)) == len(child_sol2), (
                        #     "Duplicate jobs in child solution."
                        # )
                    else:
                        # Linear order crossover (LOX)
                        child_sol1, child_sol2 = self._crossover_linear_order(
                            sol1, sol2
                        )

                        # Optional sanity checks (can be commented out for performance)
                        # assert set(child_sol1) == job_set, (
                        #     "Invalid child solution generated."
                        # )
                        # assert set(child_sol2) == job_set, (
                        #     "Invalid child solution generated."
                        # )
                        # assert len(set(child_sol1)) == len(child_sol1), (
                        #     "Duplicate jobs in child solution."
                        # )
                        # assert len(set(child_sol2)) == len(child_sol2), (
                        #     "Duplicate jobs in child solution."
                        # )

                    pop_mgr.add_individual(
                        child_sol1, self._evaluate(child_sol1), source="X1"
                    )
                    pop_mgr.add_individual(
                        child_sol2, self._evaluate(child_sol2), source="X1"
                    )
                    time_over = self.time_is_up()
                    if time_over:
                        break

            if time_over:
                updated = self._log_best_fitness_to_obj_store("TIME UP")
                if updated:
                    record = pop_mgr.get_last_trajectory_record()
                    if record is None:
                        raise RuntimeError("No trajectory record found after update.")
                    logging.info(
                        f"Gen {record.generation} at {record.timestamp:.2f}: "
                        f"Best obj (sumTj) = {record.obj_value} by {record.source}"
                    )
                break

            # Mutation
            if P_prev:
                for _ in range(mut_size):
                    parent_sol = random.choice(P_prev)
                    mutation_type = random.choice(
                        [
                            "FI",
                            "BI",
                        ]
                    )
                    if mutation_type == "FI":
                        child_sol = self._mutate_forward_insertion(parent_sol)
                    else:  # backward_insertion
                        child_sol = self._mutate_backward_insertion(parent_sol)

                    pop_mgr.add_individual(
                        child_sol, self._evaluate(child_sol), source=mutation_type
                    )
                    time_over = self.time_is_up()
                    if time_over:
                        break

                if time_over:
                    updated = self._log_best_fitness_to_obj_store("TIME UP")
                    if updated:
                        record = pop_mgr.get_last_trajectory_record()
                        if record is None:
                            raise RuntimeError(
                                "No trajectory record found after update."
                            )
                        logging.info(
                            f"Gen {record.generation} at {record.timestamp:.2f}: "
                            f"Best obj (sumTj) = {record.obj_value} by {record.source}"
                        )
                    break

            # Elitist replacement
            pop_mgr.elitist_replace()
            updated = self._log_best_fitness_to_obj_store(str(pop_mgr.generation))
            if updated:
                record = pop_mgr.get_last_trajectory_record()
                if record is None:
                    raise RuntimeError("No trajectory record found after update.")
                logging.info(
                    f"Gen {record.generation} at {record.timestamp:.2f}: "
                    f"Best obj (sumTj) = {record.obj_value} by {record.source}"
                )

            # Check time limit for next generation
            time_over = self.time_is_up()

        # Wrap up
        # Update objective store
        timestamp_obj_value_list = pop_mgr.get_best_obj_series()
        for timestamp, obj_value in timestamp_obj_value_list:
            last_obj_value = self.obj_store.get_last_obj_value()
            if last_obj_value is None or obj_value < last_obj_value:
                logging.info(
                    f"Gen {record.generation} at {record.timestamp:.2f}: "
                    f"Best obj (sumTj) = {record.obj_value} by {record.source}"
                )
            self.obj_store.add_obj_value(
                timestamp=timestamp,
                value=obj_value,
                is_maximize=False,
            )

        # Register final solution
        obj_value = pop_mgr.get_best_fitness()
        last_job_seq = pop_mgr.get_best_solution()
        report = FsSubroutineReport(
            elapsed_time=self.timer.elapsed_sec,
            obj_value=obj_value,
            obj_bound=None,
            is_init=False,
        )
        if last_job_seq is None:
            raise RuntimeError("No best solution found in population manager.")
        last_schedule = self._dispatch_permutation(list(last_job_seq))
        self.solution_manager.register(report, last_schedule)

    # End subalgorithm definition

    # Start subalgorithm helper methods

    def _evaluate(self, solution: tuple[str, ...]) -> int:
        schedule = PermutationFlowshopScheduleLite(
            self.stage_ids, self.job_2_stage_2_p_dict, self.instance.job_2_duedate_map
        )
        schedule.extend_jobs(solution)
        return schedule.get_total_tardiness()

    def _build_edd_solution(self) -> tuple[str, ...]:
        sorted_jobs = sorted(
            self.instance.job_2_duedate_map.items(), key=lambda item: item[1]
        )
        return tuple(job_id for job_id, _ in sorted_jobs)

    def _get_random_solutions(self, count: int) -> list[tuple[str, ...]]:
        job_ids = self.instance.job_id_list
        n = len(job_ids)
        solutions: set[tuple[str, ...]] = set()
        while len(solutions) < count:
            solutions.add(tuple(random.sample(job_ids, k=n)))
        return list(solutions)

    def _initialize_population(self) -> None:
        mgr = self.population_manager
        # Initialize population
        edd_solution = self._build_edd_solution()

        # One solution by EDD
        edd_fitness = self._evaluate(edd_solution)
        self.population_manager.add_individual(
            edd_solution, edd_fitness, source="EDD_INIT"
        )

        # Remaining solution by random permutations
        random_solutions = self._get_random_solutions(mgr.pop_size - 1)
        for sol in random_solutions:
            fitness = self._evaluate(sol)
            self.population_manager.add_individual(sol, fitness, source="RAND_INIT")

    def _crossover_one_point(
        self, parent_a: tuple[str, ...], parent_b: tuple[str, ...]
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """
        X1: one-point crossover (paper definition).

        Let A = A1//A2 and B = B1//B2 by a single crossover point c.
        Offspring O1: jobs of A1 in order of A, then jobs of A2 in order of B.
        Offspring O2: jobs of B1 in order of B, then jobs of B2 in order of A.
        """
        n = len(parent_a)
        if n != len(parent_b):
            raise ValueError("Parents must have the same length.")
        if n <= 1:
            return parent_a, parent_b

        c = random.randrange(1, n)  # split point (1..n-1)

        A1 = parent_a[:c]
        B1 = parent_b[:c]

        # O1: A1 + (remaining jobs in order of B)
        used1 = set(A1)
        tail1 = [j for j in parent_b if j not in used1]
        o1 = tuple(A1 + tuple(tail1))

        # O2: B1 + (remaining jobs in order of A)
        used2 = set(B1)
        tail2 = [j for j in parent_a if j not in used2]
        o2 = tuple(B1 + tuple(tail2))

        return o1, o2

    def _crossover_linear_order(
        self, parent_a: tuple[str, ...], parent_b: tuple[str, ...]
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """
        LOX: linear order crossover (paper definition).

        Choose two crossover points a < b.
        Let A = A1//A2//A3, B = B1//B2//B3 (A2 and B2 are the middle segments).

        Offspring O1:
        - Middle: jobs of A2 in order of A (fixed in positions a..b-1)
        - Fill remaining positions (first part + last part) with jobs of A1 \cup A3
            in the order of B.

        Offspring O2:
        - Middle: jobs of B2 in order of B (fixed in positions a..b-1)
        - Fill remaining positions with jobs of B1 \cup B3 in the order of A.
        """
        n = len(parent_a)
        if n != len(parent_b):
            raise ValueError("Parents must have the same length.")
        if n <= 2:
            # LOX needs meaningful middle; fallback to X1-style behavior
            return self._crossover_one_point(parent_a, parent_b)

        a = random.randrange(0, n - 1)
        b = random.randrange(a + 1, n)  # middle is [a, b)

        # --- O1 ---
        child1: list[str | None] = [None] * n
        mid1 = parent_a[a:b]  # A2
        child1[a:b] = list(mid1)
        used_mid1 = set(mid1)

        # positions outside middle: first then last
        outside_positions = list(range(0, a)) + list(range(b, n))

        fill_jobs1 = [
            j for j in parent_b if j not in used_mid1
        ]  # A1 \cup A3 in order of B
        # if len(fill_jobs1) != len(outside_positions):
        #     raise RuntimeError("LOX internal error for O1 (size mismatch).")

        for pos, job in zip(outside_positions, fill_jobs1):
            child1[pos] = job

        o1: tuple[str, ...] = tuple(child1)  # type: ignore[arg-type]

        # --- O2 ---
        child2: list[str | None] = [None] * n
        mid2 = parent_b[a:b]  # B2
        child2[a:b] = list(mid2)
        used_mid2 = set(mid2)

        fill_jobs2 = [
            j for j in parent_a if j not in used_mid2
        ]  # B1 \cup B3 in order of A
        # if len(fill_jobs2) != len(outside_positions):
        #     raise RuntimeError("LOX internal error for O2 (size mismatch).")

        for pos, job in zip(outside_positions, fill_jobs2):
            child2[pos] = job

        o2: tuple[str, ...] = tuple(child2)  # type: ignore[arg-type]

        return o1, o2

    def _mutate_forward_insertion(self, solution: tuple[str, ...]) -> tuple[str, ...]:
        """
        Forward insertion (FI):
        choose i < j, remove job at i and insert it at j.
        """
        sol: list[str] = list(solution)
        n: int = len(sol)
        if n <= 1:
            return solution

        i: int = random.randrange(0, n - 1)
        j: int = random.randrange(i + 1, n)

        job: str = sol.pop(i)
        sol.insert(j, job)
        return tuple(sol)

    def _mutate_backward_insertion(self, solution: tuple[str, ...]) -> tuple[str, ...]:
        """
        Backward insertion (BI):
        choose i > j, remove job at i and insert it at j.
        """
        sol: list[str] = list(solution)
        n: int = len(sol)
        if n <= 1:
            return solution

        i: int = random.randrange(1, n)
        j: int = random.randrange(0, i)

        job: str = sol.pop(i)
        sol.insert(j, job)
        return tuple(sol)

    # End subalgorithm helper methods

    # Start logging helper methods

    def _log_best_fitness_to_obj_store(self, timestamp_note: str) -> bool:
        """Log the best fitness in the population manager to the objective store.

        Args:
            timestamp_note (str): Note to associate with the timestamp in the objective store.

        Returns:
            bool: True if the best fitness was updated, False otherwise.
        """
        return_val = False

        log_time = self.timer.elapsed_sec
        obj_value = self.population_manager.get_best_fitness()
        if obj_value is not None:
            last_obj_value = self.obj_store.get_last_obj_value()
            if last_obj_value is None or obj_value < last_obj_value:
                return_val = True
            self.obj_store.add_obj_value(log_time, obj_value, is_maximize=False)
            self.obj_store.add_last_timestamp_note(
                timestamp_note,
                obj_value_is_valid=True,
            )
        return return_val

    def _dispatch_permutation(self, job_sequence: list[str]) -> FlowshopSchedule:
        """Create a permutation flow shop schedule by serial dispatch."""
        schedule = FlowshopSchedule.from_stage_name_list(self.stage_ids)

        # completion frontier per stage
        stage_2_end: dict[str, int] = {i: 0 for i in self.stage_ids}

        for job_id in job_sequence:
            prev_end = 0
            for stage_id in self.stage_ids:
                p = int(self.job_2_stage_2_p_dict[job_id][stage_id])
                start = max(stage_2_end[stage_id], prev_end)
                end = start + p

                stage = schedule.get_stage_by_name(stage_id)
                op = stage.add_operation(
                    FlowshopOperation(
                        job_name=job_id,
                        stage_name=stage_id,
                        start=start,
                        end=end,
                    )
                )
                if op is None:
                    raise RuntimeError(
                        f"Failed to schedule operation job={job_id}, stage={stage_id}"
                    )

                stage_2_end[stage_id] = end
                prev_end = end

        return schedule

    # End logging helper methods


def _compute_simple_horizon(instance: FlowshopDuedateParameters) -> int:
    """Return a safe (loose) horizon if none is provided.

    Uses total processing time sum_{i,j} p_{i,j}.
    """
    pmap = instance.get_job_2_stage_2_builtin_int_p_map()
    total = 0
    for j in instance.job_id_list:
        for i in instance.stage_id_list:
            total += int(pmap[j][i])
    return int(total)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Solve one FlowshopDuedateParameters instance using CPLEX/docplex MILP and print the result."
    )
    parser.add_argument(
        "instance_path",
        type=str,
        help="Path to a VRM benchmark instance file (e.g. Inputs/.../123.txt).",
    )
    parser.add_argument(
        "--timelimit",
        type=float,
        default=30.0,
        help="Time limit in seconds for MILP solve.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (used for any randomized subroutines / reproducibility).",
    )
    parser.add_argument(
        "--pop_size",
        type=int,
        default=150,
        help="Population size for the genetic algorithm.",
    )
    parser.add_argument(
        "--cross_size",
        type=int,
        default=200,
        help="Number of crossovers per generation for the genetic algorithm.",
    )
    parser.add_argument(
        "--mut_size",
        type=int,
        default=100,
        help="Number of mutations per generation for the genetic algorithm.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    instance_path = Path(args.instance_path)
    if not instance_path.exists():
        raise FileNotFoundError(f"Instance file not found: {instance_path}")

    ins_name = instance_path.stem
    with instance_path.open("r", encoding="utf-8") as f:
        instance = FlowshopDuedateParameters.from_vrm_data(ins_name, f)

    horizon = _compute_simple_horizon(instance)
    shared_param_dict = {"horizon": horizon}

    stopping_criteria = StoppingCriteria({"timelimit": float(args.timelimit)})
    ctrl = FlowshopTardinessGeneticAlgorithmController(
        instance=instance,
        shared_param_dict=shared_param_dict,
        subroutine_flow=DynamicDataObject.from_obj([]),
        stopping_criteria=stopping_criteria,
    )

    if args.seed:
        ctrl.set_random_seed(int(args.seed))

    ctrl.ga_edd(
        pop_size=int(args.pop_size),
        cross_size=int(args.cross_size),
        mut_size=int(args.mut_size),
    )

    best = ctrl.solution_manager.get_incumbent()
    if best is None:
        print("No solution found.")
        return 2

    best_obj = ctrl.population_manager.get_best_fitness()
    perm = ctrl.population_manager.get_best_solution()

    print("==== GA-EDD Result ====")
    print(f"instance: {instance.name}")
    print(f"jobs: {instance.job_count}, stages: {instance.stage_count}")
    print(
        f"timelimit_s: {args.timelimit}, pop_size: {args.pop_size}"
        f", cross_size: {args.cross_size}, mut_size: {args.mut_size}"
    )
    print(f"horizon: {horizon}")
    print(f"objective (sumTj): {best_obj}")
    print(f"perm(first stage): {perm}")
    print("=======================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
