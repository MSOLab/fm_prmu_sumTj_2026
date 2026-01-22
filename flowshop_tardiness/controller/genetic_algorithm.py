import argparse
import logging
import random
from pathlib import Path
from typing import Any, Sequence

from routix import DynamicDataObject, ElapsedTimer, StoppingCriteria
from schore.parameters_examples.shop.flow import FlowshopDuedateParameters
from schore.schedule_examples.shop.flow import FlowshopOperation, FlowshopSchedule

from flowshop_tardiness.controller.flowshop_batch_eval import (
    PermutationFlowshopSubseqEvaluator,
)
from flowshop_tardiness.fm_prmu import PermutationFlowshopScheduleLite
from flowshop_tardiness.report import FsSubroutineReport

from ..genetic_algorithm_model import PopulationManager
from .base_flowshop_controller import BaseFlowshopController


class FlowshopTardinessGeneticAlgorithmController(BaseFlowshopController):
    pop_mgr: PopulationManager
    """Population manager for genetic algorithm."""

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

    # Start subalgorithm definition: GA_EDD by Ta et al. (2018)

    def ga_edd(self, pop_size: int, cross_size: int, mut_size: int):
        """Genetic Algorithm with EDD initialization for Flowshop Tardiness by Ta et al. (2018).

        Args:
            pop_size (int): Population size
            cross_size (int): Number of crossovers per generation
            mut_size (int): Number of mutations per generation

        Raises:
            RuntimeError: If no trajectory record found after initialization.
            RuntimeError: If population is empty during main loop.
            RuntimeError: If no trajectory record found after update.
            RuntimeError: If no best solution found in population manager.
        """
        sub_timer = ElapsedTimer()
        _cross_cnt = int(cross_size / 2)  # two children per crossover
        # Initialize population manager
        self.pop_mgr = PopulationManager(pop_size, timer=self.timer)
        # job_set = set(self.instance.job_id_list)  # for sanity check

        # Initialize population
        self._initialize_population()
        # self._log_best_fitness_to_obj_store("1st")
        self.pop_mgr.generation = 1
        record = self.pop_mgr.get_last_trajectory_record()
        if record is None:
            raise RuntimeError("No trajectory record found after initialization.")
        logging.info(
            f"Gen {record.generation} at {record.timestamp:.2f}: "
            f"Best obj (sumTj) = {record.obj_value} by {record.source}"
        )
        gen_best_obj_value = record.obj_value

        is_timeover: bool = self.time_is_up()
        is_optimal: bool = gen_best_obj_value == 0
        # Start main loop
        while not (is_timeover or is_optimal):
            self.pop_mgr.generation += 1
            P_prev = self.pop_mgr.get_this_population_list()
            if not P_prev:
                raise RuntimeError("Population is empty, cannot proceed with GA.")

            # Crossover
            if len(P_prev) >= 2:
                for _ in range(_cross_cnt):
                    is_timeover = self.time_is_up()
                    if is_timeover:
                        break
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

                    self.pop_mgr.add_individual(
                        child_sol1, self._evaluate(child_sol1), source=cross_type
                    )
                    self.pop_mgr.add_individual(
                        child_sol2, self._evaluate(child_sol2), source=cross_type
                    )

                this_obj_value = self.pop_mgr.get_best_fitness()
                if is_timeover:
                    # updated = self._log_best_fitness_to_obj_store("TIME UP")
                    if gen_best_obj_value != this_obj_value:
                        record = self.pop_mgr.get_last_trajectory_record()
                        if record is None:
                            raise RuntimeError(
                                "No trajectory record found after update."
                            )
                        logging.info(
                            f"Gen {record.generation} at {record.timestamp:.2f}: "
                            f"Best obj (sumTj) = {record.obj_value} by {record.source}"
                        )
                        gen_best_obj_value = this_obj_value
                    break
                is_optimal = this_obj_value == 0
                if is_optimal:
                    gen_best_obj_value = this_obj_value
                    logging.info("GA-EDD: reached optimal solution (obj=0).")
                    break

            # Mutation
            if P_prev:
                for _ in range(mut_size):
                    is_timeover = self.time_is_up()
                    if is_timeover:
                        break
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

                    self.pop_mgr.add_individual(
                        child_sol, self._evaluate(child_sol), source=mutation_type
                    )

                this_obj_value = self.pop_mgr.get_best_fitness()
                if is_timeover:
                    # updated = self._log_best_fitness_to_obj_store("TIME UP")
                    if gen_best_obj_value != this_obj_value:
                        record = self.pop_mgr.get_last_trajectory_record()
                        if record is None:
                            raise RuntimeError(
                                "No trajectory record found after update."
                            )
                        logging.info(
                            f"Gen {record.generation} at {record.timestamp:.2f}: "
                            f"Best obj (sumTj) = {record.obj_value} by {record.source}"
                        )
                        gen_best_obj_value = this_obj_value
                    break
                is_optimal = this_obj_value == 0
                if is_optimal:
                    gen_best_obj_value = this_obj_value
                    logging.info("GA-EDD: reached optimal solution (obj=0).")
                    break

            # Elitist replacement
            self.pop_mgr.elitist_replace()

            # Log best fitness if improved
            this_obj_value = self.pop_mgr.get_best_fitness()
            if gen_best_obj_value != this_obj_value:
                record = self.pop_mgr.get_last_trajectory_record()
                if record is None:
                    raise RuntimeError("No trajectory record found after update.")
                logging.info(
                    f"Gen {record.generation} at {record.timestamp:.2f}: "
                    f"Best obj (sumTj) = {record.obj_value} by {record.source}"
                )
                gen_best_obj_value = this_obj_value

            # Check time limit for next generation
            is_timeover = self.time_is_up()
            is_optimal = gen_best_obj_value == 0
            if is_optimal:
                logging.info("GA-EDD: reached optimal solution (obj=0).")

        # End of main loop
        last_obj_value = self.pop_mgr.cumulative_best_fitness
        last_job_seq = self.pop_mgr.cumulative_best_sol

        # Wrap up
        if last_obj_value is not None:
            if last_job_seq is None:
                raise RuntimeError("No best solution found in population manager.")
            # Update objective store
            timestamp_obj_value_list = self.pop_mgr.get_best_obj_series()
            for timestamp, obj_value in timestamp_obj_value_list:
                self.obj_store.add_obj_value(
                    timestamp=timestamp,
                    value=obj_value,
                    is_maximize=False,
                )
            log_time = self.timer.elapsed_sec
            self.obj_store.add_obj_value(log_time, last_obj_value, is_maximize=None)
            _last_timestamp_note = self._get_call_context_of_current_method()
            self.obj_store.add_last_timestamp_note(
                _last_timestamp_note, obj_value_is_valid=True
            )

            # Register final solution
            report = FsSubroutineReport(
                elapsed_time=sub_timer.elapsed_sec,
                obj_value=last_obj_value,
                obj_bound=None,
                is_init=False,
            )
            last_schedule = self._dispatch_permutation(list(last_job_seq))
            self.solution_manager.register(report, last_schedule)

    # End subalgorithm definition: GA_EDD by Ta et al. (2018)

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

    def _initialize_population(self, add_neh_edd_sol: bool = False) -> None:
        """Initialize population with EDD, optional NEHedd, and random permutations.

        Args:
            add_neh_edd_sol (bool, optional): Whether to add NEHedd solution.
                Defaults to False.
        """
        mgr = self.pop_mgr

        # Add EDD solution
        if not hasattr(self, "edd_solution"):
            self.edd_solution: tuple[str, ...] = self._build_edd_solution()
        if not hasattr(self, "edd_fitness"):
            self.edd_fitness: int = self._evaluate(self.edd_solution)
        self.pop_mgr.add_individual(
            self.edd_solution, self.edd_fitness, source="EDD_INIT"
        )
        random_sol_cnt = mgr.pop_size - 1

        # (Optional) Add NEHedd solution
        if add_neh_edd_sol:
            if not hasattr(self, "neh_edd_solution"):
                self.neh_edd_solution: tuple[str, ...] = self._build_neh_edd_solution()
            if not hasattr(self, "neh_edd_fitness"):
                self.neh_edd_fitness: int = self._evaluate(self.neh_edd_solution)
            self.pop_mgr.add_individual(
                self.neh_edd_solution, self.neh_edd_fitness, source="NEHedd_INIT"
            )
            random_sol_cnt -= 1

        # Remaining solutions by random permutations
        random_solutions = self._get_random_solutions(random_sol_cnt)
        for sol in random_solutions:
            fitness = self._evaluate(sol)
            self.pop_mgr.add_individual(sol, fitness, source="RAND_INIT")

    @staticmethod
    def _crossover_one_point(
        parent_a: tuple[str, ...], parent_b: tuple[str, ...]
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

    @staticmethod
    def _crossover_linear_order(
        parent_a: tuple[str, ...], parent_b: tuple[str, ...]
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
            return FlowshopTardinessGeneticAlgorithmController._crossover_one_point(
                parent_a, parent_b
            )

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

        for pos, job in zip(outside_positions, fill_jobs2):
            child2[pos] = job

        o2: tuple[str, ...] = tuple(child2)  # type: ignore[arg-type]

        return o1, o2

    @staticmethod
    def _mutate_forward_insertion(solution: tuple[str, ...]) -> tuple[str, ...]:
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

    @staticmethod
    def _mutate_backward_insertion(solution: tuple[str, ...]) -> tuple[str, ...]:
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
        obj_value = self.pop_mgr.get_best_fitness()
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

    # Start subalgorithm definition: GAPR by Vallada and Ruiz (2010)

    def gapr(
        self,
        P_size: int | None = None,
        div: float | None = None,
        pressure: float | None = None,
        P_m: float | None = None,
        P_ls: float | None = None,
    ):
        """Genetic Algorithm with Path Relinking for Flowshop Tardiness by Vallada and Ruiz (2010).

        Args:
            P_size (int | None, optional): Population size.
                If None, defaults to 30. Defaults to None.
            Div (float | None, optional): Diversity threshold.
                If None, defaults to 0.4. Defaults to None.
            Pressure (float | None, optional): Pressure for the selection.
                If None, defaults to 0.3 (30% in the paper). Defaults to None.
            P_m (float | None, optional): Mutation probability.
                If None, defaults to 0.02. Defaults to None.
            P_ls (float | None, optional): Local search probability.
                If None, defaults to 0.15. Defaults to None.
        """
        # --- defaults ---
        if P_size is None:
            P_size = 30
        if div is None:
            div = 0.4
        if pressure is None:
            pressure = 0.3
        if P_m is None:
            P_m = 0.02
        if P_ls is None:
            P_ls = 0.15

        _pressure = self._normalize_pressure(pressure)

        sub_timer = ElapsedTimer()
        logging.info(
            "Starting GAPR method with parameters: "
            f"P_size={P_size}, Div={div}, Pressure={_pressure}, P_m={P_m}, P_ls={P_ls}"
        )
        # Statistics
        stats: dict[str, Any] = {
            "gen_start": 1,
            "gen_end": 1,
            "restart_cnt": 0,
            "div_check_cnt": 0,
            "div_below_cnt": 0,
            "pr_children_generated": 0,  # Children count generated by PR (usually 2 each)
            "children_added_cnt": 0,  # Children count accepted to population
            "children_dup_rejected_cnt": 0,
            "ls_called_cnt": 0,
            "ls_improved_cnt": 0,
            "pr_called_cnt": 0,
            "pr_candidates_generated": 0,
            "pr_added_cnt": 0,
            "pr_dup_rejected_cnt": 0,
            "elitist_replace_cnt": 0,
            "best_update_cnt": 0,
            "best_by_source": {},  # Best update counts by sources (NEHedd_INIT/PR/...)
        }

        # Initialize population manager
        self.pop_mgr = PopulationManager(P_size, timer=self.timer)

        # Build initial population
        self._initialize_population(add_neh_edd_sol=True)

        self.pop_mgr.generation = 1
        record = self.pop_mgr.get_last_trajectory_record()
        if record is None:
            raise RuntimeError("No trajectory record found after initialization.")
        logging.info(
            f"GAPR Gen {record.generation} at {record.timestamp:.2f}: "
            f"Best obj (sumTj) = {record.obj_value} by {record.source}"
        )
        this_gen_obj_value = record.obj_value

        # Track duplicates using current population content
        pop_set: set[tuple[str, ...]] = self.pop_mgr.get_population_set()

        # PR 'marked' control (to reduce repeated PR pairs)
        marked: set[tuple[str, ...]] = set()

        is_timeover: bool = self.time_is_up()
        is_optimal: bool = this_gen_obj_value == 0

        # ---------------- main loop (steady-state style) ----------------
        while not (is_timeover or is_optimal):
            self.pop_mgr.generation += 1
            stats["gen_end"] += 1
            P_prev = self.pop_mgr.get_this_population_list()
            if not P_prev:
                raise RuntimeError("Population is empty, cannot proceed with GAPR.")

            # Refresh population set (in case elitist_replace changed membership)
            pop_set = set(P_prev)

            # 1) Diversity check & restart if needed
            stats["div_check_cnt"] += 1
            div_value = self._compute_population_diversity(P_prev)
            if div_value < div:
                stats["div_below_cnt"] += 1
                stats["restart_cnt"] += 1
                self._restart_population_type1()
                marked.clear()
                P_prev = self.pop_mgr.get_this_population_list()
                pop_set = set(P_prev)

            # 2) Parent selection (tournament with pressure)
            # ensure distinct parents when possible
            if len(P_prev) >= 2:
                unmarked_cnt = sum(1 for s in P_prev if s not in marked)
                if unmarked_cnt < 2:
                    marked.clear()
                parent1, parent2 = self._select_pr_pair_pr1(P_prev, _pressure, marked)
            else:
                parent1 = P_prev[0]
                parent2 = P_prev[0]

            # 3) Offspring by path relinking
            if len(P_prev) >= 2:
                children = self._path_relink_bidirectional_best(parent1, parent2)
                marked.add(parent1)
                marked.add(parent2)
                child_source = "PR"
                stats["pr_called_cnt"] += 1
                stats["pr_children_generated"] += len(children)
            else:
                children = [parent1]
                child_source = "PR_DEGEN"

            # 4) Mutation (insertion-based, probability P_m)
            mutated_children: list[tuple[str, ...]] = []
            for ch in children:
                mutated_children.append(self._mutate_insertion_per_gene(ch, P_m))

            # 5) Local search with probability P_ls (light insertion LS)
            improved_children: list[tuple[str, ...]] = []
            for ch in mutated_children:
                before = ch
                after = ch
                if P_ls > 0.0 and random.random() < P_ls:
                    stats["ls_called_cnt"] += 1
                    after = self._search_insertion_neighborhood(ch)
                    if after != before:
                        stats["ls_improved_cnt"] += 1
                improved_children.append(after)

            # 6) Acceptance: accept if (unique) AND (better than worst)
            accepted_any = False

            # Worst fitness in current population
            worst_fit = self.pop_mgr.get_worst_fitness()
            if worst_fit is None:
                raise RuntimeError("Cannot determine worst fitness in population.")

            for ch in improved_children:
                is_timeover = self.time_is_up()
                if is_timeover:
                    break

                if ch in pop_set:
                    stats["children_dup_rejected_cnt"] += 1
                    continue

                fit = self._evaluate(ch)

                # Reject if not better than worst
                if fit >= worst_fit:
                    continue

                # Accept
                self.pop_mgr.add_individual(ch, fit, source=child_source)
                pop_set.add(ch)
                accepted_any = True
                stats["children_added_cnt"] += 1

            # Maintain population size
            if accepted_any:
                self.pop_mgr.elitist_replace()
                stats["elitist_replace_cnt"] += 1

            # 7) Log best fitness if improved
            this_obj_value = self.pop_mgr.get_best_fitness()
            if this_obj_value is None:
                raise RuntimeError("No best fitness found in population manager.")
            if this_obj_value != this_gen_obj_value:
                record = self.pop_mgr.get_last_trajectory_record()
                if record is None:
                    raise RuntimeError("No trajectory record found after update.")
                logging.info(
                    f"GAPR Gen {record.generation} at {record.timestamp:.2f}: "
                    f"Best obj (sumTj) = {record.obj_value} by {record.source}"
                )
                this_gen_obj_value = this_obj_value
                stats["best_update_cnt"] += 1
                src = record.source if record else "UNKNOWN"
                stats["best_by_source"][src] = stats["best_by_source"].get(src, 0) + 1

            # termination checks
            is_timeover = self.time_is_up()
            is_optimal = this_gen_obj_value == 0
            if is_optimal:
                logging.info("GAPR: reached optimal solution (obj=0).")

        # End of main loop
        last_obj_value = self.pop_mgr.cumulative_best_fitness
        last_job_seq = self.pop_mgr.cumulative_best_sol

        # Gather statistics
        elapsed = sub_timer.elapsed_sec
        pop_size_final = len(self.pop_mgr.get_this_population_list())

        logging.info("===== GAPR Summary =====")
        logging.info(
            "gens: %d -> %d (total %d), elapsed: %.2fs, best_obj: %s, pop_size_final: %d",
            stats["gen_start"],
            stats["gen_end"],
            stats["gen_end"] - stats["gen_start"] + 1,
            elapsed,
            str(last_obj_value),
            pop_size_final,
        )
        logging.info(
            "restart: %d, div_checks: %d, div_below: %d",
            stats["restart_cnt"],
            stats["div_check_cnt"],
            stats["div_below_cnt"],
        )
        logging.info(
            "OP children generated: %d, children added: %d, dup rejected: %d, elitist_replace calls: %d",
            stats["pr_children_generated"],
            stats["children_added_cnt"],
            stats["children_dup_rejected_cnt"],
            stats["elitist_replace_cnt"],
        )
        logging.info(
            "LS called: %d, LS improved: %d (P_ls=%.3f)",
            stats["ls_called_cnt"],
            stats["ls_improved_cnt"],
            P_ls,
        )
        logging.info(
            "PR called: %d, PR candidates: %d, PR added: %d, PR dup rejected: %d",
            stats["pr_called_cnt"],
            stats["pr_candidates_generated"],
            stats["pr_added_cnt"],
            stats["pr_dup_rejected_cnt"],
        )
        logging.info(
            "Best updates: %d, by source: %s",
            stats["best_update_cnt"],
            stats["best_by_source"],
        )
        logging.info("========================")

        # Wrap up
        if last_obj_value is not None:
            if last_job_seq is None:
                raise RuntimeError("No best solution found in population manager.")
            # Update objective store
            timestamp_obj_value_list = self.pop_mgr.get_best_obj_series()
            for timestamp, obj_value in timestamp_obj_value_list:
                self.obj_store.add_obj_value(
                    timestamp=timestamp,
                    value=obj_value,
                    is_maximize=False,
                )
            log_time = self.timer.elapsed_sec
            self.obj_store.add_obj_value(log_time, last_obj_value, is_maximize=None)
            _last_timestamp_note = self._get_call_context_of_current_method()
            self.obj_store.add_last_timestamp_note(
                _last_timestamp_note, obj_value_is_valid=True
            )

            # Register final solution
            report = FsSubroutineReport(
                elapsed_time=sub_timer.elapsed_sec,
                obj_value=last_obj_value,
                obj_bound=None,
                is_init=False,
            )
            last_schedule = self._dispatch_permutation(list(last_job_seq))
            self.solution_manager.register(report, last_schedule)

    # End subalgorithm definition: GAPR by Vallada and Ruiz (2010)

    # Start NEHedd helper (not used in GA-EDD)

    def _build_neh_edd_solution(self) -> tuple[str, ...]:
        # 1) EDD order
        edd_sequence = self._build_edd_solution()

        return_seq: list[str] = [edd_sequence[0]]
        # 2) NEH insertion by EDD order with NEW acceleration evaluation
        for j in edd_sequence[1:]:
            pos_list = self._get_best_pos_list_and_metric_new_acc(return_seq, j)
            pos = pos_list[0]
            return_seq.insert(pos, j)

        return tuple(return_seq)

    def _get_best_pos_list_and_metric_new_acc(
        self,
        seq_now: list[str],
        job_id_seq: Sequence[str] | str,
    ) -> list[int]:
        """
        Evaluate insertion of job_id into seq_now using NEW acceleration
        (Fernandez-Viagas et al., 2020) evaluator.

        Args:
            seq_now (list[str]): current sequence of job IDs
            job_id_seq (Sequence[str] | str): job ID (string) or sequence of job IDs to insert

        Returns:
            tuple[list[int], int]: list of best insertion positions & objective value
        """
        _job_id_seq: Sequence[str]
        if isinstance(job_id_seq, str):
            _job_id_seq = [job_id_seq]
        else:
            _job_id_seq = job_id_seq

        evaluator, job_id_to_idx, _ = self._get_new_acc_evaluator()

        # convert current sequence to index sequence
        pi_idx = [job_id_to_idx[j] for j in seq_now]
        sigma_idx_seq = [job_id_to_idx[job_id] for job_id in _job_id_seq]

        # NEW evaluator returns best position and best objective1 value (sumTj)
        best_pos_list, _ = evaluator.get_best_position(pi_idx, sigma_idx_seq)
        return best_pos_list

    def _get_new_acc_evaluator(self):
        """
        Build (and cache) a PermutationFlowshopSubseqEvaluator that uses 0-based
        integer job indices internally.

        Returns:
            (evaluator, job_id_to_idx, idx_to_job_id)
        """
        # cache on controller instance to avoid rebuilding on every call
        if hasattr(self, "_new_acc_cache") and self._new_acc_cache is not None:
            return self._new_acc_cache

        job_ids: list[str] = list(self.instance.job_id_list)
        stage_ids: list[str] = list(self.stage_ids)

        job_id_to_idx: dict[str, int] = {jid: k for k, jid in enumerate(job_ids)}
        idx_to_job_id: dict[int, str] = {k: jid for jid, k in job_id_to_idx.items()}

        m = len(stage_ids)
        n = len(job_ids)

        # Build p[m][n] aligned with (stage_idx, job_idx)
        p = [[0] * n for _ in range(m)]
        for i, stage_id in enumerate(stage_ids):
            for jid in job_ids:
                j = job_id_to_idx[jid]
                p[i][j] = int(self.stage_2_job_2_p_dict[stage_id][jid])

        # Build due[n]
        due = [0] * n
        for jid in job_ids:
            j = job_id_to_idx[jid]
            due[j] = int(self.instance.job_2_duedate_map[jid])

        evaluator = PermutationFlowshopSubseqEvaluator(p, due)

        self._new_acc_cache: tuple[
            PermutationFlowshopSubseqEvaluator, dict[str, int], dict[int, str]
        ] = (evaluator, job_id_to_idx, idx_to_job_id)
        return self._new_acc_cache

    # End NEHedd helper (not used in GA-EDD)

    # Start GAPR helper methods

    @staticmethod
    def _normalize_pressure(pressure: float) -> float:
        """Normalize pressure into [0, 1].

        Args:
            pressure (float): Input pressure value, which can be a fraction (e.g., 0.3) or a percentage (e.g., 30).

        Returns:
            float:
              - If user passes 0.3, keep 0.3 (30%).
              - If user passes 30, interpret as 30% => 0.3.
        """
        if pressure > 1.0:
            return pressure / 100.0
        return max(0.0, min(1.0, pressure))

    def _tournament_select(
        self, population: list[tuple[str, ...]], pressure: float
    ) -> tuple[str, ...]:
        """n-tournament selection with pressure fraction.

        Args:
            population (list[tuple[str, ...]]): Current population of solutions.
            pressure (float): Normalized pressure for the selection.

        Raises:
            ValueError: If the population is empty.
            RuntimeError: If tournament selection fails.

        Returns:
            tuple[str, ...]: Selected solution.
        """
        if not population:
            raise ValueError("Population is empty.")
        k = max(2, int((len(population) * pressure) + 0.999999))  # ceil
        k = min(k, len(population))
        sample = (
            random.sample(population, k=k) if k < len(population) else list(population)
        )

        # choose best (min objective) among sample
        best = None
        best_fit = None
        for sol in sample:
            fit = self._evaluate(sol)
            if best_fit is None or fit < best_fit:
                best_fit = fit
                best = sol
        if best is None:
            raise RuntimeError("Tournament selection failed.")
        return best

    @staticmethod
    def _mutate_insertion_per_gene(
        solution: tuple[str, ...], p_m: float
    ) -> tuple[str, ...]:
        """Insertion mutation applied per gene with probability p_m.

        For each position i, with prob p_m:
        - remove job at i (current list index)
        - insert at a random position
        """
        if p_m <= 0.0:
            return solution
        sol = list(solution)
        n = len(sol)
        if n <= 1:
            return solution

        # apply sequentially (note indices shift)
        i = 0
        while i < len(sol):
            if random.random() < p_m:
                job = sol.pop(i)
                j = random.randrange(0, len(sol) + 1)
                sol.insert(j, job)
                # do not increment i aggressively; keep moving forward
            i += 1
        return tuple(sol)

    def _search_insertion_neighborhood(
        self, solution: tuple[str, ...]
    ) -> tuple[str, ...]:
        base: list[str] = list(solution)
        n: int = len(base)
        if n <= 2:
            return solution

        # Get NEW evaluator + id<->idx mapping (cached)
        evaluator, job_id_to_idx, idx_to_job_id = self._get_new_acc_evaluator()

        # Current solution in idx form
        pi_idx_full: list[int] = [job_id_to_idx[j] for j in solution]

        # Current objective via evaluator (fast enough to compute once using normal eval if you prefer)
        # We'll keep self._evaluate here once; it's OK and consistent with rest of code.
        base_obj: int = self._evaluate(solution)
        best_sol: tuple[str, ...] | None = None
        best_obj: int = base_obj

        for pos in list(range(n)):
            job_idx: int = pi_idx_full[pos]

            # Build pi without this job (still idx-based)
            pi_wo: list[int] = pi_idx_full[:pos] + pi_idx_full[pos + 1 :]

            # Ask evaluator: where to insert job_idx (sigma length 1) to minimize sumTj
            best_pos_list, best_obj_val = evaluator.get_best_position(pi_wo, [job_idx])

            # If no improvement, skip
            if best_obj_val >= best_obj:
                continue

            # Construct improved permutation (job inserted at earliest best position)
            insert_pos = best_pos_list[0]

            new_pi = pi_wo[:insert_pos] + [job_idx] + pi_wo[insert_pos:]
            new_sol = tuple(idx_to_job_id[k] for k in new_pi)

            # best-improvement mode: keep best found in this pass
            best_obj = best_obj_val
            best_sol = new_sol

        return best_sol if best_sol is not None else solution

    def _compute_population_diversity(self, population: list[tuple[str, ...]]) -> float:
        """
        Paper-style diversity (Wineberg & Oppacher):
        Div = (1/(n-1)) * sum_{k=1..n} sum_{l=1..n} f_k(l) * (1 - f_k(l))
        where f_k(l) is frequency of job l at position k in the population.
        """
        m = len(population)
        if m <= 1:
            return 0.0
        n = len(population[0])
        if n <= 1:
            return 0.0

        # jobs universe: from first permutation (all perms contain same jobs)
        jobs = list(population[0])

        total = 0.0
        for pos in range(n):
            # freq at this position
            freq: dict[str, int] = {j: 0 for j in jobs}
            for sol in population:
                freq[sol[pos]] += 1

            # sum_l f_k(l) * (1 - f_k(l))
            for j in jobs:
                f = freq[j] / m
                total += f * (1.0 - f)

        div = total / (n - 1)
        # safe clamp
        return max(0.0, min(1.0, div))

    def _restart_population_type1(self) -> None:
        """
        Restart type 1 (paper): keep EDD and NEHedd, fill the rest randomly.
        """
        # reset population manager
        self.pop_mgr.clear()
        self._initialize_population(add_neh_edd_sol=True)

    def _select_pr_pair_pr1(
        self,
        population: list[tuple[str, ...]],
        pressure: float,
        marked: set[tuple[str, ...]],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """PR type 1: pick two different unmarked individuals using tournament selection."""
        if len(population) < 2:
            raise ValueError("Need at least 2 individuals for PR.")

        # filter unmarked if possible
        unmarked = population
        cand_pool = unmarked if len(unmarked) >= 2 else population

        a = self._tournament_select(cand_pool, pressure)
        b = self._tournament_select(cand_pool, pressure)
        _guard = 0
        while b == a and _guard < 20:
            b = self._tournament_select(cand_pool, pressure)
            _guard += 1
        return a, b

    def _path_relink_bidirectional_best(
        self, a: tuple[str, ...], b: tuple[str, ...]
    ) -> list[tuple[str, ...]]:
        """Generate best candidates from bidirectional path relinking (a->b and b->a)."""
        best1 = self._path_relink_best(a, b)
        best2 = self._path_relink_best(b, a)
        out: list[tuple[str, ...]] = []
        if best1 is not None:
            out.append(best1)
        if best2 is not None and best2 != best1:
            out.append(best2)
        return out

    def _path_relink_best(
        self, initiating: tuple[str, ...], guiding: tuple[str, ...]
    ) -> tuple[str, ...] | None:
        """Simple PR for permutations via position-fixing swaps.

        Starting from initiating, iteratively move toward guiding by fixing
        the leftmost position where they differ (swap in the correct job).
        Evaluate intermediates and return the best encountered (excluding start if no improvement).
        """
        if len(initiating) != len(guiding):
            raise ValueError("PR requires same length permutations.")
        n: int = len(initiating)
        if n <= 1:
            return None

        cur: list[str] = list(initiating)
        best_sol: tuple[str, ...] = tuple(cur)
        best_fit: int = self._evaluate(best_sol)

        # walk until matches guiding
        # (left-to-right fixing tends to be stable and cheap)
        pos_in_cur = {job: idx for idx, job in enumerate(cur)}
        for i in range(n):
            if cur[i] == guiding[i]:
                continue
            desired_job: str = guiding[i]
            j: int = pos_in_cur[desired_job]
            # swap positions i and j
            cur[i], cur[j] = cur[j], cur[i]
            pos_in_cur[cur[j]] = j
            pos_in_cur[cur[i]] = i

            sol = tuple(cur)
            fit = self._evaluate(sol)
            if fit < best_fit:
                best_fit = fit
                best_sol = sol

        # return best encountered (could be initiating itself)
        if best_sol == initiating:
            return None
        return best_sol

    # End GAPR helper methods


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

    # result_header_str = "==== GA-EDD Result ===="
    # ctrl.ga_edd(
    #     pop_size=int(args.pop_size),
    #     cross_size=int(args.cross_size),
    #     mut_size=int(args.mut_size),
    # )

    result_header_str = "===== GAPR Result ====="
    ctrl.gapr()

    best = ctrl.solution_manager.get_incumbent()
    if best is None:
        print("No solution found.")
        return 2

    best_obj = ctrl.pop_mgr.get_best_fitness()
    perm = ctrl.pop_mgr.get_best_solution()

    print(result_header_str)
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
