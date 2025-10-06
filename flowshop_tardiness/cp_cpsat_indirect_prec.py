from __future__ import annotations

import logging
import math
from itertools import permutations

from mbls.cpsat.cp_model_with_fixed_interval import CpModelWithFixedInterval
from ortools.sat.python.cp_model import IntVar
from routix import ElapsedTimer
from routix.util.comparison import float_equals
from schore.parameters_examples.shop.flow import FlowshopDuedateParameters

from .scheduling.flowshop_schedule import FlowshopOperation, FlowshopSchedule


class CpCpsatIndirectPrec(CpModelWithFixedInterval):
    """
    Implementation of the CP model for the flowshop problem to minimize total tardiness
    using indirect precedence constraints only.
    """

    # Indices & Parameters

    j_list: list[str]
    """$J$: job index (j) list"""

    i_list: list[str]
    """$I$: stage index (i) list"""

    p: dict[tuple[str, str], int]
    """$P_{ji}$: processing time of job j at stage i"""

    stage_start_time_lb: dict[str, int]
    """i -> lower bound on the start time of the stage."""

    stage_end_time_ub: dict[str, int]
    """i -> upper bound on the makespan of the stage."""

    D: dict[str, int]
    """$D_j$: due date of job j"""

    # Variables

    prec: dict[tuple[str, str], IntVar]
    """$prec_{j1,j2}$: Indirect precedence variables for the circuit constraints"""

    var_T: dict[str, IntVar]
    """$T_j$: tardiness of job j"""

    # Objective

    obj_var: IntVar
    """Defines the objective for the scheduling problem."""

    def __init__(self, horizon: int) -> None:
        super().__init__(horizon)

    @classmethod
    def from_instance(
        cls, instance: FlowshopDuedateParameters, horizon: int
    ) -> CpCpsatIndirectPrec:
        result = cls(horizon)
        result.name = f"{cls.__name__}_{instance.name}"
        result.define_model(instance)
        return result

    def define_model(self, instance: FlowshopDuedateParameters) -> None:
        elapsed = ElapsedTimer()
        self.define_parameters(instance)
        logging.info(f"Defined parameters; took {elapsed.elapsed_sec:.3f} sec.")

        elapsed.set_start_time_as_now()
        self.define_variables()
        logging.info(f"Defined variables; took {elapsed.elapsed_sec:.3f} sec.")

        elapsed.set_start_time_as_now()
        self.define_total_tardiness_objective()
        logging.info(f"Defined objective; took {elapsed.elapsed_sec:.3f} sec.")

        elapsed.set_start_time_as_now()
        self.define_constraints()
        logging.info(f"Defined constraints; took {elapsed.elapsed_sec:.3f} sec.")

    # Parameters

    def define_parameters(self, instance: FlowshopDuedateParameters) -> None:
        """
        Define the parameters for the model based on the FlowshopDuedateParameters instance.

        Args:
            instance (FlowshopDuedateParameters): The flow shop problem instance.
        """
        self.j_list = instance.job_id_list
        self.i_list = instance.stage_id_list
        n = len(self.j_list)

        _p = instance.p_manager.job_stage_2_value_map(self.j_list, self.i_list)
        self.p = {
            (j, i): int(round(_p[j, i])) for j in self.j_list for i in self.i_list
        }

        self.stage_start_time_lb = {}
        cumulative_p_min = 0
        for i in self.i_list:
            self.stage_start_time_lb[i] = cumulative_p_min
            cumulative_p_min += min(self.p[j, i] for j in self.j_list)

        self.stage_end_time_ub = {}
        for i_idx, i in enumerate(self.i_list):
            candid1 = 0
            p_max = 0
            for ip in self.i_list[: i_idx + 1]:
                candid1 += sum(self.p[j, ip] for j in self.j_list)
                p_max = max(p_max, max(self.p[j, ip] for j in self.j_list))
            # max processing time * (job count + stage count - 1)
            candid2 = p_max * (n + i_idx)
            self.stage_end_time_ub[i] = min(candid1, candid2)
            logging.info(f"  end_time_ub[{i}] := {self.stage_end_time_ub[i]}")

        self.D = {j: int(round(instance.job_2_duedate_map[j])) for j in self.j_list}

    # Variables

    def define_variables(self) -> None:
        j_list = self.j_list

        # Interval variables
        for j in j_list:
            for i in self.i_list:
                start_time_lb = self.stage_start_time_lb[i]
                start_time_ub = self.stage_end_time_ub[i] - self.p[j, i]
                self.define_fixed_interval_var(
                    (j, i), self.p[j, i], lb=start_time_lb, ub=start_time_ub
                )

        # Indirect precedence
        self.prec = {}
        for j1_idx, j1 in enumerate(j_list):
            for j2 in j_list[j1_idx + 1 :]:
                self.prec[j1, j2] = self.new_bool_var(f"prec_ind_{j1}_{j2}")
                self.prec[j2, j1] = self.new_bool_var(f"prec_ind_{j2}_{j1}")
                self.add(self.prec[j1, j2] + self.prec[j2, j1] == 1)

    # Objective

    def define_total_tardiness_objective(self) -> None:
        """
        Total tardiness objective: minimize \\sum_j{T_j} where T_j := max(end_j - D_j, 0).

        Uses `add_max_equality` for clarity.
        """
        j_list = self.j_list
        last_i = self.i_list[-1]

        self.var_T = {}
        total_ub = 0
        for j in j_list:
            ub = max(self.stage_end_time_ub[last_i] - self.D[j], 0)
            self.var_T[j] = self.new_int_var(0, ub, f"T_{j}")
            self.add_max_equality(
                self.var_T[j],
                [self.var_op_start[j, last_i] + self.p[j, last_i] - self.D[j], 0],
            )
            total_ub += ub

        self.obj_var = self.new_int_var(0, total_ub, "sum_Tj")
        self.add(self.obj_var == sum(self.var_T[j] for j in j_list))

        self.minimize(self.obj_var)

    def set_obj_lower_bound(self, bound: float | None) -> None:
        if bound is None:
            return
        if math.isnan(bound):
            return
        if self.obj_var is None:
            raise ValueError("Objective variable is not defined yet.")

        # If the bound is very close to an integer, treat it as such.
        # Otherwise, use ceiling to ensure we don't cut off valid integer solutions.
        if float_equals(bound, round(bound)):
            int_bound = round(bound)
        else:
            int_bound = math.ceil(bound)

        self.add(self.obj_var >= int_bound)

    # Constraints

    def define_constraints(self) -> None:
        # Alias for readability
        j_list = self.j_list
        i_list = self.i_list

        timer = ElapsedTimer()

        # NoOverlap on each stage:
        # Needless since precedence constraints and time-linking already enforce no-overlap.
        # for i in i_list:
        #     self.add_no_overlap([self.var_op_intvl[j, i] for j in j_list])

        # Precedence between consecutive stages for each job
        consecutive_stage_pairs = list(zip(i_list[:-1], i_list[1:]))
        for j in j_list:
            for i, next_i in consecutive_stage_pairs:
                self.add(
                    self.var_op_start[j, i] + self.p[j, i]
                    <= self.var_op_start[j, next_i]
                )

        logging.info(f"  Precedence constr. took {timer.elapsed_sec:.3f} sec.")
        timer.set_start_time_as_now()

        # Link precedence and time
        # prec[j1,j2] is True -> start[j1,i] + p[j1,i] <= start[j2,i] for all i
        # -> -stage_end_time_ub[i] + p[j2,i] <= start[j1,i] - start[j2,i] <= -p[j1,i]
        for j1, j2 in permutations(j_list, 2):
            for i in i_list:
                domain = (-self.stage_end_time_ub[i] + self.p[j2, i], -self.p[j1, i])
                self.add_linear_constraint_enforced_fast(
                    var_list=[self.var_op_start[j1, i], self.var_op_start[j2, i]],
                    coeff_list=[1, -1],
                    domain=domain,
                    enforcers=[self.prec[j1, j2]],
                )

        logging.info(f"  Time-linking constr. took {timer.elapsed_sec:.3f} sec.")

    #     self.add_rank_constraints()

    # def add_rank_constraints(self) -> None:
    #     sub_timer = ElapsedTimer()

    #     # Define rank variables
    #     rank_vars: dict[str, IntVar] = {}
    #     n = len(self.j_list)
    #     for j in self.j_list:
    #         rank_vars[j] = self.new_int_var(0, n - 1, f"rank_{j}")

    #     self.add_all_different(list(rank_vars.values()))

    #     # Link rank variables with precedence variables
    #     for j in self.j_list:
    #         self.add(
    #             sum(self.prec[j, jp] for jp in self.j_list if jp != j) == rank_vars[j]
    #         )

    #     logging.info(f"  Rank constraints took {sub_timer.elapsed_sec:.3f} sec.")

    # Extraction methods

    def extract_start_end_time_map(
        self,
    ) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int]]:
        """Extracts start and end times from a solved CP model.

        Returns:
            tuple: A tuple containing two dictionaries:
            - dict[tuple[str, str], int]: (job, stage) -> start time
            - dict[tuple[str, str], int]: (job, stage) -> end time
        """
        start_time_map: dict[tuple[str, str], int] = {}
        end_time_map: dict[tuple[str, str], int] = {}

        for j in self.j_list:
            for i in self.i_list:
                start = self.solver.value(self.var_op_start[j, i])
                start_time_map[j, i] = start
                end_time_map[j, i] = start + self.p[j, i]

        return start_time_map, end_time_map

    def extract_Tj_map(self) -> dict[str, int]:
        """Extract per-job tardiness values from solved CP model.

        Returns:
            dict[str, int]: job -> T_j
        """
        Tj_map: dict[str, int] = {}
        for j, var in self.var_T.items():
            Tj_map[j] = self.solver.value(var)
        return Tj_map

    def create_schedule_by_start_end_time(self) -> FlowshopSchedule:
        start_time_map, end_time_map = self.extract_start_end_time_map()
        schedule = FlowshopSchedule.from_stage_name_list(self.i_list)

        for j in self.j_list:
            for i in self.i_list:
                s = int(start_time_map[j, i])
                e = int(end_time_map[j, i])
                op = FlowshopOperation(job_name=j, stage_name=i, start=s, end=e)
                added = schedule.schedule_operation(op)
                assert added is not None, f"Failed to add operation {j},{i} to schedule"

        return schedule

    def create_schedule_from_sequence(self) -> FlowshopSchedule:
        i_list = self.i_list

        j_2_rank_map = {
            j: self.solver.value(sum(self.prec[jp, j] for jp in self.j_list if jp != j))
            for j in self.j_list
        }
        job_sequence = sorted(self.j_list, key=lambda j: j_2_rank_map[j])
        schedule = FlowshopSchedule.from_stage_name_list(self.i_list)

        for j in job_sequence:
            i_2_p_map = {i: self.p[j, i] for i in i_list}
            schedule.dispatch_job_by_stages(j, i_list, i_2_p_map, after_last=True)

        return schedule

    # methods to add hints

    def add_hints_from_schedule(self, schedule: FlowshopSchedule) -> None:
        self.add_tardiness_hints_from_Tj_map(schedule.get_job_2_tardiness_map(self.D))
        self.add_start_hints_from_start_time_map(schedule.get_start_time_map())
        self.add_indirect_precedence_hints(schedule.get_last_stage_job_list())

    def add_indirect_precedence_hints(self, job_sequence: list[str]) -> None:
        for j1_idx, j1 in enumerate(job_sequence):
            for j2 in job_sequence[j1_idx + 1 :]:
                self.add_hint(self.prec[(j1, j2)], 1)
                self.add_hint(self.prec[(j2, j1)], 0)

    def add_start_hints_from_start_time_map(
        self, start_time_map: dict[tuple[str, str], int]
    ) -> None:
        for (j, i), s_time in start_time_map.items():
            if (j, i) not in self.var_op_start:
                raise KeyError(f"Invalid job-stage pair: ({j}, {i})")
            self.add_hint(self.var_op_start[j, i], s_time)

    def add_tardiness_hints_from_Tj_map(self, Tj_map: dict[str, int]) -> None:
        sum_Tj = 0
        for j in self.j_list:
            Tj = Tj_map.get(j, 0)
            self.add_hint(self.var_T[j], Tj)
            sum_Tj += Tj
        self.add_hint(self.obj_var, sum_Tj)

    # Profiling methods

    def add_indirect_precedence_constraints_by_sequence(
        self, job_sequence: list[str]
    ) -> None:
        for idx, j1 in enumerate(job_sequence):
            for j2 in job_sequence[idx + 1 :]:
                self.add(self.prec[j1, j2] == 1)
                self.add(self.prec[j2, j1] == 0)

    # Subproblem generation

    def create_problem_of_job_subset(self, job_subset: set[str]) -> CpCpsatIndirectPrec:
        if not job_subset.issubset(set(self.j_list)):
            raise ValueError("job_subset must be a subset of the original job list.")
        new_model = self.__class__(self.horizon)

        # Filter parameters based on job_subset
        new_model.j_list = [j for j in self.j_list if j in job_subset]
        new_model.i_list = self.i_list.copy()
        new_model.p = {
            (j, i): self.p[j, i] for j in new_model.j_list for i in new_model.i_list
        }
        new_model.D = {j: self.D[j] for j in new_model.j_list}
        # Define variables, objective, and constraints
        new_model.define_variables()
        new_model.define_total_tardiness_objective()
        new_model.define_constraints()

        return new_model
