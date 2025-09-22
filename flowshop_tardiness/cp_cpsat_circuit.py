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


class CpCpsatCircuit(CpModelWithFixedInterval):
    """
    Implementation of the CP model for the flowshop problem to minimize total tardiness
    using circuit constraints.

    Reference:

    - [Google OR-Tools sample for sequences in no-overlap](https://github.com/google/or-tools/blob/7ee639cf6981a9beeba908cf543a50f4ee7413ad/ortools/sat/samples/sequences_in_no_overlap_sample_sat.py#L148)
    """

    # Indices & Parameters

    j_list: list[str]
    """$J$: job index (j) list"""

    i_list: list[str]
    """$I$: stage index (i) list"""

    p: dict[tuple[str, str], int]
    """$P_{ji}$: processing time of job j at stage i"""

    D: dict[str, int]
    """$D_j$: due date of job j"""

    # Variables

    var_Tj: dict[str, IntVar]
    """$T_j$: tardiness of job j"""

    # var_Lj: dict[str, IntVar]
    # """$L_j$: lateness of job j"""

    # var_Uj: dict[str, IntVar]
    # """$U_j$: binary variable indicating if job j is late (1 if late, 0 otherwise)"""

    # Objective

    obj_var: IntVar
    """Defines the objective for the scheduling problem."""

    def __init__(self, horizon: int) -> None:
        super().__init__(horizon)

    @classmethod
    def from_instance(
        cls, instance: FlowshopDuedateParameters, horizon: int
    ) -> CpCpsatCircuit:
        result = cls(horizon)
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
        _p = instance.p_manager.job_stage_2_value_map(self.j_list, self.i_list)
        self.p = {
            (j, i): int(round(_p[j, i])) for j in self.j_list for i in self.i_list
        }
        self.D = {j: int(round(instance.job_2_duedate_map[j])) for j in self.j_list}

    # Variables

    def define_variables(self) -> None:
        # Interval variables
        for j in self.j_list:
            for i in self.i_list:
                self.define_fixed_interval_var((j, i), self.p[j, i])

    # Objective

    def define_total_tardiness_objective(self) -> None:
        """
        Total tardiness objective: minimize \\sum_j{T_j} where T_j := max(end_j - D_j, 0).

        Rationale for not using add_max_equality:
        https://d-krupke.github.io/cpsat-primer/04_modelling.html?highlight=add_max_equality#04-modelling-absmaxmin
        """
        j_list = self.j_list
        last_i = self.i_list[-1]

        self.var_Tj = {}
        total_ub = 0
        for j in j_list:
            ub = max(self.horizon - self.D[j], 0)
            self.var_Tj[j] = self.new_int_var(0, ub, f"T_{j}")
            self.add(self.var_Tj[j] >= self.var_op_end[j, last_i] - self.D[j])
            total_ub += ub

        self.obj_var = self.new_int_var(0, total_ub, "sum_Tj")
        self.add(self.obj_var == sum(self.var_Tj[j] for j in j_list))

        self.minimize(self.obj_var)

    def define_total_tardiness_objective_max_equality(self) -> None:
        """
        Total tardiness objective: minimize \\sum_j{T_j} where T_j := max(end_j - D_j, 0).

        Uses `add_max_equality` for clarity.
        """
        j_list = self.j_list
        last_i = self.i_list[-1]

        self.var_Tj = {}
        total_ub = 0
        for j in j_list:
            ub = max(self.horizon - self.D[j], 0)
            self.var_Tj[j] = self.new_int_var(0, ub, f"T_{j}")
            self.add_max_equality(
                self.var_Tj[j], [self.var_op_end[j, last_i] - self.D[j], 0]
            )
            total_ub += ub

        self.obj_var = self.new_int_var(0, total_ub, "sum_Tj")
        self.add(self.obj_var == sum(self.var_Tj[j] for j in j_list))

        self.minimize(self.obj_var)

    # def define_total_tardiness_objective_1(self) -> None:
    #     """Total tardiness objective: minimize \\sum_j{T_j} where T_j := max(end_j - D_j, 0)."""
    #     j_list = self.j_list
    #     last_i = self.i_list[-1]

    #     self.var_Lj = {}
    #     self.var_Uj = {}
    #     self.var_Tj = {}
    #     total_ub = 0
    #     for j in j_list:
    #         self.var_Uj[j] = self.new_bool_var(f"U_{j}")

    #         self.var_Lj[j] = self.new_int_var(
    #             -self.D[j], self.horizon - self.D[j], f"L_{j}"
    #         )
    #         self.add(self.var_Lj[j] == self.var_op_end[j, last_i] - self.D[j])

    #         self.add(self.var_Lj[j] >= 1).only_enforce_if(self.var_Uj[j])
    #         self.add(self.var_Lj[j] <= 0).only_enforce_if(self.var_Uj[j].Not())

    #         ub = max(self.horizon - self.D[j], 0)
    #         self.var_Tj[j] = self.new_int_var(0, ub, f"T_{j}")
    #         self.add(self.var_Tj[j] == self.var_Lj[j]).only_enforce_if(self.var_Uj[j])
    #         self.add(self.var_Tj[j] == 0).only_enforce_if(self.var_Uj[j].Not())
    #         total_ub += ub

    #     self.obj_var = self.new_int_var(0, total_ub, "sum_Tj")
    #     self.add(self.obj_var == sum(self.var_Tj[j] for j in j_list))

    #     self.minimize(self.obj_var)

    # def define_total_tardiness_objective_2(self) -> None:
    #     """Total tardiness objective: minimize \\sum_j{T_j} where T_j := max(end_j - D_j, 0)."""
    #     j_list = self.j_list
    #     last_i = self.i_list[-1]

    #     self.var_Lj = {}
    #     self.var_Ej = {}
    #     self.var_Tj = {}
    #     total_ub = 0
    #     for j in j_list:
    #         self.var_Lj[j] = self.new_int_var(
    #             -self.D[j], self.horizon - self.D[j], f"L_{j}"
    #         )
    #         self.add(self.var_Lj[j] == self.var_op_end[j, last_i] - self.D[j])

    #         self.var_Ej[j] = self.new_int_var(0, self.D[j], f"E_{j}")

    #         ub = max(self.horizon - self.D[j], 0)
    #         self.var_Tj[j] = self.new_int_var(0, ub, f"T_{j}")

    #         self.add(self.var_Lj[j] == self.var_Tj[j] - self.var_Ej[j])

    #         total_ub += ub

    #     self.obj_var = self.new_int_var(0, total_ub, "sum_Tj")
    #     self.add(self.obj_var == sum(self.var_Tj[j] for j in j_list))

    #     self.minimize(self.obj_var)

    def set_obj_lower_bound(self, bound: float) -> None:
        if self.obj_var is None:
            return

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

        # Precedence between consecutive stages for each job
        consecutive_stage_pairs = list(zip(i_list[:-1], i_list[1:]))
        for jInt in j_list:
            for i, next_i in consecutive_stage_pairs:
                self.add(self.var_op_end[jInt, i] <= self.var_op_start[jInt, next_i])

        logging.info(f"  Precedence constraints took {timer.elapsed_sec:.3f} sec.")
        timer.set_start_time_as_now()

        # NoOverlap on each stage: needless since circuit constraint implies it
        # for i in i_list:
        #     self.add_no_overlap([self.var_op_intvl[j, i] for j in j_list])

        # Circuit constraints to enforce permutation schedule on each stage
        # Integer list for jobs (1..n) and a dummy (0)
        integer_j_2_j_map = {idx + 1: j for idx, j in enumerate(j_list)}
        integer_j_list = sorted(integer_j_2_j_map.keys())
        integer_j_and_dummy_list = [0] + integer_j_list

        arcs = {}
        arc_list = []
        # \forall j\in J, j'\in J, j \neq j':
        for jInt, jpInt in permutations(integer_j_and_dummy_list, 2):
            j = integer_j_2_j_map.get(jInt, "dummy")
            jp = integer_j_2_j_map.get(jpInt, "dummy")
            j_before_jp = self.new_bool_var(f"arc_{j}_{jp}")
            arcs[jInt, jpInt] = j_before_jp
            arc_list.append((jInt, jpInt, j_before_jp))

        # Single hamiltonian circuit
        self.add_circuit(arc_list)

        logging.info(f"  Circuit constraints took {timer.elapsed_sec:.3f} sec.")
        timer.set_start_time_as_now()

        # Link circuit arcs with start/end times
        # If arc (j, j') is selected, then end_j <= start_j' \forall j\in J, j'\in J, j \neq j'
        # Dummy arcs are not time-linked intentionally
        lb = -self.horizon
        ub = 0
        for jInt, jpInt in permutations(integer_j_list, 2):
            j = integer_j_2_j_map.get(jInt, "dummy")
            jp = integer_j_2_j_map.get(jpInt, "dummy")
            j_before_jp = arcs[jInt, jpInt]

            for i in i_list:
                self.add_linear_constraint_enforced_fast(
                    var_list=[self.var_op_end[j, i], self.var_op_start[jp, i]],
                    coeff_list=[1, -1],
                    domain=(lb, ub),
                    enforcers=[j_before_jp],
                )

        logging.info(f"  Time-linking constraints took {timer.elapsed_sec:.3f} sec.")
        # timer.set_start_time_as_now()

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
                end = self.solver.value(self.var_op_end[j, i])
                start_time_map[j, i] = start
                end_time_map[j, i] = end

        return start_time_map, end_time_map

    def extract_Tj_map(self) -> dict[str, int]:
        """Extract per-job tardiness T_j values from solved CP model.

        Returns:
            dict[str, int]: job -> T_j
        """
        Tj_map: dict[str, int] = {}
        for j, var in self.var_Tj.items():
            Tj_map[j] = self.solver.value(var)
        return Tj_map

    def create_schedule(self) -> FlowshopSchedule:
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

    # methods to add hints

    def add_start_hints_from_start_time_map(
        self, start_time_map: dict[tuple[str, str], int]
    ) -> None:
        for (j, i), s_time in start_time_map.items():
            if (j, i) not in self.var_op_start:
                raise KeyError(f"Invalid job-stage pair: ({j}, {i})")
            self.add_hint(self.var_op_start[j, i], s_time)
