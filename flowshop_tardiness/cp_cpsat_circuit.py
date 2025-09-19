from __future__ import annotations

from itertools import permutations

from mbls.cpsat.cp_model_with_fixed_interval import CpModelWithFixedInterval
from ortools.sat.python.cp_model import IntVar
from schore.parameters_examples.shop.flow import FlowshopDuedateParameters


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
    """$D_{j}$: due date of job j"""

    # Objective

    obj_var: IntVar
    """Defines the makespan objective for the scheduling problem."""

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
        self.define_parameters(instance)
        self.define_variables()
        self.define_total_tardiness_objective()
        self.define_constraints()

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
            (j, i): int(float(_p[j, i])) for j in self.j_list for i in self.i_list
        }
        self.D = {j: int(float(instance.job_2_duedate_map[j])) for j in self.j_list}

    # Variables

    def define_variables(self) -> None:
        # Interval variables
        for j in self.j_list:
            for i in self.i_list:
                self.define_fixed_interval_var((j, i), self.p[j, i])

    # Objective

    def define_total_tardiness_objective(self) -> None:
        """Total tardiness objective: minimize sum_j max(end_j - D_j, 0)."""
        j_list = self.j_list
        last_i = self.i_list[-1]

        tard_vars: list[IntVar] = []
        for j in j_list:
            tard_j = self.new_int_var(0, self.horizon - self.D[j], f"tard_{j}")
            self.add_max_equality(tard_j, [self.var_op_end[j, last_i] - self.D[j], 0])
            tard_vars.append(tard_j)

        total_ub = sum(self.horizon - self.D[j] for j in j_list)
        total_tard = self.new_int_var(0, total_ub, "total_tardiness")
        # sum equality
        self.add(total_tard == sum(tard_vars))

        self.minimize(total_tard)
        self.obj_var = total_tard

    # Constraints

    def define_constraints(self) -> None:
        # Alias for readability
        j_list = self.j_list
        i_list = self.i_list

        # Precedence between consecutive stages for each job
        consecutive_stage_pairs = list(zip(i_list[:-1], i_list[1:]))
        for jInt in j_list:
            for i, next_i in consecutive_stage_pairs:
                self.add(self.var_op_end[jInt, i] <= self.var_op_start[jInt, next_i])

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

        # Link circuit arcs with start/end times
        # If arc (j, j') is selected, then end_j <= start_j'
        for jInt, jpInt in permutations(integer_j_list, 2):
            j = integer_j_2_j_map.get(jInt, "dummy")
            jp = integer_j_2_j_map.get(jpInt, "dummy")
            j_before_jp = arcs[jInt, jpInt]
            for i in i_list:
                self.add(
                    self.var_op_end[j, i] <= self.var_op_start[jp, i],
                ).only_enforce_if(j_before_jp)

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
