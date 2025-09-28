from __future__ import annotations

import logging
import math

from mbls.cpsat import CustomCpModel
from ortools.sat.python.cp_model import IntVar
from routix import ElapsedTimer
from routix.util.comparison import float_equals
from schore.parameters_examples.shop.flow import FlowshopDuedateParameters

from .scheduling.flowshop_schedule import FlowshopOperation, FlowshopSchedule


class CpCpsatRank(CustomCpModel):
    # Indices & Parameters

    horizon: int
    """
    The horizon for the scheduling problem, which is the maximum time
    that any operation can start or end.
    This is used to define the domain of the start and end time variables.
    """
    j_list: list[str]
    """$J$: job index (j) list"""

    i_list: list[str]
    """$I$: stage index (i) list"""

    k_list: list[int]
    """$K$: rank index (k) list"""

    p: dict[tuple[str, str], int]
    """$P_{ji}$: processing time of job j at stage i"""

    D: dict[str, int]
    """$D_j$: due date of job j"""

    # Variables

    var_op_start: dict[tuple[str, int], IntVar]
    """
    (i, k) -> start time of a k-th job in stage i
    """
    var_op_lth: dict[tuple[str, int], IntVar]
    """
    (i, k) -> processing time of a k-th job in stage i
    """
    var_op_end: dict[tuple[str, int], IntVar]
    """
    (i, k) -> end time of a k-th job in stage i
    """
    var_r: dict[int, IntVar]
    """$r_k$: index of the k-th job"""

    var_d: dict[int, IntVar]
    """$d_k$: due date of the k-th job"""

    var_T: dict[int, IntVar]
    """$T_k$: tardiness of the k-th job"""

    # Objective

    obj_var: IntVar
    """Defines the objective for the scheduling problem."""

    def __init__(self, horizon: int) -> None:
        super().__init__()
        self.horizon = horizon

    @classmethod
    def from_instance(
        cls, instance: FlowshopDuedateParameters, horizon: int
    ) -> CpCpsatRank:
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
        self.define_constraints()
        logging.info(f"Defined constraints; took {elapsed.elapsed_sec:.3f} sec.")

        elapsed.set_start_time_as_now()
        self.define_total_tardiness_objective()
        logging.info(f"Defined objective; took {elapsed.elapsed_sec:.3f} sec.")

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
        self.k_list = list(range(len(self.j_list)))

    # Variables

    def define_variables(self) -> None:
        j_list = self.j_list
        i_list = self.i_list
        k_list = self.k_list

        # Interval variables
        self.var_op_start = {}
        self.var_op_lth = {}
        self.var_op_end = {}
        for i in i_list:
            p_vals_i = [self.p[j, i] for j in j_list]
            p_min_i = min(p_vals_i)
            p_max_i = min(max(p_vals_i), self.horizon)
            for k in k_list:
                suffix = f"{i}_{k}"
                self.var_op_start[i, k] = self.NewIntVar(
                    0, self.horizon - p_min_i, f"start_{suffix}"
                )
                self.var_op_lth[i, k] = self.NewIntVar(
                    p_min_i, p_max_i, f"lth_{suffix}"
                )
                self.var_op_end[i, k] = self.NewIntVar(
                    p_min_i, self.horizon, f"end_{suffix}"
                )

        # Rank
        self.var_r = {k: self.NewIntVar(0, len(j_list) - 1, f"r_{k}") for k in k_list}

        # Due date of k-th job
        d_vals = [self.D[j] for j in j_list]
        d_min = min(d_vals)
        d_max = min(max(d_vals), self.horizon)
        self.var_d = {k: self.NewIntVar(d_min, d_max, f"d_{k}") for k in self.k_list}

        # Tardiness
        self.var_T = {}
        for k in k_list:
            self.var_T[k] = self.NewIntVar(0, self.horizon - d_min, f"T_{k}")

    # Objective

    def define_total_tardiness_objective(self) -> None:
        """
        Total tardiness objective: minimize \\sum_k{T_k} where T_k := max(end_k - D_k, 0).

        Uses `add_max_equality` for clarity.
        """
        k_list = self.k_list
        last_i = self.i_list[-1]

        for k in k_list:
            self.add_max_equality(
                self.var_T[k], [self.var_op_end[last_i, k] - self.var_d[k], 0]
            )
        total_ub = sum(max(0, self.horizon - self.D[j]) for j in self.j_list)

        self.obj_var = self.new_int_var(0, total_ub, "sum_Tk")
        self.add(self.obj_var == sum(self.var_T[k] for k in k_list))

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
        k_list = self.k_list

        timer = ElapsedTimer()

        # All-different constraint on rank variables
        self.add_all_different([self.var_r[k] for k in k_list])

        logging.info(f"  All-different constraints took {timer.elapsed_sec:.3f} sec.")
        timer.set_start_time_as_now()

        # Processing time of each operation
        # lth_{i,k} = sum_j{P_{j,i} * (r_k == j_index)}
        # This uses the element constraint.
        for i in i_list:
            p_vals_i = [self.p[j, i] for j in j_list]
            for k in k_list:
                self.add_element(self.var_r[k], p_vals_i, self.var_op_lth[i, k])

        logging.info(f"  Processing time constraints took {timer.elapsed_sec:.3f} sec.")
        timer.set_start_time_as_now()

        # Due date of each job
        # d_k = sum_j{D_j * (r_k == j_index)}
        d_vals = [self.D[j] for j in j_list]
        for k in k_list:
            self.add_element(self.var_r[k], d_vals, self.var_d[k])

        logging.info(f"  Due date constraints took {timer.elapsed_sec:.3f} sec.")
        timer.set_start_time_as_now()

        # Precedence between consecutive stages for each job
        consecutive_stage_pairs = list(zip(i_list[:-1], i_list[1:]))
        for k in k_list:
            for i, next_i in consecutive_stage_pairs:
                self.add(self.var_op_end[i, k] <= self.var_op_start[next_i, k])

        logging.info(f"  Precedence constraints took {timer.elapsed_sec:.3f} sec.")
        timer.set_start_time_as_now()

        # Precedence between operations in the same stage
        for i in i_list:
            for k in k_list:
                self.add(
                    self.var_op_end[i, k]
                    == self.var_op_start[i, k] + self.var_op_lth[i, k]
                )
            for k in k_list[1:]:
                self.add(self.var_op_start[i, k] >= self.var_op_end[i, k - 1])

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
        j_list = self.j_list
        i_list = self.i_list
        k_list = self.k_list

        start_time_map: dict[tuple[str, str], int] = {}
        end_time_map: dict[tuple[str, str], int] = {}

        for k in k_list:
            r_val = self.solver.Value(self.var_r[k])
            job = j_list[r_val]
            for i in i_list:
                start_time = self.solver.Value(self.var_op_start[i, k])
                end_time = self.solver.Value(self.var_op_end[i, k])
                start_time_map[job, i] = start_time
                end_time_map[job, i] = end_time

        return start_time_map, end_time_map

    def extract_Tj_map(self) -> dict[str, int]:
        """Extract per-job tardiness values from solved CP model.

        Returns:
            dict[str, int]: job -> T_j
        """
        j_list = self.j_list
        k_list = self.k_list

        Tj_map: dict[str, int] = {}
        for k in k_list:
            r_val = self.solver.Value(self.var_r[k])
            job = j_list[r_val]
            T_val = self.solver.Value(self.var_T[k])
            Tj_map[job] = T_val

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
