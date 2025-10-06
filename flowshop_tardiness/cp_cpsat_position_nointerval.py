from __future__ import annotations

import logging
import math

from mbls.cpsat import CustomCpModel
from ortools.sat.python.cp_model import IntVar
from routix import ElapsedTimer
from routix.util.comparison import float_equals
from schore.parameters_examples.shop.flow import FlowshopDuedateParameters

from .scheduling.flowshop_schedule import FlowshopOperation, FlowshopSchedule


class CpCpsatPosition(CustomCpModel):
    # Indices & Parameters

    j_2_job_name_map: dict[int, str]
    """Mapping from job index (j) to job name"""

    job_name_2_j_map: dict[str, int]
    """Mapping from job name to job index (j)"""

    j_list: list[int]
    """$J$: job index (j) list; 0..(n-1)"""

    j_first: int
    """Index of the first job (0)"""

    j_last: int
    """Index of the last job (n-1)"""

    i_2_stage_name_map: dict[int, str]
    """Mapping from stage index (i) to stage name"""

    i_list: list[int]
    """$I$: stage index (i) list; 0..(m-1)"""

    p: dict[tuple[int, int], int]
    """$p_{ji}$: processing time of job j at stage i"""

    stage_start_time_lb: dict[int, int]
    """i -> lower bound on the start time of the stage."""

    stage_end_time_ub: dict[int, int]
    """i -> upper bound on the makespan of the stage."""

    D: dict[int, int]
    """$D_j$: due date of job j"""

    # Variables

    var_op_start: dict[tuple[int, int], IntVar]
    """
    (i, k) -> start time of a k-th job in stage i
    """
    var_op_lth: dict[tuple[int, int], IntVar]
    """
    (i, k) -> processing time of a k-th job in stage i
    """
    var_op_end: dict[tuple[int, int], IntVar]
    """
    (i, k) -> end time of a k-th job in stage i
    """
    var_pi: dict[int, IntVar]
    """$pi_k$: job index (j) at position k"""

    var_d: dict[int, IntVar]
    """$d_k$: due date of job at k-th position"""

    var_T: dict[int, IntVar]
    """$T_k$: tardiness of job at k-th position"""

    # Objective

    obj_var: IntVar
    """Defines the objective for the scheduling problem."""

    def __init__(self, horizon: int) -> None:
        super().__init__()

    @classmethod
    def from_instance(
        cls, instance: FlowshopDuedateParameters, horizon: int
    ) -> CpCpsatPosition:
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
        """Define the parameters for the model based on the FlowshopDuedateParameters instance.

        Args:
            instance (FlowshopDuedateParameters): The flow shop problem instance.
        """
        self.j_2_job_name_map = {j: name for j, name in enumerate(instance.job_id_list)}
        self.job_name_2_j_map = {name: j for j, name in self.j_2_job_name_map.items()}
        self.j_list = list(range(len(instance.job_id_list)))
        n = len(self.j_list)
        self.j_first = 0
        self.j_last = n - 1

        self.i_2_stage_name_map = {
            i: name for i, name in enumerate(instance.stage_id_list)
        }
        self.i_list = list(range(len(instance.stage_id_list)))

        _p = instance.p_manager.job_stage_2_value_map(
            instance.job_id_list, instance.stage_id_list
        )
        self.p = {
            (j, i): int(round(_p[j_name, i_name]))
            for j, j_name in self.j_2_job_name_map.items()
            for i, i_name in self.i_2_stage_name_map.items()
        }

        self.stage_start_time_lb = {}
        cumulative_p_min = 0
        for i in self.i_list:
            self.stage_start_time_lb[i] = cumulative_p_min
            cumulative_p_min += min(self.p[j, i] for j in self.j_list)

        self.stage_end_time_ub = {}
        for i in self.i_list:
            candid1 = sum(
                self.p[j, ip] for j in self.j_list for ip in self.i_list[: i + 1]
            )
            # max processing time * (job count + stage count - 1)
            p_max = max(self.p[j, ip] for j in self.j_list for ip in range(0, i + 1))
            candid2 = p_max * (n + i)
            self.stage_end_time_ub[i] = min(candid1, candid2)

        self.D = {
            j: int(round(instance.job_2_duedate_map[j_name]))
            for j, j_name in self.j_2_job_name_map.items()
        }

    # Variables

    def define_variables(self) -> None:
        j_list = self.j_list
        i_list = self.i_list

        # Interval variables
        self.var_op_start = {}
        self.var_op_lth = {}
        self.var_op_end = {}

        for i in i_list:
            p_set = {self.p[j, i] for j in j_list}
            p_min_i = min(p_set)
            p_max_i = max(p_set)
            for k in j_list:
                suffix = f"{i}_{k}"
                self.var_op_start[i, k] = self.new_int_var(
                    self.stage_start_time_lb[i],
                    self.stage_end_time_ub[i] - p_min_i,
                    f"start_{suffix}",
                )
                self.var_op_lth[i, k] = self.new_int_var(
                    p_min_i, p_max_i, f"lth_{suffix}"
                )
                self.var_op_end[i, k] = self.new_int_var(
                    self.stage_start_time_lb[i] + p_min_i,
                    self.stage_end_time_ub[i],
                    f"end_{suffix}",
                )

        # Position variables
        self.var_pi = {
            k: self.new_int_var(self.j_first, self.j_last, f"pi_{k}") for k in j_list
        }

        D_set = {self.D[j] for j in j_list}
        d_min = min(D_set)
        d_max = max(D_set)
        self.var_d = {k: self.new_int_var(d_min, d_max, f"d_{k}") for k in j_list}

        # Tardiness of k-th job
        last_i = self.i_list[-1]
        self.var_T = {
            k: self.new_int_var(0, self.stage_end_time_ub[last_i] - d_min, f"T_{k}")
            for k in j_list
        }

    # Objective

    def define_total_tardiness_objective(self) -> None:
        """
        Total tardiness objective: minimize \\sum_k{T_k} where T_k := max(end_k - D_k, 0).

        Uses `add_max_equality` for clarity.
        """
        j_list = self.j_list
        last_i = self.i_list[-1]

        for k in j_list:
            # self.add_max_equality(
            #     self.var_T[k], [self.var_op_end[last_i, k] - self.var_d[k], 0]
            # )
            self.add(self.var_T[k] >= self.var_op_end[last_i, k] - self.var_d[k])

        total_ub = sum(
            max(0, self.stage_end_time_ub[last_i] - self.D[j]) for j in self.j_list
        )
        self.obj_var = self.new_int_var(0, total_ub, "sum_Tk")
        self.add(self.obj_var == sum(self.var_T[k] for k in j_list))

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

        # All-different constraint on sequence variables
        self.add_all_different([self.var_pi[k] for k in j_list])

        logging.info(f"  All-different constr. took {timer.elapsed_sec:.3f} sec.")
        timer.set_start_time_as_now()

        # Processing time of each operation
        # lth_{i,k} = sum_j{P_{j,i} * (pi_k == j_index)} \forall i\in I, k\in K
        # This uses the element constraint.
        for i in i_list:
            p_vals_i = [self.p[j, i] for j in j_list]
            for k in j_list:
                self.add_element(self.var_pi[k], p_vals_i, self.var_op_lth[i, k])

        logging.info(f"  Processing time constr. took {timer.elapsed_sec:.3f} sec.")
        timer.set_start_time_as_now()

        # Due date of each job
        # d_k = sum_j{D_j * (pi_k == j_index)}
        d_vals = [self.D[j] for j in j_list]
        for k in j_list:
            self.add_element(self.var_pi[k], d_vals, self.var_d[k])

        logging.info(f"  Due date constr. took {timer.elapsed_sec:.3f} sec.")
        timer.set_start_time_as_now()

        # Precedence between consecutive stages for each job
        for i, next_i in zip(i_list[:-1], i_list[1:]):
            for k in j_list:
                self.add(self.var_op_end[i, k] <= self.var_op_start[next_i, k])

        logging.info(
            f"  Precedence (inter-stage) constr. took {timer.elapsed_sec:.3f} sec."
        )
        timer.set_start_time_as_now()

        # Precedence between operations in the same stage
        for i in i_list:
            for k in j_list:
                self.add(
                    self.var_op_end[i, k]
                    == self.var_op_start[i, k] + self.var_op_lth[i, k]
                )
                if k != self.j_first:
                    self.add(self.var_op_end[i, k - 1] <= self.var_op_start[i, k])

        logging.info(
            f"  Precedence (intra-stage) constr. took {timer.elapsed_sec:.3f} sec."
        )
        # timer.set_start_time_as_now()

    # Extraction methods

    def extract_start_end_time_map(
        self,
    ) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int]]:
        """Extracts start and end times from a solved CP model.

        Returns:
            tuple: A tuple containing two dictionaries:
            - dict[tuple[str, str], int]: (job name, stage name) -> start time
            - dict[tuple[str, str], int]: (job name, stage name) -> end time
        """
        j_list = self.j_list
        i_list = self.i_list

        start_time_map: dict[tuple[str, str], int] = {}
        end_time_map: dict[tuple[str, str], int] = {}

        for k in j_list:
            j = self.solver.Value(self.var_pi[k])
            j_name = self.j_2_job_name_map[j]
            for i in i_list:
                start_time = self.solver.Value(self.var_op_start[i, k])
                end_time = self.solver.Value(self.var_op_end[i, k])
                i_name = self.i_2_stage_name_map[i]
                start_time_map[j_name, i_name] = start_time
                end_time_map[j_name, i_name] = end_time

        return start_time_map, end_time_map

    def extract_Tj_map(self) -> dict[str, int]:
        """Extract per-job tardiness values from solved CP model.

        Returns:
            dict[str, int]: job name -> T_j
        """
        j_list = self.j_list

        Tj_map: dict[str, int] = {}
        for k in j_list:
            j = self.solver.Value(self.var_pi[k])
            T_val = self.solver.Value(self.var_T[k])
            j_name = self.j_2_job_name_map[j]
            Tj_map[j_name] = T_val

        return Tj_map

    def create_schedule_by_start_end_time(self) -> FlowshopSchedule:
        i_name_list = [self.i_2_stage_name_map[i] for i in self.i_list]
        start_time_map, end_time_map = self.extract_start_end_time_map()
        schedule = FlowshopSchedule.from_stage_name_list(i_name_list)

        for j in self.j_list:
            j_name = self.j_2_job_name_map[j]
            for i in self.i_list:
                i_name = self.i_2_stage_name_map[i]
                s = int(start_time_map[j_name, i_name])
                e = int(end_time_map[j_name, i_name])
                op = FlowshopOperation(
                    job_name=j_name, stage_name=i_name, start=s, end=e
                )
                added = schedule.schedule_operation(op)
                assert added is not None, (
                    f"Failed to add operation {j_name},{i_name} to schedule"
                )

        return schedule

    def create_schedule_from_sequence(self) -> FlowshopSchedule:
        j_sequence = [self.solver.Value(self.var_pi[k]) for k in self.j_list]

        i_name_list = [self.i_2_stage_name_map[i] for i in self.i_list]
        schedule = FlowshopSchedule.from_stage_name_list(i_name_list)

        for j in j_sequence:
            j_name = self.j_2_job_name_map[j]
            i_2_p_map = {
                i_name: self.p[j, i] for i, i_name in self.i_2_stage_name_map.items()
            }
            schedule.dispatch_job_by_stages(
                j_name, i_name_list, i_2_p_map, after_last=True
            )

        return schedule

    # methods to add hints

    def add_hints_from_schedule(self, schedule: FlowshopSchedule) -> None:
        last_i = self.i_list[-1]
        last_i_name = self.i_2_stage_name_map[last_i]
        j_sequence = schedule.get_last_stage_job_list()
        start_time_map = schedule.get_start_time_map()
        sum_Tj = 0

        all_ops_in_schedule = True
        for k, j_name in enumerate(j_sequence):
            j = self.job_name_2_j_map[j_name]
            all_ops_of_j_in_schedule = True
            for i, i_name in self.i_2_stage_name_map.items():
                if (j_name, i_name) in start_time_map:
                    start_hint = start_time_map[j_name, i_name]
                    p = self.p[j, i]
                    self.add_hint(self.var_op_start[i, k], start_hint)
                    self.add_hint(self.var_op_lth[i, k], p)
                    self.add_hint(self.var_op_end[i, k], start_hint + p)
                else:
                    all_ops_in_schedule = False
                    all_ops_of_j_in_schedule = False
            self.add_hint(self.var_pi[k], j)
            self.add_hint(self.var_d[k], self.D[j])
            if all_ops_of_j_in_schedule:
                assert (j_name, last_i_name) in start_time_map, (
                    f"Last operation of job {j_name} not found in start_time_map"
                )
                Tj = max(
                    0,
                    start_time_map[j_name, last_i_name] + self.p[j, last_i] - self.D[j],
                )
                self.add_hint(self.var_T[k], Tj)
                sum_Tj += Tj
        if all_ops_in_schedule:
            self.add_hint(self.obj_var, sum_Tj)

    # Profiling methods

    def add_indirect_precedence_constraints_by_sequence(
        self, job_sequence: list[str]
    ) -> None:
        self.var_pi_inverse = {
            j: self.new_int_var(self.j_first, self.j_last, f"pi_inv_{j}")
            for j in self.j_list
        }

        for j1_name, j2_name in zip(job_sequence[:-1], job_sequence[1:]):
            j1 = self.job_name_2_j_map[j1_name]
            j2 = self.job_name_2_j_map[j2_name]
            self.add(self.var_pi_inverse[j1] + 1 <= self.var_pi_inverse[j2])

    # Subproblem generation

    def create_problem_of_job_subset(self, job_subset: set[str]) -> CpCpsatPosition:
        if not job_subset.issubset(set(self.job_name_2_j_map.keys())):
            raise ValueError("job_subset contains unknown job names.")
        new_model = self.__class__(0)

        n = len(job_subset)
        j_list = list(range(n))
        # Filter parameters based on j_subset
        new_model.j_list = j_list
        new_model.j_first = min(new_model.j_list)
        new_model.j_last = max(new_model.j_list)
        new_model.j_2_job_name_map = {
            j: name for j, name in enumerate(sorted(job_subset))
        }
        new_model.job_name_2_j_map = {
            name: j for j, name in new_model.j_2_job_name_map.items()
        }
        new_model.i_list = list(self.i_list)
        new_model.i_2_stage_name_map = dict(self.i_2_stage_name_map)

        _p = {
            (j_name, i): self.p[self.job_name_2_j_map[j_name], i]
            for j_name in job_subset
            for i in self.i_list
        }

        new_model.p = {
            (j, i): _p[j_name, i]
            for j, j_name in new_model.j_2_job_name_map.items()
            for i in new_model.i_list
        }

        new_model.stage_start_time_lb = {}
        cumulative_p_min = 0
        for i in self.i_list:
            new_model.stage_start_time_lb[i] = cumulative_p_min
            cumulative_p_min += min(new_model.p[j, i] for j in new_model.j_list)

        new_model.stage_end_time_ub = {}
        for i in self.i_list:
            candid1 = sum(
                new_model.p[j, ip]
                for j in new_model.j_list
                for ip in new_model.i_list[: i + 1]
            )
            # max processing time * (job count + stage count - 1)
            p_max = max(
                new_model.p[j, ip] for j in new_model.j_list for ip in range(0, i + 1)
            )
            candid2 = p_max * (n + i)
            new_model.stage_end_time_ub[i] = min(candid1, candid2)

        new_model.D = {j: self.D[j] for j in new_model.j_list}
        new_model.define_variables()
        new_model.define_constraints()
        new_model.define_total_tardiness_objective()

        return new_model
