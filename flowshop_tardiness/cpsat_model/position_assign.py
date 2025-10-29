from __future__ import annotations

import logging
import math
from typing import Literal

from mbls.cpsat import CustomCpModel
from ortools.sat.python.cp_model import IntVar, LinearExpr
from routix import ElapsedTimer
from schore.parameters_examples.shop.flow import FlowshopDuedateParameters
from schore.schedule_examples.shop.flow import FlowshopSchedule


class PositionAssignModel(CustomCpModel):
    # Indices & Parameters

    j_2_job_name_map: dict[int, str]
    """Mapping from job index (j) to job name"""

    job_name_2_j_map: dict[str, int]
    """Mapping from job name to job index (j)"""

    j_list: list[int]
    """$J$: job index (j) list; 0..(n-1)"""

    j_last: int
    """Index of the last job (n-1)"""

    i_2_stage_name_map: dict[int, str]
    """Mapping from stage index (i) to stage name"""

    i_list: list[int]
    """$I$: stage index (i) list; 0..(m-1)"""

    i_last: int
    """Index of the last stage (m-1)"""

    p: dict[int, dict[int, int]]
    """$p_{ij}$: processing time of job j at stage i"""

    stage_start_time_lb: dict[int, int]
    """i -> lower bound on the start time of the stage."""

    stage_end_time_ub: dict[int, int]
    """i -> upper bound on the makespan of the stage."""

    D: dict[int, int]
    """$D_j$: due date of job j"""

    # Variables

    y: dict[int, dict[int, IntVar]]
    """$y_{jr}$: assignment matrix; 1 if job j is assigned to position r, 0 otherwise"""

    P: dict[int, dict[int, LinearExpr | Literal[0]]]
    """$P_{rm}$: processing time of the job assigned to position r at stage m"""

    C: dict[int, dict[int, IntVar]]
    """$C_{rm}$: completion time of the job at position r at stage m"""

    # Objective

    obj_var: IntVar
    """Defines the objective for the scheduling problem."""

    def __init__(self, horizon: int) -> None:
        super().__init__()

    @classmethod
    def from_instance(
        cls, instance: FlowshopDuedateParameters, horizon: int
    ) -> PositionAssignModel:
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
        self.k_list = self.j_list  # alias for positions
        self.j_last = self.j_list[-1]

        self.i_2_stage_name_map = {
            i: name for i, name in enumerate(instance.stage_id_list)
        }
        self.i_list = list(range(len(instance.stage_id_list)))
        self.i_last = self.i_list[-1]

        _p = instance.p_manager.stage_2_job_2_value_map(
            instance.stage_id_list, instance.job_id_list
        )
        self.p = {
            i: {j: _p[i_name][j_name] for j, j_name in self.j_2_job_name_map.items()}
            for i, i_name in self.i_2_stage_name_map.items()
        }

        self.stage_start_time_lb = {}
        cumulative_p_min = 0
        for i in self.i_list:
            stage_p = self.p[i]
            self.stage_start_time_lb[i] = cumulative_p_min
            cumulative_p_min += min(stage_p[j] for j in self.j_list)

        n = len(self.j_list)
        self.stage_end_time_ub = {}
        for i in self.i_list:
            candid1 = sum(
                self.p[ip][j] for j in self.j_list for ip in self.i_list[: i + 1]
            )
            # max processing time * (job count + stage count - 1)
            p_max = max(self.p[ip][j] for j in self.j_list for ip in range(0, i + 1))
            candid2 = p_max * (n + i)
            self.stage_end_time_ub[i] = min(candid1, candid2)

        self.D = {
            j: int(round(instance.job_2_duedate_map[j_name]))
            for j, j_name in self.j_2_job_name_map.items()
        }

    # Variables

    def define_variables(self) -> None:
        j_list = self.j_list
        k_list = self.k_list
        i_list = self.i_list

        horizon = self.stage_end_time_ub[i_list[-1]]

        self.y = {
            j: {k: self.new_bool_var(f"y_{j}_{k}") for k in k_list} for j in j_list
        }

        # precompute per-stage coefficients
        coeffs = {i: [self.p[i][j] for j in j_list] for i in i_list}

        # preallocate var references once per column (reduce lookup overhead)
        y_col_refs = {k: [self.y[j][k] for j in j_list] for k in j_list}

        self.P = {
            k: {i: LinearExpr.weighted_sum(y_col_refs[k], coeffs[i]) for i in i_list}
            for k in k_list
        }

        self.C = {
            k: {i: self.new_int_var(0, horizon, f"C_{k}_{i}") for i in i_list}
            for k in k_list
        }
        D_set = {self.D[j] for j in j_list}
        d_min = min(D_set)

        self.var_T = {
            k: self.new_int_var(
                0, self.stage_end_time_ub[self.i_last] - d_min, f"T_{k}"
            )
            for k in k_list
        }

    def define_constraints(self) -> None:
        sub_timer = ElapsedTimer()
        j_list = self.j_list
        k_list = self.k_list
        i_list = self.i_list
        n = len(j_list)
        horizon = self.stage_end_time_ub[i_list[-1]]

        for j in j_list:
            self.add(sum(self.y[j][k] for k in k_list) == 1)
        for k in k_list:
            self.add(sum(self.y[j][k] for j in j_list) == 1)

        logging.info(
            f"Defined assignment constraints; took {sub_timer.elapsed_sec:.3f} sec."
        )
        sub_timer.set_start_time_as_now()

        # k=0, i=0
        self.add(self.C[0][0] == self.P[0][0])

        # k=0, i>=1 : 같은 작업의 다음 공정
        for i in i_list[1:]:
            self.add(self.C[0][i] == self.C[0][i - 1] + self.P[0][i])

        logging.info(
            f"Defined 1st position completion time constraints; took {sub_timer.elapsed_sec:.3f} sec."
        )
        sub_timer.set_start_time_as_now()

        # 3) 나머지 행: Max 제거 → 두 개의 ≥ 제약으로 표현
        for k in range(1, n):
            self.add(self.C[k][0] == self.C[k - 1][0] + self.P[k][0])
            for i in range(1, len(i_list)):
                # C[k][i] = max(C[k][i-1], C[k-1][i]) + P[k][i]
                self.add(self.C[k][i] >= self.C[k][i - 1] + self.P[k][i])
                self.add(self.C[k][i] >= self.C[k - 1][i] + self.P[k][i])

        logging.info(
            f"Defined 2nd-last position completion time constraints; took {sub_timer.elapsed_sec:.3f} sec."
        )
        sub_timer.set_start_time_as_now()

        self.var_d = {}
        for k in k_list:
            expr = LinearExpr.weighted_sum(
                [self.y[j][k] for j in j_list], [self.D[j] for j in j_list]
            )
            self.var_d[k] = expr
        logging.info(
            f"Defined due date variables; took {sub_timer.elapsed_sec:.3f} sec."
        )

    def define_total_tardiness_objective(self) -> None:
        j_list = self.j_list

        for k in j_list:
            self.add(self.var_T[k] >= self.C[k][self.i_last] - self.var_d[k])

        # 목적식: total tardiness
        total_ub = sum(
            max(0, self.stage_end_time_ub[self.i_last] - self.D[j]) for j in self.j_list
        )

        self.obj_var = self.new_int_var(0, total_ub, "sum_Tk")
        self.add(self.obj_var == sum(self.var_T[k] for k in j_list))
        self.minimize(self.obj_var)

    def set_obj_lower_bound(self, bound: int) -> None:
        """Sets a lower bound on the objective variable.

        Args:
            bound (int): The lower bound to set.
        """
        if bound is None:
            return
        if self.obj_var is None:
            raise ValueError("Objective variable is not defined yet.")
        # Convert to float first
        try:
            b = float(bound)
        except (TypeError, ValueError):
            return

        # Ignore NaN
        if math.isnan(b):
            return

        # Ceil and cast to int to avoid float → LinearExpr error
        int_bound = math.ceil(b)
        self.add(self.obj_var >= int_bound)

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
            j_name = self.j_2_job_name_map[k]
            for r in j_list:
                y_kr = self.y[k][r]
                if self.solver.Value(y_kr) == 1:
                    for i in i_list:
                        i_name = self.i_2_stage_name_map[i]
                        C_ri = self.C[r][i]
                        p_ij = self.p[i][k]
                        end_time = self.solver.Value(C_ri)
                        start_time = end_time - p_ij
                        start_time_map[(j_name, i_name)] = start_time
                        end_time_map[(j_name, i_name)] = end_time
                    break
        return start_time_map, end_time_map

    def extract_Tj_map(self) -> dict[str, int]:
        """Extract tardiness values per job after solving.

        Returns:
            dict[str, int]: job_name -> tardiness
        """
        j_list = self.j_list
        Tj_map = {}

        for j in j_list:
            j_name = self.j_2_job_name_map[j]
            T_val = 0
            for r in j_list:
                if self.solver.Value(self.y[j][r]) == 1:
                    # t_jr 계산이 없는 경우 total T_j를 직접 solver에서 읽어도 OK
                    # 하지만 여기서는 define_total_tardiness_objective의 구조에 맞게 계산
                    L_r = self.solver.Value(self.C[r][self.i_list[-1]])
                    d_j = self.D[j]
                    T_val = max(0, L_r - d_j)
                    break
            Tj_map[j_name] = T_val
        return Tj_map

    def create_schedule_from_sequence(self) -> FlowshopSchedule:
        """Creates a FlowshopSchedule from the current solution's job sequence.

        Returns:
            FlowshopSchedule: The constructed schedule.
        """
        j_list = self.j_list

        i_name_list = [self.i_2_stage_name_map[i] for i in self.i_list]
        schedule = FlowshopSchedule.from_stage_name_list(i_name_list)

        # Extract job sequence
        j_sequence = []
        for r in range(len(j_list)):
            for j in j_list:
                if self.solver.Value(self.y[j][r]) == 1:
                    j_sequence.append(j)
                    break

        for j in j_sequence:
            j_name = self.j_2_job_name_map[j]
            i_2_p_map = {
                i_name: self.p[i][j] for i, i_name in self.i_2_stage_name_map.items()
            }
            schedule.dispatch_job_by_stages(
                j_name, i_name_list, i_2_p_map, after_last=True
            )

        return schedule

    # methods to add hints

    def add_hints_from_schedule(self, schedule) -> None:
        """Add solver hints from a feasible FlowshopSchedule."""
        last_i = self.i_list[-1]
        last_i_name = self.i_2_stage_name_map[last_i]
        j_sequence = schedule.get_last_stage_job_list()
        start_time_map = schedule.get_start_time_map()
        sum_Tj = 0

        all_ops_in_schedule = True
        for r, j_name in enumerate(j_sequence):
            j = self.job_name_2_j_map[j_name]
            for i, i_name in self.i_2_stage_name_map.items():
                if (j_name, i_name) in start_time_map:
                    s = start_time_map[j_name, i_name]
                    e = s + self.p[i][j]
                    self.add_hint(self.C[r][i], e)
                else:
                    all_ops_in_schedule = False
            # y[j][r] 힌트
            self.add_hint(self.y[j][r], 1)
            # 나머지 위치는 0으로
            for r2 in self.j_list:
                if r2 != r:
                    self.add_hint(self.y[j][r2], 0)
            # tardiness 힌트
            if (j_name, last_i_name) in start_time_map:
                Tj = max(
                    0,
                    start_time_map[j_name, last_i_name] + self.p[last_i][j] - self.D[j],
                )
                sum_Tj += Tj

        if all_ops_in_schedule:
            self.add_hint(self.obj_var, sum_Tj)

    # Profile fixing method

    def add_indirect_precedence_constraints_by_sequence(
        self, job_sequence: list[str]
    ) -> None:
        """Adds precedence constraints enforcing a specific job order.

        For a given job sequence [A, B, C],
        it enforces position(A) < position(B) < position(C).

        Args:
            job_sequence (list[str]): Ordered list of job names defining precedence.
        """
        if not job_sequence:
            return

        # 1) job별 위치(position) 정수 변수 구성
        n = len(self.j_list)
        self.job_position_var = {}

        for j in self.j_list:
            # pos_j = sum_r r * y[j][r]
            expr = sum(r * self.y[j][r] for r in range(n))
            self.job_position_var[j] = expr

        # 2) 주어진 job_sequence에 따라 precedence 제약 추가
        for j1_name, j2_name in zip(job_sequence[:-1], job_sequence[1:]):
            if (
                j1_name not in self.job_name_2_j_map
                or j2_name not in self.job_name_2_j_map
            ):
                raise ValueError(f"Unknown job name in sequence: {j1_name}, {j2_name}")
            j1 = self.job_name_2_j_map[j1_name]
            j2 = self.job_name_2_j_map[j2_name]

            # pos(j1) + 1 <= pos(j2)
            self.add(self.job_position_var[j1] + 1 <= self.job_position_var[j2])
            logging.info(f"Added precedence: {j1_name} -> {j2_name}")
