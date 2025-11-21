from __future__ import annotations

from ortools.math_opt.python import mathopt
from ortools.math_opt.python.variables import Variable
from schore.parameters_examples.shop.flow import FlowshopDuedateParameters


class SingleMachinePreemptionModel:
    name: str

    # Indices & Parameters

    calJ: list[str]
    """List of job IDs."""

    p: dict[str, int]
    """Processing time of each job at the single machine."""

    r: dict[str, int]
    """Release time of each job."""

    d: dict[str, int]
    """Due date of each job."""

    calT: list[int]
    """List of time indices starting from 1."""

    c: dict[str, dict[int, int]]
    """job -> time index -> contribution to tardiness."""

    # Variables

    x: dict[str, dict[int, Variable]]
    """job -> time index -> 1 if processed preemptively, 0 otherwise."""

    def __init__(self):
        self.model: mathopt.Model = mathopt.Model()

    @classmethod
    def from_instance(
        cls, instance: FlowshopDuedateParameters
    ) -> SingleMachinePreemptionModel:
        result = cls()
        result.name = f"{cls.__name__}_{instance.name}"
        result.define_model(instance)
        return result

    def define_model(self, instance: FlowshopDuedateParameters) -> None:
        self.define_parameters(instance)
        self.define_variables()
        self.define_constraints()
        self.define_objective()

    def define_parameters(self, instance: FlowshopDuedateParameters) -> None:
        self.calJ = instance.job_id_list
        self.p = instance.get_job_2_p_map(instance.stage_id_list[-1])
        self.r = instance.get_job_2_p_sum_except_last_stage()
        self.d = instance.job_2_duedate_map

        # for j in self.calJ:
        #     logging.info(f"  Job {j}: p={self.p[j]}, r={self.r[j]}, d={self.d[j]}")

        # t = 1..(\max_{j\in calJ}{r_j} + \sum_{j\in calJ}{p_j})
        self.t_max = max(self.r.values()) + sum(self.p.values())
        # logging.info(f"  t_max: {self.t_max}")
        self.calT = list(range(1, self.t_max + 1))

        # c_jt = 0 if t <= d_j, else \ceil{(t - d_j)/p_j}
        self.c = {
            j: {
                t: max(0, (t - self.d[j] + self.p[j] - 1) // self.p[j])
                for t in self.calT
            }
            for j in self.calJ
        }
        # for j in self.calJ:
        #     for t in self.calT:
        #         if self.c[j][t] > 0:
        #             logging.info(f"c_({j},{t})={self.c[j][t]}")

    def define_variables(self) -> None:
        self.x = {
            j: {
                t: self.model.add_variable(
                    lb=0.0, ub=1.0, name=f"x_{j}_{t}", is_integer=False
                )
                for t in self.calT
            }
            for j in self.calJ
        }

    def define_constraints(self) -> None:
        # At most one job can be processed at any time.
        # Supply constraints of a transportation problem.
        for t in self.calT:
            self.model.add_linear_constraint(
                sum(self.x[j][t] for j in self.calJ) <= 1,
                name=f"capacity_time_{t}",
            )
        # Each job must be processed for its processing time.
        # (Exact) Demand constraints of a transportation problem.
        for j in self.calJ:
            self.model.add_linear_constraint(
                sum(self.x[j][t] for t in self.calT) == self.p[j],
                name=f"proc_time_job_{j}",
            )
        # Jobs cannot be processed before their release times.
        for j in self.calJ:
            for t in range(1, self.r[j] + 1):
                self.model.add_linear_constraint(
                    self.x[j][t] == 0,
                    name=f"release_time_job_{j}_time_{t}",
                )

    def define_objective(self) -> None:
        # Minimize total tardiness
        self.model.minimize(
            sum(self.c[j][t] * self.x[j][t] for j in self.calJ for t in self.calT)
        )

    def solve(self) -> None:
        params = mathopt.SolveParameters(enable_output=True)
        self.result = mathopt.solve(self.model, mathopt.SolverType.HIGHS, params=params)

    # Extraction methods

    def is_optimal(self) -> bool:
        return self.result.termination.reason == mathopt.TerminationReason.OPTIMAL

    def is_feasible(self) -> bool:
        return self.result.termination.reason in {
            mathopt.TerminationReason.OPTIMAL,
            mathopt.TerminationReason.FEASIBLE,
        }

    def get_obj_value(self) -> float:
        return self.result.objective_value()

    def get_variable_value_dict(self) -> dict[str, dict[int, float]]:
        var_values = self.result.variable_values()
        return {j: {t: var_values[self.x[j][t]] for t in self.calT} for j in self.calJ}

    def get_job_2_completion_time_map(self) -> dict[str, int]:
        job_2_completion_time_map: dict[str, int] = {}
        var_values = self.result.variable_values()
        for j in self.calJ:
            completion_time = 0
            for t in self.calT:
                x_jt_value = var_values[self.x[j][t]]
                if x_jt_value > 0.5:
                    completion_time = t
            job_2_completion_time_map[j] = completion_time
        return job_2_completion_time_map

    def get_job_completion_sequence(self) -> list[str]:
        # Sort jobs by their completion times
        job_2_completion_time_map = self.get_job_2_completion_time_map()
        return sorted(self.calJ, key=lambda j: job_2_completion_time_map[j])
