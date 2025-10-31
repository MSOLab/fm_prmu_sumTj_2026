from __future__ import annotations

import logging

from ortools.linear_solver.pywraplp import Objective, Solver, VariableExpr
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

    x: dict[str, dict[int, VariableExpr]]
    """job -> time index -> 1 if processed preemptively, 0 otherwise."""

    # Objective

    objective: Objective
    """Objective variable representing total tardiness."""

    def __init__(self, solver_id: str = "SCIP"):
        self.solver: Solver = Solver.CreateSolver(solver_id)

    # Solver의 메서드를 그대로 쓰고 싶으면 위임자 한 줄로 해결
    def __getattr__(self, name):
        # self에 없으면 solver로 위임 (NumVar, Add, Objective, Solve 등)
        return getattr(self.solver, name)

    @classmethod
    def create_solver(cls, solver_id: str = "SCIP") -> SingleMachinePreemptionModel:
        return cls(solver_id)

    @classmethod
    def from_instance(
        cls, instance: FlowshopDuedateParameters, solver_id: str = "SCIP"
    ) -> SingleMachinePreemptionModel:
        result = cls.create_solver(solver_id)
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

        for j in self.calJ:
            logging.info(f"  Job {j}: p={self.p[j]}, r={self.r[j]}, d={self.d[j]}")

        # t = 1..(\max_{j\in calJ}{r_j} + \sum_{j\in calJ}{p_j})
        self.t_max = max(self.r.values()) + sum(self.p.values())
        logging.info(f"  t_max: {self.t_max}")
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
            j: {t: self.NumVar(0, 1, f"x_{j}_{t}") for t in self.calT}
            for j in self.calJ
        }

    def define_constraints(self) -> None:
        # At most one job can be processed at any time.
        # Supply constraints of a transportation problem.
        for t in self.calT:
            self.Add(
                sum(self.x[j][t] for j in self.calJ) <= 1,
                f"capacity_time_{t}",
            )

        # Each job must be processed for its processing time.
        # (Exact) Demand constraints of a transportation problem.
        for j in self.calJ:
            self.Add(
                sum(self.x[j][t] for t in self.calT) == self.p[j],
                f"proc_time_job_{j}",
            )

        # Jobs cannot be processed before their release times.
        for j in self.calJ:
            for t in range(1, self.r[j] + 1):
                self.Add(
                    self.x[j][t] == 0,
                    f"release_time_job_{j}_time_{t}",
                )

    def define_objective(self) -> None:
        # max_tardiness_sum = 0
        # for j in self.calJ:
        #     max_tardiness_sum += self.t_max - self.d[j]
        self.objective = self.Objective()
        for j in self.calJ:
            for t in self.calT:
                self.objective.SetCoefficient(self.x[j][t], self.c[j][t])
        self.objective.SetMinimization()

    # Extraction methods

    def get_obj_value(self) -> float:
        return self.objective.Value()

    def get_job_2_completion_time_map(self) -> dict[str, int]:
        job_2_completion_time_map: dict[str, int] = {}
        for j in self.calJ:
            completion_time = 0
            for t in self.calT:
                x_val = self.x[j][t].solution_value()
                # if x_val > 1e-4:
                #     logging.info(f"x_({j},{t})={x_val}")
                if x_val > 0.5:
                    completion_time = t
            job_2_completion_time_map[j] = completion_time
        return job_2_completion_time_map

    def get_job_completion_sequence(self) -> list[str]:
        job_2_completion_time_map = self.get_job_2_completion_time_map()
        sorted_jobs = sorted(
            job_2_completion_time_map.items(), key=lambda item: item[1]
        )
        return [job for job, _ in sorted_jobs]

    def solve(self) -> None:
        # self.solver.EnableOutput()
        self.status = self.solver.Solve()

    def is_optimal(self) -> bool:
        return self.status == Solver.OPTIMAL

    def is_feasible(self) -> bool:
        return self.status in [Solver.OPTIMAL, Solver.FEASIBLE]
