from __future__ import annotations

import logging
from dataclasses import dataclass

from mbls.cpsat import CustomCpModel
from ortools.sat.python.cp_model import CpModel, IntVar
from routix import ElapsedTimer
from schore.parameters_examples.shop.flow import FlowshopDuedateParameters


@dataclass(frozen=True)
class Params:
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

    P: dict[tuple[int, int], int]
    """$P_{ij}$: processing time at stage i for job j (P[i, j])"""

    stage_start_time_lb: dict[int, int]
    """i -> lower bound on the start time of the stage."""

    stage_end_time_ub: dict[int, int]
    """i -> upper bound on the makespan of the stage."""

    D: dict[int, int]
    """$D_j$: due date of job j"""


@dataclass
class Vars:
    op_start: dict[tuple[int, int], IntVar]
    """
    (i, k) -> start time of a k-th job in stage i
    """
    op_lth: dict[tuple[int, int], IntVar]
    """
    (i, k) -> processing time of a k-th job in stage i
    """
    op_end: dict[tuple[int, int], IntVar]
    """
    (i, k) -> end time of a k-th job in stage i
    """
    pi: dict[int, IntVar]
    """$pi_k$: job index (j) at position k"""

    d: dict[int, IntVar]
    """$d_k$: due date of job at k-th position"""

    T: dict[int, IntVar]
    """$T_k$: tardiness of job at k-th position"""

    total_tardiness: IntVar | None = None
    """Total tardiness variable"""

    sum_latest_completion: IntVar | None = None
    """Sum of latest completion times variable"""

    # hint 대상만 반환(중요)
    def decision_vars(self):
        yield from self.op_start.values()
        yield from self.op_lth.values()
        yield from self.op_end.values()
        yield from self.pi.values()
        yield from self.d.values()
        yield from self.T.values()


class BaseModelBuilder:
    def build(
        self,
        instance: FlowshopDuedateParameters,
        stage_2_est_map: dict[str, int] | None = None,
        stage_2_lct_map: dict[str, int] | None = None,
        sumTj_offset: int | None = None,
    ) -> tuple[CustomCpModel, Params, Vars]:
        mdl = CustomCpModel()
        params: Params = self._make_params(instance, stage_2_est_map=stage_2_est_map, stage_2_lct_map=stage_2_lct_map)

        vars = self._make_vars(mdl, instance, params)
        self._add_structural_constraints(mdl, instance, params, vars)
        self._define_objectives(
            mdl, instance, params, vars, sumTj_offset=sumTj_offset
        )  # only defines vars + equalities

        return mdl, params, vars

    def _make_params(
        self,
        instance: FlowshopDuedateParameters,
        stage_2_est_map: dict[str, int] | None = None,
        stage_2_lct_map: dict[str, int] | None = None,
    ) -> Params:
        j_2_job_name_map = {j: name for j, name in enumerate(instance.job_id_list)}
        job_name_2_j_map = {name: j for j, name in j_2_job_name_map.items()}
        j_list = list(range(len(instance.job_id_list)))
        n = len(j_list)
        j_first = 0
        j_last = n - 1

        i_2_stage_name_map = {i: name for i, name in enumerate(instance.stage_id_list)}
        i_list = list(range(len(instance.stage_id_list)))

        P_dict = instance.p_manager.stage_job_2_value_map(
            instance.stage_id_list, instance.job_id_list
        )
        P = {
            (i, j): int(round(P_dict[i_name, j_name]))
            for i, i_name in i_2_stage_name_map.items()
            for j, j_name in j_2_job_name_map.items()
        }

        stage_start_time_lb = {}
        for i in i_list:
            stage_est = (
                stage_2_est_map.get(i_2_stage_name_map[i], 0) if stage_2_est_map else 0
            )
            if i > 0:
                candid1 = stage_start_time_lb[i - 1] + min(P[i - 1, j] for j in j_list)
                if stage_est < candid1:
                    stage_est = candid1
            stage_start_time_lb[i] = stage_est
            # logging.info(f"Stage {i} start time LB: {self.stage_start_time_lb[i]}")

        stage_end_time_ub = {}
        for i in i_list:
            stage_end_time_ub[i] = (
                sum(P[ip, j] for j in j_list for ip in i_list[: i + 1])
                + stage_start_time_lb[i]
            )
            if stage_2_lct_map:
                lct = stage_2_lct_map.get(i_2_stage_name_map[i], None)
                if lct is not None and stage_end_time_ub[i] > lct:
                    stage_end_time_ub[i] = lct
            # logging.info(f"Stage {i} end time UB: {self.stage_end_time_ub[i]}")
        D = {
            j: int(round(instance.job_2_duedate_map[j_name]))
            for j, j_name in j_2_job_name_map.items()
        }
        return Params(
            j_2_job_name_map=j_2_job_name_map,
            job_name_2_j_map=job_name_2_j_map,
            j_list=j_list,
            j_first=j_first,
            j_last=j_last,
            i_2_stage_name_map=i_2_stage_name_map,
            i_list=i_list,
            P=P,
            stage_start_time_lb=stage_start_time_lb,
            stage_end_time_ub=stage_end_time_ub,
            D=D,
        )

    def _make_vars(
        self,
        mdl: CpModel,
        instance: FlowshopDuedateParameters,
        params: Params,
    ) -> Vars:
        j_list = params.j_list
        i_list = params.i_list

        # Interval variables
        var_op_start = {}
        var_op_lth = {}
        var_op_end = {}

        for i in i_list:
            stage_start_time_lb = params.stage_start_time_lb[i]
            stage_end_time_ub = params.stage_end_time_ub[i]
            P_set = {params.P[i, j] for j in j_list}
            P_min_i = min(P_set)
            P_max_i = max(P_set)
            for k in j_list:
                suffix = f"{i}_{k}"
                var_op_start[i, k] = mdl.new_int_var(
                    stage_start_time_lb,
                    stage_end_time_ub - P_min_i,
                    f"start_{suffix}",
                )
                var_op_lth[i, k] = mdl.new_int_var(P_min_i, P_max_i, f"lth_{suffix}")
                var_op_end[i, k] = mdl.new_int_var(
                    stage_start_time_lb + P_min_i,
                    stage_end_time_ub,
                    f"end_{suffix}",
                )

        # Position variables
        var_pi = {
            k: mdl.new_int_var(params.j_first, params.j_last, f"pi_{k}") for k in j_list
        }

        D_set = {params.D[j] for j in j_list}
        D_min = min(D_set)
        D_max = max(D_set)
        var_d = {k: mdl.new_int_var(D_min, D_max, f"d_{k}") for k in j_list}

        # Tardiness of k-th job
        last_i = i_list[-1]
        var_T = {
            k: mdl.new_int_var(
                0, max(params.stage_end_time_ub[last_i] - D_min, 0), f"T_{k}"
            )
            for k in j_list
        }

        return Vars(
            op_start=var_op_start,
            op_lth=var_op_lth,
            op_end=var_op_end,
            pi=var_pi,
            d=var_d,
            T=var_T,
        )

    def _add_structural_constraints(
        self,
        mdl: CpModel,
        instance: FlowshopDuedateParameters,
        params: Params,
        vars: Vars,
    ) -> None:
        # Alias for readability
        j_list = params.j_list
        i_list = params.i_list

        timer = ElapsedTimer()

        # All-different constraint on sequence variables
        mdl.add_all_different([vars.pi[k] for k in j_list])

        logging.info(f"  All-different constr. took {timer.elapsed_sec:.3f} sec.")
        timer.set_start_time_as_now()

        # Processing time of each operation
        # lth_{i,k} = sum_j{P_{ij} * (pi_k == j_index)} \forall i\in I, k\in K
        # This uses the element constraint.
        for i in i_list:
            P_vals_i = [params.P[i, j] for j in j_list]
            for k in j_list:
                mdl.add_element(vars.pi[k], P_vals_i, vars.op_lth[i, k])

        logging.info(f"  Processing time constr. took {timer.elapsed_sec:.3f} sec.")
        timer.set_start_time_as_now()

        # Due date of each job
        # d_k = sum_j{D_j * (pi_k == j_index)}
        D_vals = [params.D[j] for j in j_list]
        for k in j_list:
            mdl.add_element(vars.pi[k], D_vals, vars.d[k])

        logging.info(f"  Due date constr. took {timer.elapsed_sec:.3f} sec.")
        timer.set_start_time_as_now()

        # Precedence between consecutive stages for each job
        for i, next_i in zip(i_list[:-1], i_list[1:]):
            for k in j_list:
                mdl.add(vars.op_end[i, k] <= vars.op_start[next_i, k])

        logging.info(
            f"  Precedence (inter-stage) constr. took {timer.elapsed_sec:.3f} sec."
        )
        timer.set_start_time_as_now()

        # Precedence between operations in the same stage
        for i in i_list:
            for k in j_list:
                mdl.add(vars.op_end[i, k] == vars.op_start[i, k] + vars.op_lth[i, k])
                if k != params.j_first:
                    mdl.add(vars.op_end[i, k - 1] <= vars.op_start[i, k])

        logging.info(
            f"  Precedence (intra-stage) constr. took {timer.elapsed_sec:.3f} sec."
        )

    def _define_objectives(
        self,
        mdl: CpModel,
        instance: FlowshopDuedateParameters,
        params: Params,
        vars: Vars,
        sumTj_offset: int | None = None,
    ) -> None:
        j_list = params.j_list
        last_i = params.i_list[-1]

        for k in j_list:
            mdl.add_max_equality(vars.T[k], [vars.op_end[last_i, k] - vars.d[k], 0])
        sumTj_ub = sum(
            max(0, params.stage_end_time_ub[last_i] - params.D[j]) for j in j_list
        )
        sumTj_lb = 0
        if sumTj_offset is not None:
            sumTj_ub += sumTj_offset
            sumTj_lb += sumTj_offset

        sumTj = mdl.new_int_var(sumTj_lb, sumTj_ub, "total_tardiness")
        # T == sum(tardiness_j)
        mdl.add(sumTj == sum(vars.T[k] for k in j_list) + (sumTj_offset or 0))
        vars.total_tardiness = sumTj

        S_ub = sum(params.stage_end_time_ub[i] for i in params.i_list)

        sumCi = mdl.new_int_var(0, S_ub, "sum_latest_completion")
        mdl.add(sumCi == sum(vars.op_end[i, params.j_last] for i in params.i_list))
        vars.sum_latest_completion = sumCi
