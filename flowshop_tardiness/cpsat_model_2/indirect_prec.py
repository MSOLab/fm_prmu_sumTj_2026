from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

from mbls.cpsat import CustomCpModel
from ortools.sat.python.cp_model import CpModel, IntervalVar, IntVar
from schore.parameters_examples.shop.flow import FlowshopDuedateParameters

from .params import Params


@dataclass
class IndirectPrecVars:
    op_start: dict[tuple[int, int], IntVar]
    """(i, j) -> start time of job j in stage i"""

    op_end: dict[tuple[int, int], IntVar]
    """(i, j) -> end time of job j in stage i"""

    op_intvl: dict[tuple[int, int], IntervalVar]
    """(i, j) -> interval variable of job j in stage i"""

    prec: dict[tuple[int, int], IntVar]
    """
    $prec_{j1,j2}$: Indirect precedence variables;
    1 if job j1 (not necessarily immediately) before job j2
    """

    T: dict[int, IntVar]
    """$T_k$: tardiness of job at k-th position"""

    C: dict[int, IntVar]
    """$C_i$: latest completion time of stage i"""

    total_tardiness: IntVar | None = None
    """Total tardiness variable"""

    sum_latest_completion: IntVar | None = None
    """Sum of latest completion times variable"""

    # hint 대상만 반환(중요)
    def decision_vars(self):
        yield from self.op_start.values()
        yield from self.op_end.values()
        yield from self.prec.values()
        yield from self.T.values()


class BaseModelBuilder:
    def build(
        self,
        instance: FlowshopDuedateParameters,
        stage_2_est_map: dict[str, int] | None = None,
        stage_2_lct_map: dict[str, int] | None = None,
        sumTj_offset: int | None = None,
        profile_fixed_job_list: list[str] | None = None,
    ) -> tuple[CustomCpModel, Params, IndirectPrecVars]:
        mdl = CustomCpModel()
        params: Params = self._make_params(
            instance, stage_2_est_map=stage_2_est_map, stage_2_lct_map=stage_2_lct_map
        )

        vars: IndirectPrecVars = self._make_vars(mdl, instance, params)
        self._add_structural_constraints(mdl, instance, params, vars)
        self._define_objectives(
            mdl, instance, params, vars, sumTj_offset=sumTj_offset
        )  # only defines vars + equalities
        if profile_fixed_job_list is not None:
            # Add profile-fixing constraints
            self._add_profile_fixing_constraints(
                mdl, instance, params, vars, profile_fixed_job_list
            )

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
    ) -> IndirectPrecVars:
        i_list = params.i_list
        j_list = params.j_list

        # Interval variables
        var_op_start = {}
        var_op_end = {}
        var_op_intvl = {}

        for i in i_list:
            stage_start_time_lb = params.stage_start_time_lb[i]
            stage_end_time_ub = params.stage_end_time_ub[i]
            P_set = {params.P[i, j] for j in j_list}
            P_min_i = min(P_set)
            for j in j_list:
                suffix = f"{i}_{j}"
                var_op_start[i, j] = mdl.new_int_var(
                    stage_start_time_lb,
                    stage_end_time_ub - P_min_i,
                    f"start_{suffix}",
                )
                var_op_end[i, j] = mdl.new_int_var(
                    stage_start_time_lb + P_min_i,
                    stage_end_time_ub,
                    f"end_{suffix}",
                )
                var_op_intvl[i, j] = mdl.new_interval_var(
                    var_op_start[i, j],
                    params.P[i, j],
                    var_op_end[i, j],
                    f"intvl_{suffix}",
                )

        # Indirect precedence variables
        var_prec = {}
        for j1_idx, j1 in enumerate(j_list):
            for j2 in j_list[j1_idx + 1 :]:
                var_prec[j1, j2] = mdl.new_bool_var(f"prec_ind_{j1}_{j2}")
                var_prec[j2, j1] = mdl.new_bool_var(f"prec_ind_{j2}_{j1}")
                mdl.add(var_prec[j1, j2] + var_prec[j2, j1] == 1)

        D_set = {params.D[j] for j in j_list}
        D_min = min(D_set)

        # Tardiness of k-th job
        last_i = i_list[-1]
        var_T = {
            k: mdl.new_int_var(
                0, max(params.stage_end_time_ub[last_i] - D_min, 0), f"T_{k}"
            )
            for k in j_list
        }

        # Latest completion time of each stage
        var_Ci = {}
        for i in i_list:
            var_Ci[i] = mdl.new_int_var(0, params.stage_end_time_ub[i], f"C_{i}")

        return IndirectPrecVars(
            op_start=var_op_start,
            op_end=var_op_end,
            op_intvl=var_op_intvl,
            prec=var_prec,
            T=var_T,
            C=var_Ci,
        )

    def _add_structural_constraints(
        self,
        mdl: CpModel,
        instance: FlowshopDuedateParameters,
        params: Params,
        vars: IndirectPrecVars,
    ) -> None:
        # Alias for readability
        i_list = params.i_list
        j_list = params.j_list

        # Precedence between consecutive stages for each job
        consecutive_stage_pairs = list(zip(i_list[:-1], i_list[1:]))
        for j in j_list:
            for i, next_i in consecutive_stage_pairs:
                mdl.add(
                    vars.op_start[i, j] + params.P[i, j] <= vars.op_start[next_i, j]
                )

        # Link precedence and time
        for i in i_list:
            for j1, j2 in permutations(j_list, 2):
                mdl.add(vars.op_start[i, j2] >= vars.op_end[i, j1]).only_enforce_if(
                    vars.prec[j1, j2]
                )

    def _define_objectives(
        self,
        mdl: CpModel,
        instance: FlowshopDuedateParameters,
        params: Params,
        vars: IndirectPrecVars,
        sumTj_offset: int | None = None,
    ) -> None:
        j_list = params.j_list
        i_list = params.i_list
        last_i = i_list[-1]

        # Tardiness of each job
        for j in j_list:
            mdl.add(vars.T[j] >= vars.op_end[last_i, j] - params.D[j])

        # Total tardiness
        sumTj_ub = sum(
            max(0, params.stage_end_time_ub[last_i] - params.D[j]) for j in j_list
        )
        sumTj_lb = 0
        if sumTj_offset is not None:
            sumTj_ub += sumTj_offset
            sumTj_lb += sumTj_offset
        sumTj = mdl.new_int_var(sumTj_lb, sumTj_ub, "total_tardiness")
        mdl.add(sumTj == sum(vars.T[j] for j in j_list) + (sumTj_offset or 0))
        vars.total_tardiness = sumTj

        # Latest completion time of each stage
        for i in i_list:
            for j in j_list:
                mdl.add(vars.C[i] >= vars.op_end[i, j])

        # Sum of latest completion times of all stages
        S_ub = sum(params.stage_end_time_ub[i] for i in i_list)
        sumCi = mdl.new_int_var(0, S_ub, "sum_latest_completion")
        mdl.add(sumCi == sum(vars.C[i] for i in i_list))
        vars.sum_latest_completion = sumCi

    def _add_profile_fixing_constraints(
        self,
        mdl: CpModel,
        instance: FlowshopDuedateParameters,
        params: Params,
        vars: IndirectPrecVars,
        profile_fixed_job_list: list[str],
    ) -> None:
        """
        Add profile-fixing constraints to the model.
        Jobs in profile_fixed_job_list should maintain their relative order in the solution.

        Example:
            If all job list = ['JobA', 'JobB', 'JobC'] and profile_fixed_job_list = ['JobA', 'JobC'],
            possible sequences are:
                - ['JobB', 'JobA', 'JobC']
                - ['JobA', 'JobB', 'JobC']
                - ['JobA', 'JobC', 'JobB']
            but not:
                - ['JobC', 'JobA', 'JobB']

        Args:
            mdl (CpModel): model to which constraints are added
            instance (FlowshopDuedateParameters): instance parameters
            params (Params): parameters of the model
            vars (Vars): variables of the model
            profile_fixed_job_list (list[str]): List of job names to fix the profile
        """
        if len(profile_fixed_job_list) < 2:
            # No need to add constraints if less than 2 jobs are specified
            return

        job_name_2_j_map = params.job_name_2_j_map

        # Convert job names to job indices
        fixed_job_indices = [job_name_2_j_map[name] for name in profile_fixed_job_list]

        # Add constraints to maintain relative order
        for j, jp in zip(fixed_job_indices[:-1], fixed_job_indices[1:]):
            mdl.add(vars.prec[j, jp] == 1)
            mdl.add(vars.prec[jp, j] == 0)
