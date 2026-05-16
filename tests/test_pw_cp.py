from unittest.mock import MagicMock

import pytest
from itertools import permutations
from mbls.cpsat import CpsatSolverReport, CpsatStatus
from schore.parameters_examples.shop.flow import FlowshopDuedateParameters

from flowshop_tardiness.controller.pw_cp import PwCpConstructor, PwCpContext, PwCpResult
from flowshop_tardiness.cpsat_model_2.indirect_prec import IndirectPrecVars
from flowshop_tardiness.cpsat_model_2.params import Params


@pytest.fixture
def mock_context():
    ctx = MagicMock(spec=PwCpContext)

    # Setup data attributes
    ctx.instance = MagicMock(spec=FlowshopDuedateParameters)
    ctx.instance.job_2_duedate_map = {"J1": 1, "J2": 1}
    ctx.instance.job_id_list = ["J1", "J2"]
    ctx.instance.stage_id_list = ["S1", "S2"]
    ctx.instance.name = "TestInstance"

    # Mock p_manager
    p_manager = MagicMock()
    # returns { (stage, job): value }
    p_manager.stage_job_2_value_map.return_value = {
        ("S1", "J1"): 5,
        ("S1", "J2"): 4,
        ("S2", "J1"): 3,
        ("S2", "J2"): 6,
    }
    ctx.instance.p_manager = p_manager

    # Mock get_subinstance
    def get_subinstance_mock(job_list):
        sub = MagicMock(spec=FlowshopDuedateParameters)
        sub.job_id_list = list(job_list)
        sub.stage_id_list = ctx.instance.stage_id_list
        sub.job_2_duedate_map = {j: ctx.instance.job_2_duedate_map[j] for j in job_list}
        sub.p_manager = ctx.instance.p_manager
        return sub

    ctx.instance.get_subinstance.side_effect = get_subinstance_mock

    ctx.stage_ids = ("S1", "S2")
    ctx.job_2_stage_2_p_dict = {"J1": {"S1": 5, "S2": 3}, "J2": {"S1": 4, "S2": 6}}

    ctx.params = MagicMock(spec=Params)
    ctx.solver = MagicMock()

    # Setup methods
    ctx.get_remaining_time_limit.return_value = 10.0

    # Use real logic for get_obj_value and check_feasibility
    def get_obj_value_real(schedule):
        return schedule.get_total_tardiness(ctx.instance.job_2_duedate_map)

    ctx.get_obj_value.side_effect = get_obj_value_real

    def check_feasibility_real(schedule):
        return get_obj_value_real(schedule)

    ctx.check_feasibility.side_effect = check_feasibility_real

    ctx.solution_manager = MagicMock()
    ctx.solution_manager.best_obj_bound = None

    # Default solver report (Success)
    ctx.solve_cp_model_2.return_value = CpsatSolverReport(
        elapsed_time=1.0,
        obj_value=0,
        obj_bound=0,
        status=CpsatStatus.OPTIMAL,
        obj_value_records=[],
        obj_bound_records=[],
    )

    return ctx


@pytest.fixture
def pw_cp(mock_context):
    return PwCpConstructor(mock_context)


def test_initialization(pw_cp, mock_context):
    assert pw_cp.ctx == mock_context
    assert pw_cp._st is None


def test_run_basic_flow(pw_cp, mock_context):
    """Test a simple run with a small batch of jobs."""
    # Use jobs that are NOT J1, J2 to avoid conflict with mock_context defaults if any,
    # but run() clears state so it should be fine.
    job_sequence = ["J1", "J2"]

    # Configure mock for building model
    def build_side_effect(sub_instance, **kwargs):
        m_mdl = MagicMock()
        m_params = MagicMock(spec=Params)
        m_params.j_list = list(range(len(sub_instance.job_id_list)))
        m_vars = MagicMock(spec=IndirectPrecVars)
        m_vars.total_tardiness = MagicMock()
        m_vars.sum_latest_completion = MagicMock()
        m_vars.prec = {}
        for j1, j2 in permutations(m_params.j_list, 2):
            m_vars.prec[j1, j2] = 1
            m_vars.prec[j2, j1] = 0
        return m_mdl, m_params, m_vars

    pw_cp.builder = MagicMock()
    pw_cp.builder.build.side_effect = build_side_effect
    
    # Mock solver values and sequence reconstruction
    mock_context.solver.Value.side_effect = lambda x: x
    mock_context.from_job_prec_get_sequence.side_effect = lambda params, prec: params.j_list

    result = pw_cp.run(
        job_sequence=job_sequence, added_batch_size=2, solver_thread_cnt=1
    )

    assert isinstance(result, PwCpResult)
    # With batch=2, all 2 jobs are added, CP runs once.
    assert len(result.schedule.get_last_stage_job_list()) == 2
    assert mock_context.solve_cp_model_2.call_count >= 1


def test_run_two_batches(pw_cp, mock_context):
    """Test running with batch size smaller than job list."""
    job_sequence = ["J1", "J2"]

    def build_side_effect(sub_instance, **kwargs):
        m_mdl = MagicMock()
        m_params = MagicMock(spec=Params)
        m_params.j_list = list(range(len(sub_instance.job_id_list)))
        m_vars = MagicMock(spec=IndirectPrecVars)
        m_vars.total_tardiness = MagicMock()
        m_vars.sum_latest_completion = MagicMock()
        m_vars.prec = {}
        for j1, j2 in permutations(m_params.j_list, 2):
            m_vars.prec[j1, j2] = 1
            m_vars.prec[j2, j1] = 0
        return m_mdl, m_params, m_vars

    pw_cp.builder = MagicMock()
    pw_cp.builder.build.side_effect = build_side_effect
    
    # Mock solver values and sequence reconstruction
    mock_context.solver.Value.side_effect = lambda x: x
    mock_context.from_job_prec_get_sequence.side_effect = lambda params, prec: params.j_list

    result = pw_cp.run(
        job_sequence=job_sequence, added_batch_size=1, solver_thread_cnt=1
    )

    assert isinstance(result, PwCpResult)
    assert len(result.schedule.get_last_stage_job_list()) == 2
    assert mock_context.solve_cp_model_2.call_count >= 2


def test_log_snapshot(pw_cp, mock_context):
    """Test that _log_snapshot actually records objective values in sub_obj_store."""

    # Needs at least 2 jobs to enter optimization loop
    job_sequence = ["J1", "J2"]

    # Correctly mock builder to return iterable j_list and valid variables
    def build_side_effect(sub_instance, **kwargs):
        m_mdl = MagicMock()
        m_params = MagicMock(spec=Params)
        m_params.j_list = list(range(len(sub_instance.job_id_list)))
        m_vars = MagicMock(spec=IndirectPrecVars)
        m_vars.total_tardiness = MagicMock()
        m_vars.sum_latest_completion = MagicMock()
        m_vars.pi = {0: 0}
        m_vars.prec = {}
        for j1, j2 in permutations(m_params.j_list, 2):
            m_vars.prec[j1, j2] = 1
            m_vars.prec[j2, j1] = 0

        return m_mdl, m_params, m_vars

    pw_cp.builder = MagicMock()
    pw_cp.builder.build.side_effect = build_side_effect

    # Mock solver values and sequence reconstruction
    mock_context.solver.Value.side_effect = lambda x: x
    mock_context.from_job_prec_get_sequence.side_effect = lambda params, prec: params.j_list

    result = pw_cp.run(job_sequence, added_batch_size=1)

    # Verify that log entries exist in the store
    assert len(result.sub_obj_store.obj_value_series) > 0

    # Verify valid objective value
    assert result.sub_obj_store.get_last_obj_value() >= 0


def test_solver_infeasible(pw_cp, mock_context):
    """Test handling when solver returns infeasible."""
    # Needs at least 2 jobs to enter optimization loop
    job_sequence = ["J1", "J2"]

    # Configure mock
    def build_side_effect(sub_instance, **kwargs):
        m_mdl = MagicMock()
        m_params = MagicMock(spec=Params)
        m_params.j_list = list(range(len(sub_instance.job_id_list)))
        m_vars = MagicMock(spec=IndirectPrecVars)
        m_vars.pi = {0: 0}
        m_vars.total_tardiness = MagicMock()
        m_vars.sum_latest_completion = MagicMock()
        m_vars.prec = {}
        for j1, j2 in permutations(m_params.j_list, 2):
            m_vars.prec[j1, j2] = 1
            m_vars.prec[j2, j1] = 0
        return m_mdl, m_params, m_vars

    pw_cp.builder = MagicMock()
    pw_cp.builder.build.side_effect = build_side_effect
    
    # Mock solver values and sequence reconstruction
    mock_context.solver.Value.side_effect = lambda x: x
    mock_context.from_job_prec_get_sequence.side_effect = lambda params, prec: params.j_list

    # Return infeasible report
    mock_context.solve_cp_model_2.return_value = CpsatSolverReport(
        elapsed_time=1.0,
        obj_value=None,
        obj_bound=None,
        status=CpsatStatus.INFEASIBLE,
        obj_value_records=[],
        obj_bound_records=[],
    )

    result = pw_cp.run(job_sequence, added_batch_size=1)
    assert isinstance(result, PwCpResult)
    # Even if infeasible, it should return a schedule with all jobs dispatched (fallback to base)
    assert len(result.schedule.get_last_stage_job_list()) == 2
