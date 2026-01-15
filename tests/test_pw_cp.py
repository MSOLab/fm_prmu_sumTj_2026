import logging
import math
from unittest.mock import MagicMock, Mock, ANY

import pytest
from mbls.cpsat import CpsatSolverReport, CpsatStatus
from schore.parameters_examples.shop.flow import FlowshopDuedateParameters
from schore.schedule_examples.shop.flow import FlowshopSchedule

from flowshop_tardiness.controller.pw_cp import PwCpConstructor, PwCpContext, PwCpResult
from flowshop_tardiness.cpsat_model_2.position import Params, Vars
from flowshop_tardiness.fm_prmu import PermutationFlowshopScheduleLite

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
        ("S1", "J1"): 5, ("S1", "J2"): 4,
        ("S2", "J1"): 3, ("S2", "J2"): 6
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
    ctx.job_2_stage_2_p_dict = {
        "J1": {"S1": 5, "S2": 3},
        "J2": {"S1": 4, "S2": 6}
    }

    ctx.params = MagicMock(spec=Params)
    ctx.solver = MagicMock()

    # Setup methods
    ctx.get_remaining_time_limit.return_value = 100.0
    
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
        obj_bound_records=[]
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
        m_vars = MagicMock(spec=Vars)
        m_vars.total_tardiness = MagicMock()
        m_vars.sum_latest_completion = MagicMock()
        # Map k to j (index)
        m_vars.pi = {k: k for k in m_params.j_list}
        return m_mdl, m_params, m_vars

    pw_cp.builder = MagicMock()
    pw_cp.builder.build.side_effect = build_side_effect

    # Mock solver values: ctx.solver.Value(vars.pi[k]) should return j (index)
    mock_context.solver.Value.side_effect = lambda x: x # vars.pi[k] is k

    result = pw_cp.run(
        job_sequence=job_sequence,
        added_batch_size=2,
        solver_thread_cnt=1
    )

    assert isinstance(result, PwCpResult)
    assert len(result.schedule.get_last_stage_job_list()) == 2
    assert mock_context.solve_cp_model_2.call_count >= 1

def test_run_two_batches(pw_cp, mock_context):
    """Test running with batch size smaller than job list."""
    job_sequence = ["J1", "J2"]

    def build_side_effect(sub_instance, **kwargs):
        m_mdl = MagicMock()
        m_params = MagicMock(spec=Params)
        m_params.j_list = list(range(len(sub_instance.job_id_list)))
        m_vars = MagicMock(spec=Vars)
        m_vars.total_tardiness = MagicMock()
        m_vars.sum_latest_completion = MagicMock()
        m_vars.pi = {k: k for k in m_params.j_list}
        return m_mdl, m_params, m_vars

    pw_cp.builder = MagicMock()
    pw_cp.builder.build.side_effect = build_side_effect
    mock_context.solver.Value.side_effect = lambda x: x

    result = pw_cp.run(
        job_sequence=job_sequence,
        added_batch_size=1, 
        solver_thread_cnt=1
    )

    assert isinstance(result, PwCpResult)
    assert len(result.schedule.get_last_stage_job_list()) == 2
    assert mock_context.solve_cp_model_2.call_count >= 2

def test_log_snapshot(pw_cp, mock_context):
    """Test _log_snapshot functionality via run state."""
    job_sequence = ["J1"]
    
    pw_cp.builder = MagicMock()
    pw_cp.builder.build.return_value = (MagicMock(), MagicMock(), MagicMock())
    # satisfy PwCpConstructor._solve_cp_model_lexico_for_batch requirements if called
    
    # We just want to see if run() completes and log_snapshot was called internally
    result = pw_cp.run(job_sequence, added_batch_size=1)
    assert result.last_obj_value >= 0

def test_solver_infeasible(pw_cp, mock_context):
    """Test handling when solver returns infeasible."""
    job_sequence = ["J1"]

    pw_cp.builder = MagicMock()
    m_mdl, m_params, m_vars = MagicMock(), MagicMock(), MagicMock()
    m_params.j_list = [0]
    m_vars.pi = {0: 0}
    pw_cp.builder.build.return_value = (m_mdl, m_params, m_vars)

    # Return infeasible report
    mock_context.solve_cp_model_2.return_value = CpsatSolverReport(
        elapsed_time=1.0,
        obj_value=None,
        obj_bound=None,
        status=CpsatStatus.INFEASIBLE,
        obj_value_records=[],
        obj_bound_records=[]
    )

    result = pw_cp.run(job_sequence, added_batch_size=1)
    assert isinstance(result, PwCpResult)
    assert len(result.schedule.get_last_stage_job_list()) == 1