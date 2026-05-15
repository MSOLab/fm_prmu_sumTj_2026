"""Verify that solver libs do not get pulled into the dashboards/CLI imports.

The dashboards subpackage and ``scripts/aggregate_analysis.py`` must remain
pure pandas/numpy/plotly/xlsxwriter so they run on machines without the
solver stack installed (CP-SAT via ``mbls``, CPLEX via ``docplex`` /
``cplex``).

Each import is exercised in a **fresh subprocess** so that solver modules
already loaded by sibling tests in the same pytest session do not mask
real regressions: an in-process ``sys.modules`` snapshot would record
those pre-loaded modules in ``pre`` and silently allow new transitive
imports to slip through.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SOLVER_MODULE_PREFIXES: tuple[str, ...] = ("mbls", "docplex", "cplex")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _list_solver_modules_after(import_stmt: str) -> list[str]:
    """Run ``import_stmt`` in a fresh interpreter; return the solver
    modules that ended up in ``sys.modules``."""
    code = textwrap.dedent(
        f"""
        import sys
        {import_stmt}
        prefixes = {SOLVER_MODULE_PREFIXES!r}
        loaded = sorted(
            m for m in sys.modules
            if any(m == p or m.startswith(p + ".") for p in prefixes)
        )
        for m in loaded:
            print(m)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


@pytest.mark.parametrize(
    "import_stmt",
    [
        (
            "from flowshop_tardiness.report.dashboards import ("
            " apply_timelimit_trim, build_rpdf_comparison_df,"
            " write_post_run_dashboard_artifacts )"
        ),
        (
            "from flowshop_tardiness.report.dashboards.multi_scenario_report"
            " import ("
            " aggregate_scenario_summaries, build_dashboard_df,"
            " write_multi_scenario_excel_report )"
        ),
        "from scripts.aggregate_analysis import main",
    ],
    ids=[
        "dashboards_subpackage",
        "multi_scenario_report_module",
        "aggregate_analysis_cli",
    ],
)
def test_solver_libs_not_imported(import_stmt: str) -> None:
    loaded = _list_solver_modules_after(import_stmt)
    assert not loaded, (
        f"Import should be solver-free but pulled in: {loaded}\n"
        f"  stmt: {import_stmt}"
    )


def test_solver_detection_actually_works() -> None:
    """Negative control: when something solver-y *is* imported, the helper
    must detect it. Without this, a broken subprocess setup (e.g. swallowed
    output, wrong cwd) would make every check above trivially green."""
    loaded = _list_solver_modules_after(
        "from flowshop_tardiness.report import FsCpsatSolverReport"
    )
    assert any(m.startswith("mbls") for m in loaded), (
        f"Expected mbls.* in sys.modules after importing FsCpsatSolverReport,"
        f" but got: {loaded}"
    )
