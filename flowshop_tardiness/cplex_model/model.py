from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from docplex.mp.linear import LinearExpr
from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution


@dataclass(frozen=True)
class TBB2018Data:
    """Data for permutation flow shop total tardiness MILP."""

    p: list[list[float]]  # p[i][j] processing time at stage i for job j
    d: list[float]  # d[j] due date for job j

    @property
    def m(self) -> int:
        return len(self.p)

    @property
    def n(self) -> int:
        if not self.p:
            return 0
        return len(self.p[0])

    def validate(self) -> None:
        if self.m <= 0:
            raise ValueError("p must have at least 1 stage (m>=1).")
        n0 = len(self.p[0])
        if n0 <= 0:
            raise ValueError("p must have at least 1 job (n>=1).")
        for i, row in enumerate(self.p):
            if len(row) != n0:
                raise ValueError(
                    f"p row length mismatch at stage {i}: {len(row)} != {n0}"
                )
            if any(v < 0 for v in row):
                raise ValueError(f"p contains negative processing time at stage {i}.")
        if len(self.d) != n0:
            raise ValueError(f"d length mismatch: len(d)={len(self.d)} != n={n0}")
        if any(v < 0 for v in self.d):
            raise ValueError("d contains negative due date.")


@dataclass
class TBB2018Vars:
    # x[j,k] binary: job j assigned to position k
    x: dict[tuple[int, int], Any]
    # C[i,k] >= 0: completion time of position k on stage i
    C: dict[tuple[int, int], Any]
    # T[k] >= 0: tardiness of position k
    T: dict[int, Any]


class TBB2018MilpModelBuilder:
    """
    Builds the MILP model (1)-(9) in the uploaded LaTeX:
      min sum_k T_k
      s.t. assignment constraints on x_{j,k}
           completion time recurrences C_{i,k}
           tardiness definition T_k
    """

    def __init__(self, data: TBB2018Data, model_name: str = "tbb_2018_milp"):
        data.validate()
        self.data = data
        self.model_name = model_name

    def build(
        self,
        time_limit_s: float | None = None,
        threads: int | None = None,
        mip_gap: float | None = None,
        incumbent_perm: Sequence[int] | None = None,  # list of jobs in positions 0..n-1
    ) -> tuple[Model, TBB2018Vars]:
        """
        Returns (model, vars).

        incumbent_perm (optional):
          a permutation of jobs (0..n-1) representing positions k=0..n-1.
          If provided, a MIP start is added: x[perm[k], k] = 1.
        """
        m, n = self.data.m, self.data.n
        p, d = self.data.p, self.data.d

        mdl = Model(self.model_name)

        # ---- decision variables ----
        # x_{j,k} binary
        x = {
            (j, k): mdl.binary_var(name=f"x_{j}_{k}")
            for j in range(n)
            for k in range(n)
        }

        # C_{i,k} continuous >= 0
        C = {
            (i, k): mdl.continuous_var(lb=0, name=f"C_{i}_{k}")
            for i in range(m)
            for k in range(n)
        }

        # T_k continuous >= 0
        T = {k: mdl.continuous_var(lb=0, name=f"T_{k}") for k in range(n)}

        vars_ = TBB2018Vars(x=x, C=C, T=T)

        # ---- objective (1): min sum_k T_k ----
        mdl.minimize(mdl.sum(T[k] for k in range(n)))

        # ---- constraints ----
        # (2) sum_k x_{j,k} = 1  for all j
        for j in range(n):
            mdl.add_constraint(
                mdl.sum(x[j, k] for k in range(n)) == 1, ctname=f"assign_job_{j}"
            )

        # (3) sum_j x_{j,k} = 1  for all k
        for k in range(n):
            mdl.add_constraint(
                mdl.sum(x[j, k] for j in range(n)) == 1, ctname=f"assign_pos_{k}"
            )

        # Helper: expression sum_j p_{i,j} * x_{j,k}
        def proc_expr(i: int, k: int) -> LinearExpr:
            return mdl.sum(p[i][j] * x[j, k] for j in range(n))

        # (4) C_{1,1} = sum_j p_{1,j} x_{j,1}
        # zero-based: C[0,0] = sum_j p[0][j] x[j,0]
        mdl.add_constraint(C[0, 0] == proc_expr(0, 0), ctname="C_0_0_def")

        # (5) C_{1,k} = C_{1,k-1} + sum_j p_{1,j} x_{j,k},  k=2..n
        # zero-based: k=1..n-1
        for k in range(1, n):
            mdl.add_constraint(
                C[0, k] == C[0, k - 1] + proc_expr(0, k),
                ctname=f"C_0_{k}_def",
            )

        # (6) C_{i,1} = C_{i-1,1} + sum_j p_{i,j} x_{j,1},  i=2..m
        # zero-based: i=1..m-1 at k=0
        for i in range(1, m):
            mdl.add_constraint(
                C[i, 0] == C[i - 1, 0] + proc_expr(i, 0),
                ctname=f"C_{i}_0_def",
            )

        # (7) C_{i,k} >= C_{i-1,k} + sum_j p_{i,j} x_{j,k},  i=2..m, k=1..n
        # zero-based: i=1..m-1, k=0..n-1
        for i in range(1, m):
            for k in range(n):
                mdl.add_constraint(
                    C[i, k] >= C[i - 1, k] + proc_expr(i, k),
                    ctname=f"stage_prec_{i}_{k}",
                )

        # (8) C_{i,k} >= C_{i,k-1} + sum_j p_{i,j} x_{j,k},  i=2..m, k=1..n
        # zero-based: i=1..m-1, k=1..n-1
        for i in range(1, m):
            for k in range(1, n):
                mdl.add_constraint(
                    C[i, k] >= C[i, k - 1] + proc_expr(i, k),
                    ctname=f"machine_prec_{i}_{k}",
                )

        # (9) T_k >= C_{m,k} - sum_j d_j x_{j,k},  k=1..n
        # zero-based: T[k] >= C[m-1,k] - sum_j d[j] x[j,k]
        for k in range(n):
            due_expr = mdl.sum(d[j] * x[j, k] for j in range(n))
            mdl.add_constraint(
                T[k] >= C[m - 1, k] - due_expr,
                ctname=f"tard_{k}",
            )

        # ---- parameters ----
        if time_limit_s is not None:
            mdl.parameters.timelimit = float(time_limit_s)
        if threads is not None:
            mdl.parameters.threads = int(threads)
        if mip_gap is not None:
            mdl.parameters.mip.tolerances.mipgap = float(mip_gap)

        # ---- optional MIP start from incumbent permutation ----
        if incumbent_perm is not None:
            if len(incumbent_perm) != n:
                raise ValueError(
                    f"incumbent_perm length {len(incumbent_perm)} != n {n}"
                )
            if sorted(incumbent_perm) != list(range(n)):
                raise ValueError("incumbent_perm must be a permutation of 0..n-1")

            start = mdl.new_solution()
            for k, j in enumerate(incumbent_perm):
                start.add_var_value(x[j, k], 1)
            mdl.add_mip_start(start)

        return mdl, vars_

    @staticmethod
    def extract_permutation_from_solution(
        sol: SolveSolution, x_vars: dict[tuple[int, int], Any], n: int
    ) -> list[int]:
        """Given a solved model, reconstruct the permutation (job at each position k)."""
        perm = [-1] * n
        for k in range(n):
            chosen = None
            for j in range(n):
                val = sol.get_value(x_vars[j, k])
                if val is not None and val > 0.5:
                    chosen = j
                    break
            if chosen is None:
                raise RuntimeError(f"Could not decode permutation at position k={k}")
            perm[k] = chosen
        return perm
