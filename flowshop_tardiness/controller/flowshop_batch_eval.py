from dataclasses import dataclass
from typing import Sequence

from .obj_val_vector import ObjValVector


@dataclass
class PrecompSubseq:
    c: list[list[int]]
    """
    forward DP completion time of pi (paper's c_{ij}; size m x L)
    """

    cbar: list[list[int]]
    """
    reverse DP completion time of pi (paper's \\bar{c}_{ij}; size m x L)
    """

    cp: list[list[int]]
    """
    direction chosen while computing cbar (paper's cp_{ij})
    cp=0 means we chose "down" (i+1,j), cp=1 means we chose "right" (i,j+1)
    """

    subseq_end: list[list[int]]
    """
    subseq_end[i][pos]: completion time on machine i AFTER processing the whole subseq
    when the subseq is inserted at position pos (pos in [0..L])
    size m x (L+1)
    """

    subseq_last: list[list[int]]
    """
    subseq_last[t][pos]: completion time on LAST machine after t-th job of subseq
    (t in [0..b-1]) when inserted at pos (pos in [0..L])
    size b x (L+1)
    """

    # prefix_tardy[pos]: sum of tardiness of prefix pi[0..pos-1] (pos in [0..L])
    prefix_tardy: list[int]
    """
    AOF[pos]: sum of tardiness of prefix pi[0..pos-1] (pos in [0..L])
    """


class PermutationFlowshopSubseqEvaluator:
    """
    Evaluate insertion of a fixed-order contiguous serial subseq of jobs into a permutation flow shop.

    - pi excludes the subseq jobs (length L = n-b).
    - subseq = [s0, s1, ..., s(b-1)] stays contiguous and in this order.
    - Try all positions pos in [0..L] => (n-b+1) positions.

    Objective: total tardiness (sum of max(0, Cj - dj)).

    Acceleration:
      - Fig.9-style precompute on pi: c, cbar, cp, prefix tardiness.
      - Subseq DP per pos: O(m*b) to get boundary Load after subseq (subseq_end) and subseq_last (for tardiness).
      - Suffix evaluation: generalized Fig.10 boundary-walk using cp and i* starting from Load=subseq_end[:,pos].
    """

    def __init__(self, p: Sequence[Sequence[int]], due: Sequence[int]):
        self.p = p
        self.due = due
        self.m = len(p)
        self.n_jobs = len(due)

    def get_tardiness(self, job: int, completion_time: int) -> int:
        return completion_time - self.due[job] if completion_time > self.due[job] else 0

    # ----------------------------
    # Fig.9-style precompute for pi
    # ----------------------------
    def precompute(self, pi: Sequence[int], subseq: Sequence[int]) -> PrecompSubseq:
        m = self.m
        L = len(pi)
        b = len(subseq)

        # 1) forward DP c: m x L
        c = [[0] * L for _ in range(m)]
        for j in range(L):
            job = pi[j]
            for i in range(m):
                up = c[i - 1][j] if i > 0 else 0
                left = c[i][j - 1] if j > 0 else 0
                c[i][j] = (up if up > left else left) + self.p[i][job]

        # 2) reverse DP cbar/cp: m x L
        cbar_full = [[0] * (L + 1) for _ in range(m + 1)]
        cp_full = [[0] * (L + 1) for _ in range(m + 1)]

        if L > 0:
            last_j = L - 1
            last_job = pi[last_j]
            for i in range(m - 1, -1, -1):
                cbar_full[i][last_j] = cbar_full[i + 1][last_j] + self.p[i][last_job]
                cp_full[i][last_j] = 0

            for j in range(L - 2, -1, -1):
                job = pi[j]

                i = m - 1
                cbar_full[i][j] = cbar_full[i][j + 1] + self.p[i][job]
                cp_full[i][j] = 1

                for i in range(m - 2, -1, -1):
                    down = cbar_full[i + 1][j]
                    right = cbar_full[i][j + 1]
                    if down >= right:
                        cbar_full[i][j] = down + self.p[i][job]
                        cp_full[i][j] = 0
                    else:
                        cbar_full[i][j] = right + self.p[i][job]
                        cp_full[i][j] = 1

        cbar = [row[:L] for row in cbar_full[:m]]
        cp = [row[:L] for row in cp_full[:m]]

        # 3) prefix tardiness on pi
        prefix_tardy = [0] * (L + 1)
        for t in range(1, L + 1):
            job = pi[t - 1]
            C_last = c[m - 1][t - 1]
            prefix_tardy[t] = prefix_tardy[t - 1] + self.get_tardiness(job, C_last)

        # 4) subseq completion DP for each pos in [0..L]
        subseq_end = [[0] * (L + 1) for _ in range(m)]
        subseq_last = [[0] * (L + 1) for _ in range(b)]

        for pos in range(L + 1):
            left_boundary = [c[i][pos - 1] if pos > 0 else 0 for i in range(m)]
            F_prev_job = [0] * m

            for t, job in enumerate(subseq):
                F_curr_job = [0] * m
                for i in range(m):
                    left = left_boundary[i] if t == 0 else F_prev_job[i]
                    up = F_curr_job[i - 1] if i > 0 else 0
                    start = left if left > up else up
                    F_curr_job[i] = start + self.p[i][job]
                subseq_last[t][pos] = F_curr_job[m - 1]
                F_prev_job = F_curr_job

            for i in range(m):
                subseq_end[i][pos] = F_prev_job[i]

        return PrecompSubseq(
            c=c,
            cbar=cbar,
            cp=cp,
            subseq_end=subseq_end,
            subseq_last=subseq_last,
            prefix_tardy=prefix_tardy,
        )

    # ----------------------------
    # i* for subseq (subseq_end + cbar)
    # ----------------------------
    def find_i_star_subseq(self, pre: PrecompSubseq, pos: int) -> tuple[int, int]:
        m = self.m
        L = len(pre.c[0]) if m > 0 else 0

        best_val = -1
        i_star = 0
        for i in range(m):
            suffix = pre.cbar[i][pos] if pos < L else 0
            cand = pre.subseq_end[i][pos] + suffix
            if cand > best_val or (cand == best_val and i > i_star):
                best_val = cand
                i_star = i
        return i_star, best_val

    # ----------------------------
    # Generalized Fig.10 boundary-walk for suffix from arbitrary Load
    # ----------------------------
    def _suffix_tardiness_fig10_like(
        self,
        pi: Sequence[int],
        start_pos: int,
        Load_init: Sequence[int],
        i_star: int,
        cp: Sequence[Sequence[int]],
    ) -> int:
        """
        Compute tardiness of suffix pi[start_pos..] given:
          - Load_init[i]: completion time on machine i right BEFORE starting suffix (after subseq).
          - i_star: critical machine at the start boundary (from subseq_end + cbar).
          - cp: direction table for pi (size m x L).

        This is the single-σ Fig.10 logic, generalized to "start from arbitrary boundary state"
        by treating the boundary state as the column at index start_pos (like sigma column),
        and shifting suffix columns by +1 relative to pi indices.

        Returns: total tardiness of suffix jobs.
        """
        m = self.m
        L = len(pi)

        if start_pos >= L:
            return 0

        # Load is updated in-place; copy for safety
        Load = list(Load_init)  # size m

        # We will store only what Fig.10 writes, on columns t=start_pos..L
        # C[:, start_pos] is the boundary state (after subseq).
        C = [[0] * (L + 1) for _ in range(m)]
        for i in range(m):
            C[i][start_pos] = Load[i]

        # Equivalent of Fig.10 "if j1 <= Length" pre-update for the first suffix job on i*
        first_job = pi[start_pos]
        Load[i_star] = Load[i_star] + self.p[i_star][first_job]
        C[i_star][start_pos + 1] = Load[i_star]

        # Main loop: j = start_pos .. L-1
        for j in range(start_pos, L):
            job = pi[j]

            if cp[i_star][j] == 1:
                # Update machines below boundary for current job
                for i in range(i_star + 1, m):
                    if Load[i - 1] > Load[i]:
                        Load[i] = Load[i - 1]
                    Load[i] += self.p[i][job]
                    C[i][j + 1] = Load[i]

                # Pre-update boundary machine for next job
                if (j + 1) < L:
                    next_job = pi[j + 1]
                    Load[i_star] = Load[i_star] + self.p[i_star][next_job]
                    C[i_star][j + 2] = Load[i_star]

            else:
                # Move boundary down at least once (if possible)
                if i_star + 1 < m:
                    Load[i_star + 1] = Load[i_star] + self.p[i_star + 1][job]
                    C[i_star + 1][j + 1] = Load[i_star + 1]
                    i_star += 1

                # Keep moving boundary down while cp indicates vertical continuation
                while i_star < m - 1 and cp[i_star][j] == 0:
                    Load[i_star + 1] = Load[i_star] + self.p[i_star + 1][job]
                    C[i_star + 1][j + 1] = Load[i_star + 1]
                    i_star += 1

                # Finish remaining machines for current job
                for i in range(i_star + 1, m):
                    if Load[i - 1] > Load[i]:
                        Load[i] = Load[i - 1]
                    Load[i] += self.p[i][job]
                    C[i][j + 1] = Load[i]

                # Pre-update boundary machine for next job
                if (j + 1) < L:
                    next_job = pi[j + 1]
                    Load[i_star] = Load[i_star] + self.p[i_star][next_job]
                    C[i_star][j + 2] = Load[i_star]

        # Tardiness of suffix jobs: pi[j] completion at last machine is C[m-1][j+1]
        total = 0
        for j in range(start_pos, L):
            total += self.get_tardiness(pi[j], C[m - 1][j + 1])
        return total

    # ----------------------------
    # Full evaluation: prefix + subseq + accelerated suffix
    # ----------------------------
    def evaluate_position_total_tardiness(
        self,
        pi: Sequence[int],
        subseq: Sequence[int],
        pre: PrecompSubseq,
        pos: int,
        i_star: int,
    ) -> int:
        """
        Total tardiness for:
            pi[0..pos-1] + subseq + pi[pos..]
        computed as:
          - prefix: pre.prefix_tardy[pos]
          - subseq: sum tardiness using pre.subseq_last[:,pos]
          - suffix: generalized Fig.10 boundary-walk from Load=pre.subseq_end[:,pos]
        """
        total = pre.prefix_tardy[pos]

        # subseq tardiness
        for t, job in enumerate(subseq):
            total += self.get_tardiness(job, pre.subseq_last[t][pos])

        # suffix tardiness (accelerated)
        Load_init = [pre.subseq_end[i][pos] for i in range(self.m)]
        total += self._suffix_tardiness_fig10_like(
            pi=pi,
            start_pos=pos,
            Load_init=Load_init,
            i_star=i_star,
            cp=pre.cp,
        )
        return total

    def get_best_position(
        self,
        pi: Sequence[int],
        subseq: Sequence[int] | int,
        tie_breaker: str = "default",
    ) -> tuple[int, int]:
        """Return (best_pos, best_total_tardiness) for inserting subseq into pi.

        Args:
            pi (Sequence[int]): permutation without the subseq jobs
            subseq (Sequence[int] | int): subseq of jobs to insert
            tie_breaker (str, optional): Defaults to "default".
                - "default": minimize total tardiness only
                - "makespan": if tardiness ties, choose smaller makespan

        Raises:
            ValueError: no insertion positions evaluated

        Returns:
            tuple[int, int]: (best_pos, best_total_tardiness)
        """
        _subseq: Sequence[int]
        if isinstance(subseq, int):
            _subseq = [subseq]
        else:
            _subseq = subseq
        pre = self.precompute(pi, _subseq)
        L = len(pi)

        best_pos = 0
        best_obj_vals: ObjValVector | None = None

        for pos in range(L + 1):
            i_star, makespan = self.find_i_star_subseq(pre, pos)

            sum_Tj = self.evaluate_position_total_tardiness(
                pi=pi, subseq=_subseq, pre=pre, pos=pos, i_star=i_star
            )
            if tie_breaker == "makespan":
                obj_vals = ObjValVector(sum_Tj, makespan)
            else:
                obj_vals = ObjValVector(sum_Tj)

            if best_obj_vals is None or obj_vals < best_obj_vals:
                best_pos = pos
                best_obj_vals = obj_vals

        if best_obj_vals is None:
            raise ValueError("No insertion positions evaluated.")
        return best_pos, best_obj_vals.obj1_val
