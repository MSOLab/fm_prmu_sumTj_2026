import logging
import time
from collections import defaultdict
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

        self._timing_enabled = True  # change to True to enable timing
        self._timing_stats: defaultdict[str, int | float] = defaultdict(float)
        self._timing_counts: defaultdict[str, int] = defaultdict(int)

    def log_timing_as_info(self):
        if not self._timing_stats:
            return
        timing_str = "\n==== Evaluator Timing Summary ===="
        keys = sorted(
            self._timing_stats.keys(), key=lambda k: self._timing_stats[k], reverse=True
        )
        for k in keys:
            tot = self._timing_stats[k]
            cnt = self._timing_counts.get(k, 0)
            avg = tot / cnt if cnt else 0.0
            timing_str += f"\n{k:30s} total={tot:10.6f}s  cnt={cnt:6d}  avg={avg:10.6f}s"
        timing_str += "\n==================================="
        logging.info(timing_str)

    def get_tardiness(self, job: int, completion_time: int) -> int:
        return completion_time - self.due[job] if completion_time > self.due[job] else 0

    # ----------------------------
    # Fig.9-style precompute for pi
    # ----------------------------
    def precompute(self, pi: Sequence[int], subseq: Sequence[int]) -> PrecompSubseq:
        m = self.m
        L = len(pi)
        b = len(subseq)
        p: Sequence[Sequence[int]] = self.p

        # 1) forward DP c: m x L
        c = [[0] * L for _ in range(m)]
        for j in range(L):
            job = pi[j]
            for i in range(m):
                up = c[i - 1][j] if i > 0 else 0
                left = c[i][j - 1] if j > 0 else 0
                c[i][j] = (up if up > left else left) + p[i][job]

        # 2) reverse DP cbar/cp: m x L
        cbar_full = [[0] * (L + 1) for _ in range(m + 1)]
        cp_full = [[0] * (L + 1) for _ in range(m + 1)]

        if L > 0:
            last_j = L - 1
            last_job = pi[last_j]
            for i in range(m - 1, -1, -1):
                cbar_full[i][last_j] = cbar_full[i + 1][last_j] + p[i][last_job]
                cp_full[i][last_j] = 0

            for j in range(L - 2, -1, -1):
                job = pi[j]

                i = m - 1
                cbar_full[i][j] = cbar_full[i][j + 1] + p[i][job]
                cp_full[i][j] = 1

                for i in range(m - 2, -1, -1):
                    down = cbar_full[i + 1][j]
                    right = cbar_full[i][j + 1]
                    if down >= right:
                        cbar_full[i][j] = down + p[i][job]
                        cp_full[i][j] = 0
                    else:
                        cbar_full[i][j] = right + p[i][job]
                        cp_full[i][j] = 1

        cbar = [row[:L] for row in cbar_full[:m]]
        cp = [row[:L] for row in cp_full[:m]]

        # 3) prefix tardiness on pi
        prefix_tardy = [0] * (L + 1)
        for t in range(1, L + 1):
            job = pi[t - 1]
            C_last = c[m - 1][t - 1]
            dj = self.due[job]
            prefix_tardy[t] = prefix_tardy[t - 1] + (C_last - dj if C_last > dj else 0)

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
                    F_curr_job[i] = start + p[i][job]
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

        Optimized version:
        - DO NOT allocate/store full completion-time matrix C.
        - Accumulate tardiness on-the-fly using Load[m-1] when each suffix job finishes.
        """
        m = self.m
        L = len(pi)
        p: Sequence[Sequence[int]] = self.p

        if start_pos >= L:
            return 0

        # Copy boundary completion times into Load (we will update in-place)
        Load = list(Load_init)  # size m

        obj_val = 0

        # --- Pre-update for the first suffix job on boundary machine i_star (same logic as before)
        # This is exactly what your original code did before entering the main loop.
        first_job = pi[start_pos]
        Load[i_star] = Load[i_star] + p[i_star][first_job]

        # Main loop: j = start_pos .. L-1
        for j in range(start_pos, L):
            job = pi[j]
            dj = self.due[job]

            if cp[i_star][j] == 1:
                # Update machines below boundary for current job
                for i in range(i_star + 1, m):
                    if Load[i - 1] > Load[i]:
                        Load[i] = Load[i - 1]
                    Load[i] += p[i][job]

                # Current job completes on last machine now -> accumulate tardiness
                C_last = Load[m - 1]
                obj_val += C_last - dj if C_last > dj else 0

                # Pre-update boundary machine for next job
                if (j + 1) < L:
                    next_job = pi[j + 1]
                    Load[i_star] = Load[i_star] + p[i_star][next_job]

            else:
                # Move boundary down at least once (if possible)
                if i_star + 1 < m:
                    Load[i_star + 1] = Load[i_star] + p[i_star + 1][job]
                    i_star += 1

                # Keep moving boundary down while cp indicates vertical continuation
                while i_star < m - 1 and cp[i_star][j] == 0:
                    Load[i_star + 1] = Load[i_star] + p[i_star + 1][job]
                    i_star += 1

                # Finish remaining machines for current job
                for i in range(i_star + 1, m):
                    if Load[i - 1] > Load[i]:
                        Load[i] = Load[i - 1]
                    Load[i] += p[i][job]

                # Current job completes on last machine now -> accumulate tardiness
                C_last = Load[m - 1]
                obj_val += C_last - dj if C_last > dj else 0

                # Pre-update boundary machine for next job
                if (j + 1) < L:
                    next_job = pi[j + 1]
                    Load[i_star] = Load[i_star] + p[i_star][next_job]

        return obj_val

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
        obj_val = pre.prefix_tardy[pos]

        # subseq tardiness
        for t, job in enumerate(subseq):
            dj: int = self.due[job]
            C_last: int = pre.subseq_last[t][pos]
            obj_val += C_last - dj if C_last > dj else 0

        # suffix tardiness (accelerated)
        Load_init: list[int] = [row[pos] for row in pre.subseq_end]
        obj_val += self._suffix_tardiness_fig10_like(
            pi=pi,
            start_pos=pos,
            Load_init=Load_init,
            i_star=i_star,
            cp=pre.cp,
        )
        return obj_val

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

        timing_enabled = getattr(self, "_timing_enabled", False)
        if timing_enabled:
            stats = self._timing_stats
            counts = self._timing_counts
            t_all0 = time.perf_counter()

        # (1) precompute
        if timing_enabled:
            t0 = time.perf_counter()
        pre = self.precompute(pi, _subseq)
        if timing_enabled:
            stats["precompute"] += time.perf_counter() - t0
            counts["precompute"] += 1

        # (2) position loop
        if timing_enabled:
            t0 = time.perf_counter()
        best_pos = 0
        best_obj_vals: ObjValVector | None = None
        L = len(pi)
        for pos in range(L + 1):
            # (2-a) i_star 찾기
            if timing_enabled:
                t1 = time.perf_counter()
            i_star, makespan = self.find_i_star_subseq(pre, pos)
            if timing_enabled:
                stats["find_i_star_subseq"] += time.perf_counter() - t1
                counts["find_i_star_subseq"] += 1

            # (2-b) objective 평가
            if timing_enabled:
                t1 = time.perf_counter()
            sum_Tj = self.evaluate_position_total_tardiness(
                pi=pi, subseq=_subseq, pre=pre, pos=pos, i_star=i_star
            )
            if tie_breaker == "makespan":
                obj_vals = ObjValVector(sum_Tj, makespan)
            else:
                obj_vals = ObjValVector(sum_Tj)
            if timing_enabled:
                stats["eval_total_tardiness"] += time.perf_counter() - t1
                counts["eval_total_tardiness"] += 1

            if best_obj_vals is None or obj_vals < best_obj_vals:
                best_pos = pos
                best_obj_vals = obj_vals

        if timing_enabled:
            stats["pos_loop_total"] += time.perf_counter() - t0
            counts["pos_loop_total"] += 1

            stats["get_best_position_total"] += time.perf_counter() - t_all0
            counts["get_best_position_total"] += 1

        if best_obj_vals is None:
            raise ValueError("No insertion positions evaluated.")
        return best_pos, best_obj_vals.obj1_val
