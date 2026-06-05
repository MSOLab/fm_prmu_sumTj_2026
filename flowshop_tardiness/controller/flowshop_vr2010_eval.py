"""Single-job insertion evaluators for GAPR's selectable insertion speedup.

GAPR (Vallada & Ruiz, 2010) always inserts exactly one job during NEH
construction and local search. This module provides the two non-default
speedup modes that mirror that paper:

- ``VR2010InsertionEvaluator``: the paper's §3.3 (p.59) *prefix bookkeeping*.
  The prefix completion times before the insertion point are carried over
  between positions, but the suffix is recomputed at every position. Cost per
  full position sweep is ``O(L^2 m / 2)`` -- a constant-factor saving over the
  naive baseline, not the asymptotic improvement of Fernandez-Viagas et al.
  (2020), which is the point of offering it as a faithful-reproduction option.
- ``NaiveInsertionEvaluator``: no speedup at all. Every candidate position is
  scored by a full ``O(L m)`` completion-time DP, giving ``O(L^2 m)`` per sweep.

Both share the public contract of
:class:`~flowshop_tardiness.controller.flowshop_batch_eval.PermutationFlowshopSubseqEvaluator`::

    get_best_position(pi, subseq, tie_breaker="default")
        -> (best_pos_list, best_total_tardiness)

where ``best_pos_list`` is ascending and ``best_pos_list[0]`` is the smallest
position attaining the minimum total tardiness (the only entry GAPR reads).
Only single-job insertion and ``tie_breaker="default"`` are supported; anything
else raises ``NotImplementedError`` (YAGNI -- makespan tie-breaking is used by
CP-LNS, not GAPR).
"""

from typing import Sequence


def _single_job(subseq: Sequence[int] | int) -> int:
    """Return the one job index from a length-1 subseq (or a bare int)."""
    if isinstance(subseq, int):
        return subseq
    if len(subseq) != 1:
        raise NotImplementedError(
            "single-job insertion evaluators support len(subseq) == 1 only"
        )
    return subseq[0]


class VR2010InsertionEvaluator:
    """Prefix-bookkeeping insertion evaluator (Vallada & Ruiz, 2010 §3.3)."""

    def __init__(self, p: Sequence[Sequence[int]], due: Sequence[int]):
        """
        Args:
            p (Sequence[Sequence[int]]): processing times ``p[i][j]`` for
                machine ``i`` and job index ``j``.
            due (Sequence[int]): due date ``due[j]`` for job index ``j``.
        """
        self.p = p
        self.due = due
        self.m = len(p)

    def get_best_position(
        self,
        pi: Sequence[int],
        subseq: Sequence[int] | int,
        tie_breaker: str = "default",
    ) -> tuple[list[int], int]:
        """Best single-job insertion position via prefix bookkeeping.

        Args:
            pi (Sequence[int]): permutation (job indices) excluding the
                inserted job.
            subseq (Sequence[int] | int): the single job to insert.
            tie_breaker (str, optional): only ``"default"`` is supported.

        Returns:
            tuple[list[int], int]: ``(best_pos_list, best_total_tardiness)``.
        """
        if tie_breaker != "default":
            raise NotImplementedError("vr2010 supports tie_breaker='default' only")
        sigma = _single_job(subseq)
        p, due, m, L = self.p, self.due, self.m, len(pi)

        # Completion times on each machine of the last job of the current prefix
        # pi[:pos] and the running tardiness of that prefix. Both are reused
        # across positions; only the suffix is recomputed each time.
        c_prefix = [0] * m
        prefix_tard = 0
        best_pos_list: list[int] = []
        best_obj: int | None = None

        for pos in range(L + 1):
            # (1) Insert sigma right after the prefix -- O(m).
            c = [0] * m
            c[0] = c_prefix[0] + p[0][sigma]
            for i in range(1, m):
                prev_m = c_prefix[i] if c_prefix[i] > c[i - 1] else c[i - 1]
                c[i] = prev_m + p[i][sigma]
            total = prefix_tard + (
                c[m - 1] - due[sigma] if c[m - 1] > due[sigma] else 0
            )

            # (2) Recompute the suffix pi[pos:] from sigma's completion -- O((L-pos) m).
            prev = c
            for job in pi[pos:]:
                cur = [0] * m
                cur[0] = prev[0] + p[0][job]
                for i in range(1, m):
                    prev_m = prev[i] if prev[i] > cur[i - 1] else cur[i - 1]
                    cur[i] = prev_m + p[i][job]
                if cur[m - 1] > due[job]:
                    total += cur[m - 1] - due[job]
                prev = cur

            # Track best (collect tied positions in ascending order).
            if best_obj is None or total < best_obj:
                best_obj, best_pos_list = total, [pos]
            elif total == best_obj:
                best_pos_list.append(pos)

            # (3) Absorb pi[pos] into the prefix for the next iteration -- O(m).
            if pos < L:
                job = pi[pos]
                nc = [0] * m
                nc[0] = c_prefix[0] + p[0][job]
                for i in range(1, m):
                    prev_m = c_prefix[i] if c_prefix[i] > nc[i - 1] else nc[i - 1]
                    nc[i] = prev_m + p[i][job]
                if nc[m - 1] > due[job]:
                    prefix_tard += nc[m - 1] - due[job]
                c_prefix = nc

        if best_obj is None:
            raise ValueError("No insertion positions evaluated.")
        return best_pos_list, best_obj


class NaiveInsertionEvaluator:
    """No-speedup insertion evaluator: full DP recompute at every position."""

    def __init__(self, p: Sequence[Sequence[int]], due: Sequence[int]):
        """
        Args:
            p (Sequence[Sequence[int]]): processing times ``p[i][j]`` for
                machine ``i`` and job index ``j``.
            due (Sequence[int]): due date ``due[j]`` for job index ``j``.
        """
        self.p = p
        self.due = due
        self.m = len(p)

    def _full_total_tardiness(self, seq: Sequence[int]) -> int:
        """Total tardiness of a full sequence via the standard flowshop DP."""
        p, due, m = self.p, self.due, self.m
        prev = [0] * m  # completion times on each machine of the previous job
        total = 0
        for job in seq:
            cur = [0] * m
            cur[0] = prev[0] + p[0][job]
            for i in range(1, m):
                prev_m = prev[i] if prev[i] > cur[i - 1] else cur[i - 1]
                cur[i] = prev_m + p[i][job]
            if cur[m - 1] > due[job]:
                total += cur[m - 1] - due[job]
            prev = cur
        return total

    def get_best_position(
        self,
        pi: Sequence[int],
        subseq: Sequence[int] | int,
        tie_breaker: str = "default",
    ) -> tuple[list[int], int]:
        """Best single-job insertion position via full recompute per position.

        Args:
            pi (Sequence[int]): permutation (job indices) excluding the
                inserted job.
            subseq (Sequence[int] | int): the single job to insert.
            tie_breaker (str, optional): only ``"default"`` is supported.

        Returns:
            tuple[list[int], int]: ``(best_pos_list, best_total_tardiness)``.
        """
        if tie_breaker != "default":
            raise NotImplementedError("none supports tie_breaker='default' only")
        sigma = _single_job(subseq)
        pi = list(pi)
        best_pos_list: list[int] = []
        best_obj: int | None = None
        for pos in range(len(pi) + 1):
            seq = pi[:pos] + [sigma] + pi[pos:]
            total = self._full_total_tardiness(seq)
            if best_obj is None or total < best_obj:
                best_obj, best_pos_list = total, [pos]
            elif total == best_obj:
                best_pos_list.append(pos)
        if best_obj is None:
            raise ValueError("No insertion positions evaluated.")
        return best_pos_list, best_obj
