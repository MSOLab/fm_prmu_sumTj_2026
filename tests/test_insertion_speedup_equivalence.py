"""Equivalence of the three GAPR insertion-speedup evaluators.

``fv2020`` (Fernandez-Viagas et al., 2020), ``vr2010`` (Vallada & Ruiz, 2010
§3.3 prefix bookkeeping) and ``none`` (full recompute) differ only in CPU cost:
for a single-job insertion they must produce the *same* best total tardiness and
the same earliest best position. This pins all three against a brute-force
ground truth across many random instances.
"""

import random
from typing import List, Sequence

import pytest

from flowshop_tardiness.controller.flowshop_batch_eval import (
    PermutationFlowshopSubseqEvaluator,
)
from flowshop_tardiness.controller.flowshop_vr2010_eval import (
    NaiveInsertionEvaluator,
    VR2010InsertionEvaluator,
)


def _full_total_tardiness(
    p: Sequence[Sequence[int]], due: Sequence[int], seq: Sequence[int]
) -> int:
    """Brute-force total tardiness of a full sequence via the standard DP."""
    m = len(p)
    prev = [0] * m
    total = 0
    for job in seq:
        cur = [0] * m
        cur[0] = prev[0] + p[0][job]
        for i in range(1, m):
            start = prev[i] if prev[i] > cur[i - 1] else cur[i - 1]
            cur[i] = start + p[i][job]
        if cur[m - 1] > due[job]:
            total += cur[m - 1] - due[job]
        prev = cur
    return total


def _brute_force_best(
    p: Sequence[Sequence[int]],
    due: Sequence[int],
    pi: Sequence[int],
    sigma: int,
) -> tuple[int, int]:
    """Ground truth (best_total_tardiness, earliest_best_pos) by enumeration."""
    pi = list(pi)
    best_obj: int | None = None
    best_pos = 0
    for pos in range(len(pi) + 1):
        seq = pi[:pos] + [sigma] + pi[pos:]
        total = _full_total_tardiness(p, due, seq)
        if best_obj is None or total < best_obj:
            best_obj = total
            best_pos = pos
    assert best_obj is not None
    return best_obj, best_pos


def _random_instance(
    rng: random.Random, n: int, m: int, tightness: str
) -> tuple[List[List[int]], List[int]]:
    """Random (p, due). ``tightness`` controls how binding the due dates are."""
    p = [[rng.randint(1, 20) for _ in range(n)] for _ in range(m)]
    # A loose lower bound on each job's completion: sum of its processing times.
    horizon = sum(max(p[i][j] for i in range(m)) for j in range(n))
    due = []
    for j in range(n):
        if tightness == "tight":
            hi = max(1, horizon // 3)
        else:  # loose
            hi = max(1, horizon)
        due.append(rng.randint(1, hi))
    return p, due


# (n_total_jobs, m_machines) covering single/multi machine and varying lengths.
_SHAPES = [(2, 1), (3, 2), (4, 3), (6, 1), (6, 4), (8, 2), (10, 5), (12, 3)]
_TIGHTNESS = ["tight", "loose"]


@pytest.mark.parametrize("n,m", _SHAPES)
@pytest.mark.parametrize("tightness", _TIGHTNESS)
def test_three_evaluators_match_brute_force(n: int, m: int, tightness: str) -> None:
    rng = random.Random(1000 * n + 17 * m + (0 if tightness == "tight" else 1))
    for _ in range(40):
        p, due = _random_instance(rng, n, m, tightness)

        order = list(range(n))
        rng.shuffle(order)
        sigma = order[0]
        pi = order[1:]  # permutation excluding the job to insert

        gt_obj, gt_pos = _brute_force_best(p, due, pi, sigma)

        fv = PermutationFlowshopSubseqEvaluator(p, due)
        vr = VR2010InsertionEvaluator(p, due)
        nv = NaiveInsertionEvaluator(p, due)

        fv_pos, fv_obj = fv.get_best_position(pi, [sigma])
        vr_pos, vr_obj = vr.get_best_position(pi, [sigma])
        nv_pos, nv_obj = nv.get_best_position(pi, [sigma])

        # Best objective: all three agree with the ground truth.
        assert fv_obj == gt_obj, ("fv2020", p, due, pi, sigma)
        assert vr_obj == gt_obj, ("vr2010", p, due, pi, sigma)
        assert nv_obj == gt_obj, ("none", p, due, pi, sigma)

        # Earliest best position (the only entry GAPR reads) agrees.
        assert fv_pos[0] == gt_pos
        assert vr_pos[0] == gt_pos
        assert nv_pos[0] == gt_pos


def test_bare_int_subseq_supported() -> None:
    """Callers may pass the job as a bare int instead of a length-1 list."""
    p = [[3, 2, 4, 6], [5, 1, 3, 2], [4, 2, 1, 3]]
    due = [11, 7, 13, 9]
    pi = [0, 2, 1]
    sigma = 3
    for evaluator in (
        VR2010InsertionEvaluator(p, due),
        NaiveInsertionEvaluator(p, due),
    ):
        list_pos, list_obj = evaluator.get_best_position(pi, [sigma])
        int_pos, int_obj = evaluator.get_best_position(pi, sigma)
        assert list_pos == int_pos
        assert list_obj == int_obj


def test_multi_job_subseq_rejected() -> None:
    p = [[3, 2, 4, 6], [5, 1, 3, 2]]
    due = [11, 7, 13, 9]
    for evaluator in (
        VR2010InsertionEvaluator(p, due),
        NaiveInsertionEvaluator(p, due),
    ):
        with pytest.raises(NotImplementedError):
            evaluator.get_best_position([0, 1], [2, 3])


def test_makespan_tie_breaker_rejected() -> None:
    p = [[3, 2, 4, 6], [5, 1, 3, 2]]
    due = [11, 7, 13, 9]
    for evaluator in (
        VR2010InsertionEvaluator(p, due),
        NaiveInsertionEvaluator(p, due),
    ):
        with pytest.raises(NotImplementedError):
            evaluator.get_best_position([0, 1], [2], tie_breaker="makespan")
